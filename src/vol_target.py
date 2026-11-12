"""Frozen Task 14 leverage rules, metrics, and exact circular block inference."""

import math

import numpy as np
import pandas as pd

from .multi_asset import account_returns


statistic_names = (
    "annual_total_return", "annual_excess_return", "vol", "Sharpe", "MDD", "MDDn", "D", "ES1n", "ES5m",
    "worst_day", "worst_month", "share_high_rv", "max_year_vol_ratio", "log_vol_target",
    "mean_gross", "median_gross", "min_gross", "max_gross", "trades_per_year", "turnover_per_year",
    "trading_cost_per_year", "financing_cost_per_year", "n", "n_months", "n_rv_months",
)
comparison_names = ("R_D", "T", "dS", "T_m", "R_MDD")


def base_returns(prices):
    if len(prices.columns) != 2:
        raise ValueError("Base book requires equity and bond columns, in that order")
    returns = prices.pct_change(fill_method=None)
    return (returns.iloc[:, 0] * .6 + returns.iloc[:, 1] * .4).dropna().rename("base")


def forecast_vol(base, lam=.94):
    if not 0 < lam < 1 or not np.isfinite(base.to_numpy()).all():
        raise ValueError("Invalid EWMA input")
    return np.sqrt(252 * base.pow(2).ewm(alpha=1 - lam, adjust=False).mean()).rename("sigma_hat")


def target_leverage(sigma, target, first_signal, cap=3., band=.10):
    if target <= 0 or cap < 0 or band < 0:
        raise ValueError("Invalid target, cap, or band")
    usable = sigma.loc[first_signal:]
    if usable.empty or usable.index[0] != first_signal or usable.isna().any() or usable.lt(0).any():
        raise ValueError("No valid forecast at the first signal")
    with np.errstate(divide="ignore"):
        desired = (target / usable).clip(lower=0, upper=cap)
    held, previous = [], None
    for value in desired:
        move = np.inf if previous is None or (previous == 0 and value != 0) else (0 if previous == 0 else abs(value / previous - 1))
        if previous is None or move > band:
            previous = float(value)
        held.append(previous)
    return pd.DataFrame({"lstar": desired, "ltilde": held}, index=usable.index)


def book_weights(calendar, leverage, first_signal, columns):
    if first_signal not in calendar:
        raise ValueError("First signal is absent from the trading calendar")
    position = calendar.get_loc(first_signal) + 1
    if position >= len(calendar):
        raise ValueError("First signal cannot execute inside the fold")
    executed = leverage.reindex(calendar).shift(1).ffill().iloc[position:]
    if executed.isna().any() or executed.lt(0).any():
        raise ValueError("Invalid executable leverage")
    return pd.DataFrame({columns[0]: executed * .6, columns[1]: executed * .4})


def levered_account_returns(prices, cash, weights, spread):
    if spread < 0 or weights.lt(0).any().any():
        raise ValueError("No short positions or negative financing spread")
    frame = account_returns(prices[list(weights.columns)], cash, weights)
    financing = (frame.gross_weight - 1).clip(lower=0) * spread / 252
    frame["total"] = frame.total - financing
    frame["excess"] = frame.excess - financing
    frame["cost"] = frame.cost + financing
    frame["financing_cost"] = financing
    frame["held_gross"] = frame.gross_weight
    frame["initial_entry"] = False
    frame.iloc[0, frame.columns.get_loc("initial_entry")] = True
    frame["trade_date"] = frame.turnover.gt(1e-12) & ~frame.initial_entry
    change = weights.sum(axis=1).diff()
    change.iloc[0] = weights.sum(axis=1).iloc[0]
    frame["leverage_change"] = change
    return frame


def matched_constant(prices, cash, unit_weights, target_frame, spread=.015, cap=3.):
    """One hindsight risk match per book/fold; never re-matched in a bootstrap."""
    unit = levered_account_returns(prices, cash, unit_weights, 0.)
    target_sd = target_frame.excess.std(ddof=1)
    if not np.isfinite(target_sd) or target_sd <= 0:
        raise ValueError("Matching requires positive excess-return volatility")

    def ratio(leverage):
        values = leverage * unit.excess - (leverage * unit.held_gross - 1).clip(lower=0) * spread / 252
        return values.std(ddof=1) / target_sd

    low, high = 0., 1.
    while ratio(high) < 1:
        if cap is not None and high >= cap:
            raise ValueError("Matched constant is infeasible under the gross cap")
        high = high * 2 if cap is None else min(high * 2, cap)
    for _ in range(100):
        middle = (low + high) / 2
        value = ratio(middle)
        if abs(value - 1) <= 1e-9:
            break
        if value < 1:
            low = middle
        else:
            high = middle
    frame = levered_account_returns(prices, cash, unit_weights * middle, spread)
    achieved = frame.excess.std(ddof=1) / target_sd
    if abs(achieved - 1) > 1e-6:
        raise ValueError("Matched constant missed the registered tolerance")
    return middle, frame, achieved


def _ratio(numerator, denominator):
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    return np.divide(numerator, denominator, out=np.full(numerator.shape, np.nan, dtype=float), where=denominator != 0)


def _metric_arrays(values, counts, years, targets):
    """Exact metrics for columns of concatenated daily books, with month slots."""
    total, excess, cash, gross, turnover, trading, financing, trades, initial = np.moveaxis(values, 2, 0)
    n = len(total)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    sd = excess.std(axis=0, ddof=1)
    vol = sd * np.sqrt(252)
    total_sd = total.std(axis=0, ddof=1) * np.sqrt(252)
    total_equity = np.vstack([np.ones(total.shape[1]), np.cumprod(1 + total, axis=0)])
    mdd = (1 - total_equity / np.maximum.accumulate(total_equity, axis=0)).max(axis=0)
    month_square = np.add.reduceat(excess ** 2, starts, axis=0)
    rv = np.sqrt(252 * month_square / counts[:, None])
    valid_months = counts >= 15
    with np.errstate(divide="ignore", invalid="ignore"):
        dispersion = np.log(rv[valid_months]).std(axis=0, ddof=1) if valid_months.sum() >= 2 else np.full(total.shape[1], np.nan)
    month_excess = np.multiply.reduceat(1 + total, starts, axis=0) - np.multiply.reduceat(1 + cash, starts, axis=0)
    k_daily, k_month = math.ceil(.01 * n), math.ceil(.05 * len(counts))
    es1 = -np.partition(excess, k_daily - 1, axis=0)[:k_daily].mean(axis=0)
    es5 = -np.partition(month_excess, k_month - 1, axis=0)[:k_month].mean(axis=0)
    year_starts = np.r_[0, np.flatnonzero(np.diff(years)) + 1]
    year_counts = np.add.reduceat(counts, year_starts)
    year_rv = np.sqrt(252 * np.add.reduceat(month_square, year_starts, axis=0) / year_counts[:, None])
    maximum_year = year_rv[year_counts >= 126].max(axis=0) if (year_counts >= 126).any() else np.full(total.shape[1], np.nan)
    # C4/C5 exclusions remain attached to the original initial execution day.
    active = gross[~initial[:, 0].astype(bool)]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_target = np.log(vol / targets)
    result = {
        "annual_total_return": np.expm1(np.log1p(total).sum(axis=0) * 252 / n),
        "annual_excess_return": np.expm1(np.log1p(excess).sum(axis=0) * 252 / n),
        "vol": vol, "Sharpe": _ratio(excess.mean(axis=0), sd) * np.sqrt(252), "MDD": mdd,
        "MDDn": _ratio(mdd, total_sd), "D": dispersion, "ES1n": _ratio(es1, sd),
        "ES5m": _ratio(es5, sd * np.sqrt(21)), "worst_day": excess.min(axis=0),
        "worst_month": month_excess.min(axis=0),
        "share_high_rv": (rv[valid_months] > 1.5 * vol).mean(axis=0) if valid_months.any() else np.full(total.shape[1], np.nan),
        "max_year_vol_ratio": _ratio(maximum_year, vol), "log_vol_target": log_target,
        "mean_gross": active.mean(axis=0), "median_gross": np.median(active, axis=0),
        "min_gross": active.min(axis=0), "max_gross": active.max(axis=0),
        "trades_per_year": trades.sum(axis=0) * 252 / n,
        "turnover_per_year": turnover.mean(axis=0) * 252,
        "trading_cost_per_year": trading.mean(axis=0) * 252,
        "financing_cost_per_year": financing.mean(axis=0) * 252,
        "n": np.full(total.shape[1], n), "n_months": np.full(total.shape[1], len(counts)),
        "n_rv_months": np.full(total.shape[1], valid_months.sum()),
    }
    return result


packed_columns = ("total", "excess", "cash_return", "held_gross", "turnover", "trading_cost",
                  "financing_cost", "trade_date", "initial_entry")


def _pack(frames):
    first = next(iter(frames.values()))
    if not all(frame.index.equals(first.index) for frame in frames.values()):
        raise ValueError("All books must share a calendar")
    values = np.stack([frame[list(packed_columns)].to_numpy(dtype=float) for frame in frames.values()], axis=1)
    if not np.isfinite(values).all() or (values[:, :, :2] <= -1).any():
        raise ValueError("Invalid daily series for geometric metrics")
    keys = first.index.to_period("M")
    starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
    months = [np.arange(begin, finish) for begin, finish in zip(starts, np.r_[starts[1:], len(first)])]
    years = first.index[starts].year.to_numpy()
    targets = np.array([frame.attrs.get("target", np.nan) for frame in frames.values()], dtype=float)
    return values, months, years, targets


def metrics(frame, target=None):
    values, months, years, targets = _pack({"book": frame})
    if target is not None:
        targets[0] = target
    output = _metric_arrays(values, np.array([len(month) for month in months]), years, targets)
    return {name: float(output[name][0]) for name in statistic_names}


def comparison(vt, cl):
    return {"R_D": float(_ratio(vt["D"], cl["D"])),
            "T": float(1 - _ratio(vt["ES1n"], cl["ES1n"])), "dS": vt["Sharpe"] - cl["Sharpe"],
            "T_m": float(1 - _ratio(vt["ES5m"], cl["ES5m"])),
            "R_MDD": float(_ratio(vt["MDDn"], cl["MDDn"]))}


def circular_month_draws(months, seed, resamples=5000, block=6):
    starts = np.random.default_rng(seed).integers(0, months, size=(resamples, math.ceil(months / block)))
    return ((starts[:, :, None] + np.arange(block)) % months).reshape(resamples, -1)[:, :months]


def block_bootstrap(frames, seed, resamples=5000, block=6):
    values, months, years, targets = _pack(frames)
    draws = circular_month_draws(len(months), seed, resamples, block)
    names = list(frames)
    output = {name: {key: np.empty(resamples) for key in statistic_names} for name in names}
    counts = np.array([len(month) for month in months])
    for i, selected in enumerate(draws):
        positions = np.concatenate([months[month] for month in selected])
        # Calendar years belong to sequential output month slots, not source labels.
        result = _metric_arrays(values[positions], counts[selected], years, targets)
        for column, name in enumerate(names):
            for key in statistic_names:
                output[name][key][i] = result[key][column]
    comparisons = {}
    for vt, cl in (("vt20", "cl20"), ("vt40", "cl40"), ("vt20_b03", "cl20_b03")):
        if vt in output and cl in output:
            a, b = output[vt], output[cl]
            comparisons[vt + "_vs_" + cl] = {"R_D": _ratio(a["D"], b["D"]),
                                             "T": 1 - _ratio(a["ES1n"], b["ES1n"]),
                                             "dS": a["Sharpe"] - b["Sharpe"],
                                             "T_m": 1 - _ratio(a["ES5m"], b["ES5m"]),
                                             "R_MDD": _ratio(a["MDDn"], b["MDDn"])}
    if "s1l" in output and "cl20" in output:
        comparisons["s1l_vs_cl20"] = {"dS": output["s1l"]["Sharpe"] - output["cl20"]["Sharpe"]}
    return {"books": output, "comparisons": comparisons, "months": draws}


def pooled(points, draws, counts):
    total = sum(counts.values())
    weights = {fold: n / total for fold, n in counts.items()}
    common = set.intersection(*(set(value) for value in points.values()))
    point = {key: sum(weights[fold] * points[fold][key] for fold in weights) for key in sorted(common)}
    samples = {key: sum(weights[fold] * draws[fold][key] for fold in weights) for key in sorted(common)}
    return point, samples, weights


def verdict_step1(difference_ci_low, max_gross):
    return "ADOPT S1" if difference_ci_low > 0 and max_gross <= 3. else "PLANT STAYS 60/40"


def verdict_step2(main_rd_high, fresh_rd_high, pooled_t, pooled_ds, pooled_ds_low):
    k1 = main_rd_high < 1 and fresh_rd_high < 1
    k2, k3 = pooled_t >= -.05, pooled_ds >= -.10
    if not (k1 and k2 and k3):
        return "NO BENEFIT", "drop"
    return ("ADDS VALUE" if pooled_ds_low > 0 else "REGULATES RISK"), "keep"
