"""Task 14 gates, fixed books, exact inference, report-only checks and outputs."""

from contextlib import redirect_stdout
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import vol_target as vt
from src.multi_asset import month_ends, target_weights
from src.multi_asset_data import cash_returns
from src.vol_target_data import (g1_check, g2_check, g3_check, load_fresh, load_main, main_execution, main_signal)


results_dir = Path(__file__).resolve().parents[1] / "results"
core_books = ("base", "cl20", "vt20", "cl40", "vt40")
deviations = ["D1: G2 amended (section 13, A1) after the registered gate failed"]
resamples = 5000


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        text.encode("ascii")
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def ci(draws):
    values = np.asarray(draws)
    values = values[np.isfinite(values)]
    return tuple(np.percentile(values, [2.5, 97.5])) if len(values) else (np.nan, np.nan)


def first_fresh_signal(prices):
    base = vt.base_returns(prices)
    count = base.notna().cumsum().reindex(prices.index, fill_value=0)
    qualifying = prices.index[month_ends(prices.index) & count.ge(252)]
    if not len(qualifying):
        raise ValueError("Fresh fold has no qualifying month-end signal")
    return qualifying[0]


def unit_weights(prices, signal):
    path = pd.Series(1., index=prices.loc[signal:].index)
    return vt.book_weights(prices.index, path, signal, list(prices.columns))


def build_fold(prices, cash, signal):
    base = vt.base_returns(prices)
    sigma = vt.forecast_vol(base)
    unit = unit_weights(prices, signal)
    weights, frames, matches = {"base": unit}, {}, {}
    frames["base"] = vt.levered_account_returns(prices, cash, unit, .015)
    inputs = prices.pct_change(fill_method=None).loc[signal:].copy()
    inputs.columns = ["r_equity", "r_bond"]
    inputs["cash_return"] = cash.reindex(inputs.index)
    inputs["sigma_hat"] = sigma.reindex(inputs.index)
    for name, target in (("20", .20), ("40", .40)):
        path = vt.target_leverage(sigma, target, signal)
        inputs["lstar_" + name], inputs["ltilde_" + name] = path.lstar, path.ltilde
        weights["vt" + name] = vt.book_weights(prices.index, path.ltilde, signal, list(prices.columns))
        frames["vt" + name] = vt.levered_account_returns(prices, cash, weights["vt" + name], .015)
        frames["vt" + name].attrs["target"] = target
        leverage, frame, ratio = vt.matched_constant(prices, cash, unit, frames["vt" + name])
        frames["cl" + name], weights["cl" + name] = frame, unit * leverage
        matches["cl" + name] = (leverage, ratio)
    for name in ("vt20", "cl20"):
        frames[name + "_b03"] = vt.levered_account_returns(prices, cash, weights[name], .03)
        if name == "vt20":
            frames[name + "_b03"].attrs["target"] = .20
    return dict(prices=prices, cash=cash, signal=signal, frames=frames, weights=weights, matches=matches,
                inputs=inputs, counts=len(unit), unit=unit)


def add_s1(fold, all_prices, all_cash):
    weights = target_weights(all_prices, all_cash)[0]["S1"]
    leverage, frame, ratio = vt.matched_constant(all_prices, all_cash, weights, fold["frames"]["vt20"], cap=None)
    fold["frames"]["s1l"] = frame
    fold["weights"]["s1l"] = weights * leverage
    fold["matches"]["s1l"] = (leverage, ratio)


def point_comparisons(points):
    result = {}
    for a, b in (("vt20", "cl20"), ("vt40", "cl40"), ("vt20_b03", "cl20_b03")):
        if a in points and b in points:
            result[a + "_vs_" + b] = vt.comparison(points[a], points[b])
    if "s1l" in points:
        result["s1l_vs_cl20"] = {"dS": points["s1l"]["Sharpe"] - points["cl20"]["Sharpe"]}
    return result


def calculate(folds):
    for name, fold in folds.items():
        fold["points"] = {book: vt.metrics(frame) for book, frame in fold["frames"].items()}
        fold["comparisons"] = point_comparisons(fold["points"])
        print("Bootstrap %s: 5000 circular six-month resamples" % name, file=sys.stderr, flush=True)
        fold["draws"] = vt.block_bootstrap(fold["frames"], seed=0 if name == "main" else 1)
    main = folds["main"]
    main["splits"] = {}
    for period, mask in (("before_2017", main["frames"]["base"].index < pd.Timestamp("2017-01-01")),
                         ("from_2017", main["frames"]["base"].index >= pd.Timestamp("2017-01-01"))):
        names = (*core_books, "s1l")
        frames = {book: main["frames"][book].loc[mask] for book in names}
        print("Bootstrap main %s: 5000 circular six-month resamples" % period, file=sys.stderr, flush=True)
        points = {book: vt.metrics(frame) for book, frame in frames.items()}
        main["splits"][period] = dict(points=points, comparisons=point_comparisons(points),
                                     draws=vt.block_bootstrap(frames, seed=0))
    counts = {name: fold["counts"] for name, fold in folds.items()}
    pooled_books, pooled_comparisons = {}, {}
    common_books = set.intersection(*(set(fold["points"]) for fold in folds.values()))
    for book in sorted(common_books):
        pooled_books[book] = vt.pooled({name: fold["points"][book] for name, fold in folds.items()},
                                      {name: fold["draws"]["books"][book] for name, fold in folds.items()}, counts)
    common_comparisons = set.intersection(*(set(fold["comparisons"]) for fold in folds.values()))
    for name in sorted(common_comparisons):
        pooled_comparisons[name] = vt.pooled({fold: value["comparisons"][name] for fold, value in folds.items()},
                                             {fold: value["draws"]["comparisons"][name] for fold, value in folds.items()}, counts)
    return pooled_books, pooled_comparisons


def neighbors(folds):
    output = {}
    for label, lam, band in (("lambda_090", .90, .10), ("lambda_097", .97, .10),
                              ("band_000", .94, 0.), ("band_020", .94, .20)):
        output[label] = {}
        for name, fold in folds.items():
            sigma = vt.forecast_vol(vt.base_returns(fold["prices"]), lam)
            path = vt.target_leverage(sigma, .20, fold["signal"], band=band)
            weights = vt.book_weights(fold["prices"].index, path.ltilde, fold["signal"], list(fold["prices"].columns))
            frame = vt.levered_account_returns(fold["prices"], fold["cash"], weights, .015)
            leverage, constant, ratio = vt.matched_constant(fold["prices"], fold["cash"], fold["unit"], frame)
            a, b = vt.metrics(frame, .20), vt.metrics(constant)
            output[label][name] = dict(vt=a, cl=b, comparison=vt.comparison(a, b), L_c=leverage, sd_ratio=ratio)
    return output


def r5_positions(frame, balance):
    active = frame.loc[~frame.initial_entry, "held_gross"]
    positions = {"equity": balance * .6 * active / 3, "bond": balance * .4 * active / 3,
                 "cash": balance * (1 - active / 3)}
    result = {}
    for name, values in positions.items():
        result.update({name + "_median": values.median(), name + "_min": values.min(), name + "_max": values.max()})
    trades = balance * frame.loc[frame.trade_date, "leverage_change"].abs() / 3
    result.update(trades_per_year=len(trades) * 252 / len(frame), trade_median=trades.median(), trade_max=trades.max())
    return result


def summary_rows(folds, pooled_books, pooled_comparisons, nearby):
    rows = []

    def add(fold, period, book, point, draws=None):
        for statistic, value in point.items():
            low, high = ci(draws[statistic]) if draws is not None else (np.nan, np.nan)
            rows.append(dict(fold=fold, period=period, book=book, statistic=statistic, value=value, ci_low=low, ci_high=high))

    for name, fold in folds.items():
        for book, point in fold["points"].items():
            add(name, "all", book, point, fold["draws"]["books"][book])
        for book, point in fold["comparisons"].items():
            add(name, "all", book, point, fold["draws"]["comparisons"][book])
        for book, (leverage, ratio) in fold["matches"].items():
            add(name, "all", book, {"L_c": leverage, "sd_ratio": ratio})
        for book in ("base", "cl20", "vt20"):
            frame = fold["frames"][book]
            for year, sample in frame.groupby(frame.index.year):
                add(name, str(year), book, dict(total_return=(1 + sample.total).prod() - 1,
                                              RV_y=np.sqrt(252 * (sample.excess ** 2).mean()), n_days=len(sample)))
        for book in ("vt20", "cl20"):
            for balance in (200, 2000):
                add(name, "usd_" + str(balance), book, r5_positions(fold["frames"][book], balance))
        for book in ("vt20", "cl20"):
            frame = fold["frames"][book]
            for month, sample in frame.groupby(frame.index.to_period("M")):
                if len(sample) >= 15:
                    add(name, str(month), book, {"RV_m": np.sqrt(252 * (sample.excess ** 2).mean())})
    for period, result in folds["main"]["splits"].items():
        for book, point in result["points"].items():
            add("main", period, book, point, result["draws"]["books"][book])
        for book, point in result["comparisons"].items():
            add("main", period, book, point, result["draws"]["comparisons"][book])
    for book, (point, draws, _) in pooled_books.items():
        add("pooled", "all", book, point, draws)
    for book, (point, draws, weights) in pooled_comparisons.items():
        add("pooled", "all", book, point, draws)
    add("pooled", "all", "fold_weights", weights)
    for label, values in nearby.items():
        for fold, value in values.items():
            for book in ("vt", "cl"):
                add(fold, "all", book + "20_" + label, value[book])
            add(fold, "all", label + "_comparison", value["comparison"])
            add(fold, "all", "cl20_" + label, {"L_c": value["L_c"], "sd_ratio": value["sd_ratio"]})
        weights = next(iter(pooled_comparisons.values()))[2]
        point = {key: sum(weights[fold] * values[fold]["comparison"][key] for fold in weights) for key in vt.comparison_names}
        add("pooled", "all", label + "_comparison", point)
    return pd.DataFrame(rows)


def fmt(value, digits=3):
    return ("%." + str(digits) + "f") % value if np.isfinite(value) else "-"


def print_book_table(label, points, draws, names):
    print(label)
    print("book     total  excess    vol    Sharpe [95% CI]          MDD MDDn     D  ES1n ES5m worst_d worst_m meanL minL maxL trades/y turn/y trade% finance% ln(v/t)")
    for book in names:
        p = points[book]
        low, high = ci(draws[book]["Sharpe"])
        print("%-7s %6.2f %7.2f %6.2f %6.3f [%6.3f,%6.3f] %6.2f %4.2f %5.3f %5.2f %4.2f %7.2f %7.2f %5.2f %4.2f %4.2f %8.2f %6.2f %6.3f %8.3f %7s" %
              (book, 100*p["annual_total_return"], 100*p["annual_excess_return"], 100*p["vol"], p["Sharpe"], low, high,
               100*p["MDD"], p["MDDn"], p["D"], p["ES1n"], p["ES5m"], 100*p["worst_day"], 100*p["worst_month"],
               p["mean_gross"], p["min_gross"], p["max_gross"], p["trades_per_year"], p["turnover_per_year"],
               100*p["trading_cost_per_year"], 100*p["financing_cost_per_year"], fmt(p["log_vol_target"])))
    print("book     medianL high_RV_share max_year_RV/vol n_days n_months n_RV_months")
    for book in names:
        p = points[book]
        print("%-7s %7.3f %13.3f %15.3f %6d %8d %11d" %
              (book, p["median_gross"], p["share_high_rv"], p["max_year_vol_ratio"], p["n"], p["n_months"], p["n_rv_months"]))


def comparison_table(label, folds, combined, name):
    print(label)
    print("fold        R_D [95% CI]           T [95% CI]             dS [95% CI]          T_m [95% CI]           R_MDD [95% CI]")
    for fold in ("main", "fresh", "pooled"):
        if fold == "pooled":
            point, draws, _ = combined[name]
        else:
            point, draws = folds[fold]["comparisons"][name], folds[fold]["draws"]["comparisons"][name]
        cells = ["%6.3f [%6.3f,%6.3f]" % (point[key], *ci(draws[key])) for key in vt.comparison_names]
        print("%-7s " % fold + " ".join(cells))


def step2(folds, combined, name):
    point, draws, _ = combined[name]
    highs = [ci(folds[fold]["draws"]["comparisons"][name]["R_D"])[1] for fold in ("main", "fresh")]
    return vt.verdict_step2(*highs, point["T"], point["dS"], ci(draws["dS"])[0]), highs


def print_report(folds, pooled_books, combined, nearby, charts_missing):
    print("Books: return, vol, MDD, worst returns and annual costs are percent; all other entries are ratios/counts.")
    for fold in ("main", "fresh"):
        names = (*core_books, "s1l") if fold == "main" else core_books
        print_book_table(fold + " all", folds[fold]["points"], folds[fold]["draws"]["books"], names)
    for period, value in folds["main"]["splits"].items():
        print_book_table("main " + period, value["points"], value["draws"]["books"], (*core_books, "s1l"))
    print("Matched constants: L_c is fixed before bootstrap; sd ratio is constant/target excess sd.")
    for name, fold in folds.items():
        for book, (leverage, ratio) in fold["matches"].items():
            print("%-5s %-5s L_c=%.9f sd_ratio=%.9f" % (name, book, leverage, ratio))
    weights = combined["vt20_vs_cl20"][2]
    print("Pooled weights: main=%.9f fresh=%.9f" % (weights["main"], weights["fresh"]))
    comparison_table("Step 2: vt20 vs cl20", folds, combined, "vt20_vs_cl20")
    (verdict, decision), highs = step2(folds, combined, "vt20_vs_cl20")
    point, draws, _ = combined["vt20_vs_cl20"]
    print("K1: main upper R_D=%.6f < 1: %s; fresh upper R_D=%.6f < 1: %s" %
          (highs[0], highs[0] < 1, highs[1], highs[1] < 1))
    print("K2: pooled T=%.6f >= -0.05: %s" % (point["T"], point["T"] >= -.05))
    print("K3: pooled dS=%.6f >= -0.10: %s; lower 95%% dS=%.6f" % (point["dS"], point["dS"] >= -.10, ci(draws["dS"])[0]))
    print("Step 2 (vt20 vs cl20): %s -> %s" % (verdict, decision))
    main = folds["main"]
    interval = ci(main["draws"]["comparisons"]["s1l_vs_cl20"]["dS"])
    p, q = main["points"]["s1l"], main["points"]["cl20"]
    print("Step 1: Sharpe s1l=%.6f cl20=%.6f difference=%.6f 95%% CI [%.6f,%.6f] max_s1l_gross=%.6f" %
          (p["Sharpe"], q["Sharpe"], p["Sharpe"]-q["Sharpe"], *interval, p["max_gross"]))
    print("Step 1 (s1l vs cl20): " + vt.verdict_step1(interval[0], p["max_gross"]))
    comparison_table("R1 report-only, secondary: vt40 vs cl40", folds, combined, "vt40_vs_cl40")
    result, _ = step2(folds, combined, "vt40_vs_cl40")
    print("R1 report-only secondary rule: %s -> %s" % result)
    print("R2 report-only, points only: four fixed neighbors")
    print("neighbor    fold       L_c sd_ratio     vol  Sharpe     R_D       T      dS     T_m   R_MDD")
    for label, values in nearby.items():
        for name, value in values.items():
            p = value["comparison"]
            print("%-11s %-5s %8.5f %8.6f %7.4f %7.4f " % (label, name, value["L_c"], value["sd_ratio"], value["vt"]["vol"], value["vt"]["Sharpe"]) +
                  " ".join("%7.4f" % p[key] for key in vt.comparison_names))
        p = {key: sum(weights[fold]*values[fold]["comparison"][key] for fold in weights) for key in vt.comparison_names}
        print("%-11s pooled %8s %8s %7s %7s " % (label, "-", "-", "-", "-") + " ".join("%7.4f" % p[key] for key in vt.comparison_names))
    print("R3 report-only: b=0.03, positions and original matched L_c fixed, no re-matching")
    for name, fold in folds.items():
        print_book_table("R3 " + name, fold["points"], fold["draws"]["books"], ("cl20_b03", "vt20_b03"))
        ratio = fold["frames"]["cl20_b03"].excess.std(ddof=1) / fold["frames"]["vt20_b03"].excess.std(ddof=1)
        print("R3 %s resulting sd_ratio constant/target=%.9f" % (name, ratio))
    comparison_table("R3 report-only stressed comparisons", folds, combined, "vt20_b03_vs_cl20_b03")
    print("R4 report-only: main before_2017/from_2017 book statistics are printed above.")
    print("R5 report-only: calendar-year total return and RMS excess realized vol (percent), all years including partial years")
    print("fold  year    base_n base_ret base_RV cl20_n cl20_ret cl20_RV vt20_n vt20_ret vt20_RV")
    for name, fold in folds.items():
        base = fold["frames"]["base"]
        for year in base.index.year.unique():
            cells = []
            for book in ("base", "cl20", "vt20"):
                sample = fold["frames"][book].loc[base.index.year == year]
                cells.append("%6d %8.2f %7.2f" % (len(sample), 100*((1+sample.total).prod()-1), 100*np.sqrt(252*(sample.excess**2).mean())))
            print("%-5s %4d " % (name, year) + " ".join(cells))
    print("R5 report-only: 3x-fund position USD median/min/max; trade size is both funds together, initial entry excluded")
    print("fold  book balance  equity_med/min/max       bond_med/min/max        cash_med/min/max       trades/y trade_med trade_max")
    for name, fold in folds.items():
        for book in ("vt20", "cl20"):
            for balance in (200, 2000):
                p = r5_positions(fold["frames"][book], balance)
                cells = ["%7.2f/%7.2f/%7.2f" % tuple(p[asset + "_" + key] for key in ("median", "min", "max")) for asset in ("equity", "bond", "cash")]
                print("%-5s %-4s %7d " % (name, book, balance) + " ".join(cells) +
                      " %8.2f %9s %9s" % (p["trades_per_year"], fmt(p["trade_median"], 2), fmt(p["trade_max"], 2)))
    if charts_missing:
        print("Charts skipped: matplotlib is not installed.")
    print("Deviations:")
    for text in deviations:
        print(text)
    print("Configuration count: 2 decision-bearing, 6 report-only.")


def charts(folds):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return True
    plt.rcParams["axes.unicode_minus"] = False
    for name, fold in folds.items():
        fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
        for book in core_books:
            frame = fold["frames"][book]
            axes[0].plot(frame.index, (1 + frame.total).cumprod(), label=book)
        axes[0].set_yscale("log")
        axes[0].set_ylabel("Total-return equity (log)")
        for book in ("vt20", "vt40"):
            frame = fold["frames"][book]
            axes[1].plot(frame.index, frame.held_gross, label=book)
        for book in ("cl20", "cl40"):
            axes[1].axhline(fold["matches"][book][0], linestyle="--", label=book)
        axes[1].set_ylabel("Held leverage")
        for book in ("cl20", "vt20"):
            frame = fold["frames"][book]
            axes[2].plot(frame.index, np.sqrt(252 * frame.excess.pow(2).rolling(63).mean()), label=book)
        axes[2].axhline(.20, color="black", linestyle=":", label="target 0.20")
        axes[2].set_ylabel("63-day RMS excess vol")
        for axis in axes:
            axis.grid(alpha=.25)
            axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
        fig.suptitle("Task 14 " + name)
        fig.subplots_adjust(right=.81, hspace=.12)
        fig.savefig(results_dir / ("vol_target_" + name + ".png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
    return False


def run():
    all_prices, all_cash, observed = load_main()
    prices = all_prices[["spy", "ief"]]
    base_weights = unit_weights(prices, main_signal)
    base = vt.levered_account_returns(prices, all_cash, base_weights, .015)
    g1_check(base)
    fresh_prices, funds = load_fresh()
    g2_check(funds, all_prices)
    g3_check(all_prices, fresh_prices, all_cash)
    fresh_signal = first_fresh_signal(fresh_prices)
    fresh_cash = cash_returns(observed, fresh_prices.index)
    folds = {"main": build_fold(prices, all_cash, main_signal),
             "fresh": build_fold(fresh_prices, fresh_cash, fresh_signal)}
    add_s1(folds["main"], all_prices, all_cash)
    for name, fold in folds.items():
        calendar = fold["frames"]["base"].index
        print("%s: first_signal=%s first_execution=%s last_return=%s n=%d warmup_first_base_return=%s" %
              (name, fold["signal"].date(), calendar[0].date(), calendar[-1].date(), len(calendar), vt.base_returns(fold["prices"]).index[0].date()), flush=True)
    pooled_books, combined = calculate(folds)
    nearby = neighbors(folds)
    summary = summary_rows(folds, pooled_books, combined, nearby)
    returns, inputs = [], []
    columns = ["total", "excess", "cash_return", "held_gross", "turnover", "trading_cost", "financing_cost"]
    for name, fold in folds.items():
        for book in (*core_books, "s1l") if name == "main" else core_books:
            returns.append(fold["frames"][book][columns].assign(fold=name, book=book))
        inputs.append(fold["inputs"].assign(fold=name))
    return_frame = pd.concat(returns).rename_axis("date").reset_index()
    return_frame = return_frame[["date", "fold", "book", *columns]]
    input_frame = pd.concat(inputs).rename_axis("date").reset_index()
    input_frame = input_frame[["date", "fold", *[name for name in input_frame if name not in ("date", "fold")]]]
    if summary.duplicated(["fold", "period", "book", "statistic"]).any():
        raise ValueError("Duplicate summary keys")
    summary.to_csv(results_dir / "vol_target_summary.csv", index=False, float_format="%.12g")
    return_frame.to_csv(results_dir / "vol_target_returns.csv", index=False, float_format="%.12g")
    input_frame.to_csv(results_dir / "vol_target_inputs.csv", index=False, float_format="%.12g")
    missing = charts(folds)
    print_report(folds, pooled_books, combined, nearby, missing)


def main():
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "vol_target_report.txt").open("w", encoding="ascii") as report:
        with redirect_stdout(Tee(sys.stdout, report)):
            try:
                run()
            except Exception as error:
                print("STOP: " + str(error))
                print("Deviations:")
                for text in deviations:
                    print(text)
                raise


if __name__ == "__main__":
    main()
