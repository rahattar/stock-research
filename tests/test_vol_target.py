import builtins
import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.run_vol_target import r5_positions
from src import multi_asset_data as asset_data
from src import vol_target as vt
from src import vol_target_data as data
from src.multi_asset import account_returns


def synthetic(n=620):
    index = pd.bdate_range("2000-01-03", periods=n)
    rng = np.random.default_rng(41)
    returns = rng.normal([.0003, .0001], [.009, .003], size=(n, 2))
    prices = pd.DataFrame(100 * np.exp(returns.cumsum(axis=0)), columns=["equity", "bond"], index=index)
    cash = pd.Series(.02 / 252, index=index)
    return prices, cash


def path_frame(n=620, target=.20):
    prices, cash = synthetic(n)
    first = prices.index[260]
    path = vt.target_leverage(vt.forecast_vol(vt.base_returns(prices)), target, first)
    weights = vt.book_weights(prices.index, path.ltilde, first, list(prices))
    frame = vt.levered_account_returns(prices, cash, weights, .015)
    frame.attrs["target"] = target
    return prices, cash, first, path, weights, frame


def test_no_lookahead_and_execution_lag():
    prices, _ = synthetic()
    base = vt.base_returns(prices)
    first, cutoff = base.index[260], base.index[340]
    original = vt.target_leverage(vt.forecast_vol(base), .20, first)
    changed = base.copy()
    changed.loc[changed.index > cutoff] *= 30
    altered = vt.target_leverage(vt.forecast_vol(changed), .20, first)
    a = vt.book_weights(prices.index, original.ltilde, first, list(prices))
    b = vt.book_weights(prices.index, altered.ltilde, first, list(prices))
    next_date = prices.index[prices.index.get_loc(cutoff) + 1]
    pd.testing.assert_frame_equal(a.loc[:next_date], b.loc[:next_date])


def test_spike_first_changes_held_leverage_two_days_later():
    prices, cash = synthetic()
    base = pd.Series(.004, index=prices.index[1:])
    first, spike = base.index[250], base.index[300]
    original = vt.target_leverage(vt.forecast_vol(base), .20, first)
    base.loc[spike] = .20
    changed = vt.target_leverage(vt.forecast_vol(base), .20, first)
    weights = [vt.book_weights(prices.index, path.ltilde, first, list(prices)) for path in (original, changed)]
    frames = [vt.levered_account_returns(prices, cash, w, .015) for w in weights]
    differs = frames[0].held_gross != frames[1].held_gross
    assert differs.index[differs][0] == prices.index[prices.index.get_loc(spike) + 2]


def test_ewma_recursion_and_first_square():
    base = pd.Series([.01, -.02, .03, -.005], index=pd.bdate_range("2001-01-01", periods=4))
    expected, previous = [], base.iloc[0] ** 2
    for i, value in enumerate(base):
        previous = value ** 2 if i == 0 else .94 * previous + .06 * value ** 2
        expected.append(previous)
    sigma = vt.forecast_vol(base)
    np.testing.assert_allclose(sigma.to_numpy() ** 2 / 252, expected, rtol=1e-14)
    assert sigma.iloc[0] ** 2 / 252 == pytest.approx(base.iloc[0] ** 2)


def test_cap_floor_and_band_first_signal():
    index = pd.bdate_range("2001-01-01", periods=5)
    sigma = pd.Series([0., .001, .10, 1., np.inf], index=index)
    path = vt.target_leverage(sigma, .20, index[0], band=0)
    assert path.ltilde.between(0, 3).all()
    assert path.ltilde.iloc[0] == 3 and path.ltilde.iloc[-1] == 0
    sigma = pd.Series(.20 / np.array([1., 1.099, 1.101, 1.20, 1.30]), index=index)
    path = vt.target_leverage(sigma, .20, index[0])
    np.testing.assert_allclose(path.ltilde.iloc[:3], [1., 1., 1.101])


def test_financing_and_leverage_one_reproduction():
    prices, cash = synthetic(20)
    first = prices.index[2]
    for leverage in (.5, 1., 2.):
        weights = vt.book_weights(prices.index, pd.Series(leverage, index=prices.loc[first:].index), first, list(prices))
        original = account_returns(prices, cash, weights)
        actual = vt.levered_account_returns(prices, cash, weights, .015)
        expected = (original.gross_weight - 1).clip(lower=0) * .015 / 252
        np.testing.assert_allclose(actual.financing_cost, expected)
        np.testing.assert_allclose(actual.total, original.total - expected, atol=1e-16)
        np.testing.assert_allclose(actual.excess, original.excess - expected, atol=1e-16)
        if leverage == 2:
            assert actual.financing_cost.iloc[1] == pytest.approx(.015 / 252)
        if leverage == 1:
            pd.testing.assert_frame_equal(actual[original.columns], original, check_exact=True)


def test_matched_constant_and_uncapped_s1_matching():
    prices, cash, first, _, _, target = path_frame()
    unit = vt.book_weights(prices.index, pd.Series(1., index=prices.loc[first:].index), first, list(prices))
    leverage, matched, ratio = vt.matched_constant(prices, cash, unit, target)
    assert 0 <= leverage <= 3
    assert abs(ratio - 1) <= 1e-6
    assert matched.excess.std(ddof=1) == pytest.approx(target.excess.std(ddof=1), rel=1e-6)
    smaller = unit * .2
    leverage, matched, ratio = vt.matched_constant(prices, cash, smaller, target, cap=None)
    assert leverage > 3 and abs(ratio - 1) <= 1e-6


def hand_frame():
    index = pd.DatetimeIndex(list(pd.bdate_range("2001-01-02", periods=14)) +
                             list(pd.bdate_range("2001-02-01", periods=16)) +
                             list(pd.bdate_range("2001-03-01", periods=20)))
    excess = np.linspace(-.03, .02, len(index))
    total = excess + .0001
    frame = pd.DataFrame({"total": total, "excess": excess, "cash_return": .0001, "held_gross": 2.,
                          "turnover": 0., "trading_cost": 0., "financing_cost": .015 / 252,
                          "trade_date": False, "initial_entry": False}, index=index)
    frame.iloc[0, frame.columns.get_loc("held_gross")] = 0
    frame.iloc[0, frame.columns.get_loc("initial_entry")] = True
    return frame


def test_metrics_hand_examples_and_month_rules():
    frame = hand_frame()
    result = vt.metrics(frame, .20)
    groups = [sample for _, sample in frame.groupby(frame.index.month)]
    rv = [np.sqrt(252 * (sample.excess ** 2).mean()) for sample in groups if len(sample) >= 15]
    assert result["D"] == pytest.approx(np.std(np.log(rv), ddof=1))
    assert result["n_rv_months"] == 2
    k = int(np.ceil(.01 * len(frame)))
    es1 = -np.sort(frame.excess)[:k].mean() / frame.excess.std(ddof=1)
    assert result["ES1n"] == pytest.approx(es1)
    monthly = np.array([(1 + s.total).prod() - (1 + s.cash_return).prod() for s in groups])
    k = int(np.ceil(.05 * len(monthly)))
    es5 = -np.sort(monthly)[:k].mean() / (frame.excess.std(ddof=1) * np.sqrt(21))
    assert result["ES5m"] == pytest.approx(es5)
    equity = np.r_[1., (1 + frame.total).cumprod()]
    mdd = (1 - equity / np.maximum.accumulate(equity)).max()
    assert result["MDDn"] == pytest.approx(mdd / (frame.total.std(ddof=1) * np.sqrt(252)))
    assert result["mean_gross"] == result["median_gross"] == result["min_gross"] == result["max_gross"] == 2
    assert result["worst_day"] == frame.excess.min()
    assert result["worst_month"] == pytest.approx(monthly.min())
    assert np.isnan(result["max_year_vol_ratio"])


def test_ceil_tail_counts_above_integer_boundary():
    prices, cash, first, _, weights, frame = path_frame(700)
    result = vt.metrics(frame)
    k = int(np.ceil(.01 * len(frame)))
    assert k > .01 * len(frame)
    assert result["ES1n"] == pytest.approx(-np.sort(frame.excess)[:k].mean() / frame.excess.std(ddof=1))
    groups = list(frame.groupby(frame.index.to_period("M")))
    k = int(np.ceil(.05 * len(groups)))
    monthly = np.sort([(1 + sample.total).prod() - (1 + sample.cash_return).prod() for _, sample in groups])
    assert result["ES5m"] == pytest.approx(-monthly[:k].mean() / (frame.excess.std(ddof=1) * np.sqrt(21)))


def test_year_rms_126_day_rule_trades_and_positions():
    _, _, _, _, _, frame = path_frame(900)
    result = vt.metrics(frame)
    eligible = [np.sqrt(252 * (sample.excess ** 2).mean()) for _, sample in frame.groupby(frame.index.year) if len(sample) >= 126]
    assert result["max_year_vol_ratio"] == pytest.approx(max(eligible) / result["vol"])
    assert result["trades_per_year"] == pytest.approx(frame.trade_date.sum() * 252 / len(frame))
    assert result["turnover_per_year"] == pytest.approx(frame.turnover.mean() * 252)
    positions = r5_positions(frame, 200)
    active = frame.held_gross.loc[~frame.initial_entry]
    assert positions["equity_median"] == pytest.approx(200 * .6 * active.median() / 3)
    assert positions["cash_min"] == pytest.approx(200 * (1 - active.max() / 3))
    sizes = 200 * frame.loc[frame.trade_date, "leverage_change"].abs() / 3
    assert positions["trade_median"] == pytest.approx(sizes.median())


def test_pooled_fixed_fold_weights_and_draw_pairing():
    points = {"main": {"D": 2., "T": .1}, "fresh": {"D": 4., "T": -.2}}
    draws = {name: {key: np.array([value, value + 1]) for key, value in point.items()} for name, point in points.items()}
    result, sample, weights = vt.pooled(points, draws, {"main": 3, "fresh": 1})
    assert weights == {"main": .75, "fresh": .25}
    assert result["D"] == 2.5
    np.testing.assert_allclose(sample["D"], [.75 * 2 + .25 * 4, .75 * 3 + .25 * 5])


def test_bootstrap_circular_six_month_blocks_and_exact_metric_recomputation():
    _, _, _, _, _, frame = path_frame(500)
    frames = {"vt20": frame, "cl20": frame.copy()}
    draws = vt.block_bootstrap(frames, seed=9, resamples=12)
    again = vt.block_bootstrap(frames, seed=9, resamples=12)
    months = list(frame.groupby(frame.index.to_period("M")))
    m = len(months)
    starts = np.random.default_rng(9).integers(0, m, size=(12, int(np.ceil(m / 6))))
    expected = ((starts[:, :, None] + np.arange(6)) % m).reshape(12, -1)[:, :m]
    np.testing.assert_array_equal(draws["months"], expected)
    assert draws["months"].shape == (12, m)
    for key in vt.statistic_names:
        np.testing.assert_allclose(draws["books"]["vt20"][key], again["books"]["vt20"][key], equal_nan=True)
        np.testing.assert_allclose(draws["books"]["vt20"][key], draws["books"]["cl20"][key], equal_nan=True)
    selected = pd.concat([months[i][1] for i in expected[0]])
    assert draws["books"]["vt20"]["ES1n"][0] == pytest.approx(-np.sort(selected.excess)[:int(np.ceil(.01 * len(selected)))].mean() / selected.excess.std(ddof=1))
    for key in ("R_D", "R_MDD"):
        np.testing.assert_allclose(draws["comparisons"]["vt20_vs_cl20"][key], 1)
    np.testing.assert_allclose(draws["comparisons"]["vt20_vs_cl20"]["dS"], 0)


@pytest.fixture
def fund_files(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "fund_dir", tmp_path)

    def write(name="vfinx", tail=""):
        filename = tmp_path / (name + ".csv")
        filename.write_text("Date,Open,High,Low,Close,Adj Close,Volume,Dividends,Stock Splits\n"
                            "2005-09-29 00:00:00-04:00,100,101,99,100,50,0,0,0\n"
                            "2005-09-30 00:00:00-04:00,101,102,100,101,50.5,0,0,0\n" + tail, encoding="ascii")
        return filename
    return write


def test_fund_lock_rejects_before_any_date_parser(fund_files, monkeypatch):
    fund_files(tail="2005-10-03 00:00:00-04:00,NEVER-PARSE\n")
    original = data.new_york_date
    seen = []

    def guarded(value):
        assert value[:10] <= "2005-09-30"
        seen.append(value)
        return original(value)

    monkeypatch.setattr(data, "new_york_date", guarded)
    with pytest.raises(ValueError, match="before date parsing"):
        data.read_fund("vfinx")
    assert len(seen) == 2


def test_main_readers_cannot_parse_locked_dates(tmp_path, monkeypatch):
    monkeypatch.setattr(asset_data, "etf_dir", tmp_path)
    monkeypatch.setattr(asset_data, "cash_file", tmp_path / "dtb3.csv")
    (tmp_path / "spy.csv").write_text("Date,Open,High,Low,Close,Adj Close,Volume,Dividends,Stock Splits\n"
                                     "2025-10-30 00:00:00-04:00,100,101,99,100,50,10,0,0\n"
                                     "2025-11-03 00:00:00-05:00,NEVER-PARSE\n", encoding="ascii")
    asset_data.cash_file.write_text("observation_date,value\n2025-10-29,3\n2025-11-03,NEVER-PARSE\n", encoding="ascii")
    original = pd.Timestamp
    seen = []

    def timestamp(value, *args, **kwargs):
        if isinstance(value, str):
            assert value[:10] < "2025-11-01"
            seen.append(value)
        return original(value, *args, **kwargs)

    namespace = SimpleNamespace(**{name: getattr(pd, name) for name in dir(pd) if not name.startswith("__")})
    namespace.Timestamp = timestamp
    monkeypatch.setattr(asset_data, "pd", namespace)
    with data.read_guard("main"):
        frame, _, _ = asset_data._read_etf("spy")
        observed, _, _ = asset_data._read_cash()
        with (tmp_path / "spy.csv").open("rb") as stream:
            assert b"2025-11-03" not in stream.read()
    assert len(frame) == len(observed) == 1
    assert len(seen) >= 2


def test_fresh_guard_refuses_etfs_and_all_banned_roots():
    with data.read_guard("fresh"):
        for opener in (builtins.open, io.open):
            with pytest.raises(ValueError, match="not allowed"):
                opener(asset_data.etf_dir / "spy.csv")
            for name in ("download", "fresh", "hist"):
                with pytest.raises(ValueError, match="cannot read"):
                    opener(data.project / name / "not_read.csv")


def test_amended_proxy_nonoverlap_compounding_and_lags():
    rng = np.random.default_rng(2)
    pair = pd.DataFrame(rng.normal(0, .01, size=(19, 2)), columns=["fund", "etf"])
    result = data.amended_proxy_statistics(pair)
    expected = np.prod(1 + pair.iloc[:15].to_numpy().reshape(3, 5, 2), axis=1) - 1
    assert result["blocks"] == 3
    assert result["five_day_corr"] == pytest.approx(np.corrcoef(expected.T)[0, 1])
    assert result["lag_corr"] == pytest.approx(pair.fund.corr(pair.etf.shift(1)))
    assert result["lead_corr"] == pytest.approx(pair.fund.corr(pair.etf.shift(-1)))


@pytest.mark.parametrize("key,value", [("annual_diff", .00501), ("daily_corr", .9499),
                                        ("five_day_corr", .9849), ("lag_corr", .20), ("lead_corr", -.20)])
def test_amended_proxy_each_failing_criterion(key, value):
    values = dict(annual_diff=.005, daily_corr=.95, five_day_corr=.985, lag_corr=.1999, lead_corr=-.1999)
    assert data.amended_proxy_pass(values)
    values[key] = value
    assert not data.amended_proxy_pass(values)


@pytest.mark.parametrize("low,gross,expected", [(0.001, 3., "ADOPT S1"), (0, 2., "PLANT STAYS 60/40"),
                                               (.001, 3.0001, "PLANT STAYS 60/40")])
def test_step1_truth_table(low, gross, expected):
    assert vt.verdict_step1(low, gross) == expected


@pytest.mark.parametrize("values,expected", [
    ((.99, .99, -.05, -.10, .001), ("ADDS VALUE", "keep")),
    ((.99, .99, -.05, -.10, 0), ("REGULATES RISK", "keep")),
    ((1., .99, 0., 0., .1), ("NO BENEFIT", "drop")),
    ((.99, 1., 0., 0., .1), ("NO BENEFIT", "drop")),
    ((.99, .99, -.05001, 0., .1), ("NO BENEFIT", "drop")),
    ((.99, .99, 0., -.10001, .1), ("NO BENEFIT", "drop")),
])
def test_step2_truth_table_and_each_single_failure(values, expected):
    assert vt.verdict_step2(*values) == expected
