"""Task 14 input guards, fund validation, and the three frozen data gates."""

import builtins
from contextlib import contextmanager
import csv
import io
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import multi_asset_data as asset_data
from .daily_momentum import return_statistics
from .multi_asset import price_panel
from .multi_asset_data import new_york_date, validate_dates


project = Path(__file__).resolve().parents[1]
fund_dir = project / "data" / "fund"
fund_names = ("vfinx", "vfitx")
reference_file = project / "results" / "multi_asset_returns.csv"
main_signal = pd.Timestamp("2005-09-30")
main_execution = pd.Timestamp("2005-10-03")
fund_end = "2005-09-30"
lock = "2025-11-01"


def raw_date(line):
    """Inspect an ISO prefix as text, before any datetime or numeric parser."""
    value = line.partition(",")[0].strip().strip('"')[:10]
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
        raise ValueError("Expected an ISO calendar-date prefix")
    return value


def filtered_text(source, *, fund=False):
    header = source.readline()
    lines = [header]
    for line in source:
        date = raw_date(line)
        if fund and date > fund_end:
            raise ValueError("Fund row after 2005-09-30 rejected before date parsing")
        if date >= lock:
            continue
        lines.append(line)
    return io.StringIO("".join(lines))


@contextmanager
def read_guard(fold):
    """Restrict Task 14 reads and prefilter dates for unchanged legacy readers."""
    if fold not in ("main", "fresh"):
        raise ValueError("Unknown fold")
    funds = {(fund_dir / (name + ".csv")).resolve() for name in fund_names}
    etfs = {(asset_data.etf_dir / (name + ".csv")).resolve() for name in asset_data.tickers}
    cash = asset_data.cash_file.resolve()
    allowed = (etfs if fold == "main" else funds) | {cash}
    originals = builtins.open, io.open

    def guard(original):
        def checked(filename, *args, **kwargs):
            if not isinstance(filename, (str, Path)):
                return original(filename, *args, **kwargs)
            path = Path(filename).resolve()
            if any(path.is_relative_to(project / name) for name in ("download", "fresh", "hist")):
                raise ValueError("Task 14 cannot read this directory")
            if path.is_relative_to(project / "data") and path not in allowed:
                raise ValueError("Task 14 input is not allowed for this fold")
            mode = kwargs.get("mode", args[0] if args else "r")
            source = original(filename, *args, **kwargs)
            if "r" in mode and (path in allowed or path == reference_file.resolve()):
                try:
                    if "b" in mode:
                        text = filtered_text(io.StringIO(source.read().decode("utf-8-sig")), fund=path in funds)
                        return io.BytesIO(text.getvalue().encode("ascii"))
                    return filtered_text(source, fund=path in funds)
                finally:
                    source.close()
            return source
        return checked

    builtins.open, io.open = guard(originals[0]), guard(originals[1])
    try:
        yield
    finally:
        builtins.open, io.open = originals


def load_main():
    with read_guard("main"), asset_data.restricted_asset_reads():
        inputs, observed, _ = asset_data.load_inputs()
        prices = price_panel(inputs)
        cash = asset_data.cash_returns(observed, prices.index)
    return prices, cash, observed


def g1_check(frame):
    """Only the leverage-one reproduction book is evaluated before this gate."""
    with read_guard("main"):
        reference = pd.read_csv(reference_file, usecols=["date", "strategy", "total"], parse_dates=["date"])
    expected_daily = reference.loc[reference.strategy == "S2"].set_index("date").total
    if not frame.index.equals(expected_daily.index):
        raise ValueError("G1 failed: daily calendars differ")
    difference = float((frame.total - expected_daily).abs().max())
    total, excess = return_statistics(frame.total), return_statistics(frame.excess)
    actual = {"annual_total_return": total["annual_return"], "annual_excess_return": excess["annual_return"],
              "excess_sharpe": excess["sharpe"], "max_drawdown": total["max_drawdown"]}
    expected = dict(annual_total_return=.08522764, annual_excess_return=.06749890,
                    excess_sharpe=.64394582, max_drawdown=.31393532)
    passed = len(frame) == 5053 and difference < 1e-12 and all(abs(actual[name] - value) <= 1e-6 for name, value in expected.items())
    print("G1 %s: total=%.8f excess=%.8f Sharpe=%.8f MDD=%.8f n=%d max_daily_diff=%.12g" %
          ("PASS" if passed else "FAIL", *actual.values(), len(frame), difference), flush=True)
    if not passed:
        raise ValueError("G1 reproduction failed")
    return actual


def read_fund(name):
    if name not in fund_names:
        raise ValueError("Unknown fund")
    filename = fund_dir / (name + ".csv")
    rows, dates = [], []
    with filename.open(newline="", encoding="utf-8-sig") as source:
        header = next(csv.reader(source), None)
        required = asset_data.required_columns
        if (not header or header[0] != "Date" or len(set(header)) != len(header) or
                not set(required).issubset(header) or set(header) - set(required) - {"Capital Gains"}):
            raise ValueError("Malformed fund header: " + filename.name)
        positions = [header.index(column) for column in required[1:]]
        for line in source:
            if raw_date(line) > fund_end:
                raise ValueError("Fund row after 2005-09-30 rejected before date parsing")
            date = new_york_date(line.partition(",")[0].strip().strip('"'))
            fields = next(csv.reader([line]))
            if len(fields) != len(header):
                raise ValueError("Malformed fund field count: " + filename.name)
            dates.append(date)
            rows.append([float(fields[position]) for position in positions])
    if not rows:
        raise ValueError("Empty fund: " + filename.name)
    frame = pd.DataFrame(rows, index=pd.DatetimeIndex(dates, name="date"), columns=required[1:]).sort_index()
    duplicates = int(frame.index.duplicated().sum())
    nonpositive = int(frame[list(asset_data.price_columns)].le(0).any(axis=1).sum())
    jumps = int(frame["Adj Close"].pct_change(fill_method=None).abs().gt(.25).sum())
    print("%-12s rows=%d first=%s last=%s duplicates=%d nonpositive=%d absret_gt_25pct=%d" %
          (filename.name, len(frame), frame.index[0].date(), frame.index[-1].date(), duplicates, nonpositive, jumps), flush=True)
    if (not np.isfinite(frame.to_numpy()).all() or duplicates or nonpositive or jumps or
            (frame[["Volume", "Dividends", "Stock Splits"]] < 0).any().any() or
            (frame.High < frame[["Open", "Close", "Low"]].max(axis=1) - 1e-8).any() or
            (frame.Low > frame[["Open", "Close", "High"]].min(axis=1) + 1e-8).any()):
        raise ValueError("Malformed fund or failed fund quality gate: " + filename.name)
    validate_dates(frame)
    if frame.index[-1] > pd.Timestamp(fund_end):
        raise ValueError("Fund date exceeds fresh boundary")
    return frame


def load_fresh():
    with read_guard("fresh"):
        funds = {name: read_fund(name) for name in fund_names}
    later = max(frame.index[0] for frame in funds.values())
    calendars = [frame.loc[later:].index for frame in funds.values()]
    if not calendars[0].equals(calendars[1]):
        raise ValueError("G2 failed: fund trading dates differ after later inception")
    prices = pd.concat({name: funds[name]["Adj Close"] for name in fund_names}, axis=1).loc[later:]
    return prices, funds


def g2_check(funds, main_prices):
    print("G2 proxies: annualized mean difference is fund minus ETF", flush=True)
    print("fund   etf  first       last         n       corr   annual_diff  min_corr max_abs_diff result", flush=True)
    checks = (("vfinx", "spy", "2000-01-04", .97, .005), ("vfitx", "ief", "2002-07-31", .85, .02))
    bond_pass = False
    amended = None
    for fund, etf, first, minimum, maximum in checks:
        a = funds[fund]["Adj Close"].pct_change(fill_method=None).loc[first:fund_end]
        b = main_prices[etf].pct_change(fill_method=None).loc[first:fund_end]
        pair = pd.concat([a.rename("fund"), b.rename("etf")], axis=1).dropna()
        correlation = pair.fund.corr(pair.etf)
        difference = (pair.fund - pair.etf).mean() * 252
        good = bool(len(pair) > 1 and correlation >= minimum and abs(difference) <= maximum)
        print("%-6s %-4s %s %s %5d %10.6f %12.6f %9.2f %12.3f %s" %
              (fund, etf, pair.index[0].date(), pair.index[-1].date(), len(pair), correlation, difference,
               minimum, maximum, "PASS" if good else "FAIL"), flush=True)
        if fund == "vfinx":
            amended = amended_proxy_statistics(pair)
            print("G2 registered vfinx/spy: " + ("PASS" if good else "FAIL"), flush=True)
        else:
            bond_pass = good
    print("G2 amended A1 vfinx/spy: daily_corr=%.6f five_day_corr=%.6f lag_corr=%.6f lead_corr=%.6f annual_diff=%.6f blocks=%d %s" %
          (amended["daily_corr"], amended["five_day_corr"], amended["lag_corr"], amended["lead_corr"],
           amended["annual_diff"], amended["blocks"], "PASS" if amended_proxy_pass(amended) else "FAIL"), flush=True)
    if not bond_pass or not amended_proxy_pass(amended):
        raise ValueError("Amended G2 proxy checks failed")
    print("G2 fund calendars identical after later inception; quality checks PASS", flush=True)


def amended_proxy_statistics(pair):
    blocks = len(pair) // 5
    five_day = np.prod(1 + pair[["fund", "etf"]].iloc[:blocks * 5].to_numpy().reshape(blocks, 5, 2), axis=1) - 1
    return {"daily_corr": pair.fund.corr(pair.etf), "five_day_corr": np.corrcoef(five_day.T)[0, 1],
            "lag_corr": pair.fund.corr(pair.etf.shift(1)), "lead_corr": pair.fund.corr(pair.etf.shift(-1)),
            "annual_diff": (pair.fund - pair.etf).mean() * 252, "blocks": blocks}


def amended_proxy_pass(values):
    return bool(abs(values["annual_diff"]) <= .005 and values["daily_corr"] >= .95 and
                values["five_day_corr"] >= .985 and abs(values["lag_corr"]) < .20 and abs(values["lead_corr"]) < .20)


def g3_check(main_prices, fresh_prices, cash):
    if main_prices.index[-1] >= pd.Timestamp(lock) or cash.index[-1] >= pd.Timestamp(lock) or fresh_prices.index[-1] > pd.Timestamp(fund_end):
        raise ValueError("G3 boundary check failed")
    print("G3 PASS: raw ISO prefixes gated before datetime/numeric parsing; main < 2025-11-01; funds <= 2005-09-30.", flush=True)
