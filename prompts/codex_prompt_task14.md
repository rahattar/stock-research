# Codex prompt, task 14: levered 60/40 and volatility targeting

You are working in the forex-research repo root. Python: /opt/miniconda3/envs/pix2pixpolar-dev/bin/python.

Read prereg_task14.md completely before writing code. It is the frozen specification: every
rule, parameter, metric, bootstrap detail and decision rule is defined there. This prompt only
says how to build and report it. If this prompt and the pre-registration disagree, the
pre-registration wins. If anything is ambiguous, STOP and ask; do not choose.

## Hard rules

- Do not change any rule, parameter or threshold. Do not add, drop or tune configurations.
- Do not modify existing files in src/, scripts/, tests/ or results/. Add new files only.
  Allowed edits to existing files: .gitignore (add data/fund/) and data/readme.md (append the
  download command below). Do not edit summary.md, spec.md or prereg_task14.md.
- Do not modify scripts/sim_vol_target_power.py (already in the repo; not part of this task).
- Do not read download/, fresh/ or hist/. Do not parse any date on or after 2025-11-01.
  Do not parse any fund row after 2005-09-30.
- ASCII only in code, comments, output and file contents. Lowercase file names.
- Simple, readable code. pandas, numpy, pytest; matplotlib only for the charts.
- Report what the files say. No interpretation beyond the pre-registered verdicts.
- If you rerun the evaluation after seeing results, give the reason under Deviations.

## Step A: data

    mkdir -p data/fund
    python - <<'EOF'
    import yfinance as yf
    for t in ["VFINX", "VFITX"]:
        yf.Ticker(t).history(start="1991-01-01", end="2005-10-01", auto_adjust=False).to_csv(f"data/fund/{t.lower()}.csv")
    EOF

If the download fails (no network or missing yfinance), print the command and STOP. Do not
inspect the fund files beyond what the loader and the G2 checks need. Append the
command to data/readme.md under a heading "Funds, task 14 fresh fold (data/fund/)". Add
data/fund/ to .gitignore.

## Step B: code (new files)

src/vol_target_data.py
- Fund reader for data/fund/vfinx.csv and vfitx.csv with the same checks as
  src.multi_asset_data._read_etf (header, field count, finite values, positive prices,
  duplicates, high/low consistency), New York calendar dates, and a hard cutoff: rows after
  2005-09-30 are never parsed. Reuse new_york_date and validate_dates.
- A read guard like restricted_asset_reads that allows only the two fund files and dtb3.csv
  for the fresh fold. The main fold uses the existing load_inputs, price_panel, cash_returns
  inside the existing restricted_asset_reads.
- The G2 proxy checks of the pre-registration, printed as a table; raise on failure.

src/vol_target.py
- base_returns(prices): 0.6 / 0.4 daily base-book return.
- forecast_vol(base, lam=0.94): the EWMA of the pre-registration, annualized.
- target_leverage(sigma, target, first_signal, cap=3.0, band=0.10): lstar and ltilde.
- book_weights(...): weights in the task 13 convention (row at execution date t+1 holds
  ltilde_t * (0.6, 0.4); index starts at the first execution date).
- levered_account_returns(prices, cash, weights, spread): calls
  src.multi_asset.account_returns(prices[columns], cash, weights) and subtracts
  max(gross_weight - 1, 0) * spread / 252 from total and excess; adds a financing_cost column.
- matched_constant(...): the constant leverage whose sd(excess) equals the target book's
  (relative tolerance 1e-6). Also used for s1l with the task 13 S1 weights
  (src.multi_asset.target_weights), uncapped.
- metrics(frame, target=None): every per-book metric of section 6.
- comparison(vt, cl): R_D, T, dS, T_m, R_MDD.
- block_bootstrap(frames, seed, resamples=5000, block=6): the paired circular month-block
  bootstrap of section 7, returning draws of every statistic for every book and comparison.
  A plain loop over resamples is fine; do not approximate.
- pooled(...): fold weights, pooled point estimates and pooled draws.
- verdict_step1(...), verdict_step2(...): the rules of section 8, as pure functions of numbers.

scripts/run_vol_target.py
- Order: load main fold; G1 (STOP on failure, before any other book); load fresh fold; G2 and G3;
  build books; metrics; bootstrap; verdicts; report-only items R1 to R5; save; print.
- Print the full report to stdout and write the same text to results/vol_target_report.txt.

## Step C: tests (tests/test_vol_target.py, synthetic data only)

1. No lookahead: changing base returns after date t leaves weights at every index <= t+1 unchanged.
2. Timing: a volatility spike on day t first changes held leverage on day t+2.
3. EWMA: equals the explicit recursion on a hand example; first value equals the first square.
4. Cap and floor: leverage always in [0, 3].
5. Band: no change at a 9.9% move, change at a 10.1% move; the first signal sets lstar.
6. Financing: gross 2 for one day costs spread / 252 more than the unlevered formula; gross <= 1 costs 0.
7. Leverage 1 reproduces src.multi_asset.account_returns for the S2 weights exactly.
8. Matched constant: sd(excess) equal within 1e-6 relative.
9. Metrics on hand examples: RV_m with the 15-day rule, D, ES1n and ES5m with the ceil rule,
   MDDn, pooled weights.
10. Bootstrap: same seed gives identical draws; blocks are six consecutive months modulo M;
    all books share the same months; resample length is M months.
11. Locks: the fund reader rejects a row dated 2005-10-03; no reader parses 2025-11-03;
    the fresh guard refuses data/etf files.
12. Verdicts: truth table for step 1 and step 2, including each single failing criterion.

Run the whole suite: python -m pytest -q. All existing tests must still pass, unmodified.

## Step D: outputs

- results/vol_target_report.txt
- results/vol_target_summary.csv: long format, one row per (fold or pooled, period, book or
  comparison, statistic) with value, ci_low, ci_high where they exist.
- results/vol_target_returns.csv: date, fold, book, total, excess, cash_return, held_gross,
  turnover, trading_cost, financing_cost for base, cl20, vt20, cl40, vt40 (both folds) and
  s1l (main). float_format %.12g.
- results/vol_target_inputs.csv: date, fold, r_equity, r_bond, cash_return, sigma_hat,
  lstar_20, ltilde_20, lstar_40, ltilde_40, by signal date. float_format %.12g.
- results/vol_target_main.png and results/vol_target_fresh.png: three stacked panels sharing
  the date axis: (1) total-return equity, log scale, base, cl20, vt20, cl40, vt40;
  (2) held leverage of vt20 and vt40, with the cl20 and cl40 levels as horizontal lines;
  (3) 63-day realized vol of cl20 and vt20 with a line at 0.20. Legends outside the plot area
  (right side). If matplotlib is missing, skip the charts and say so.

## Step E: report layout (plain text, fixed-width tables)

1. Data report: file audit lines, G1 result with the four numbers and the max daily difference,
   G2 table, G3 statement, first signal / first execution / last return / n days per fold.
2. Books, per fold and for the main-fold split before_2017 / from_2017: total, excess, vol,
   Sharpe [95% CI], MDD, MDDn, D, ES1n, ES5m, worst day, worst month, mean held gross,
   min/max gross, trades/yr, turnover/yr, trading cost/yr, financing cost/yr, ln(vol/target).
3. Matched constants: L_c per fold for cl20, cl40, s1l, and the achieved sd ratio.
4. Step 2 table for vt20 against cl20: R_D, T, dS, T_m, R_MDD with 95% CI, for main, fresh and
   pooled. Then K1 (each fold), K2, K3 with the numbers used, and the verdict line:
   "Step 2 (vt20 vs cl20): <ADDS VALUE|REGULATES RISK|NO BENEFIT> -> <keep|drop>".
5. Step 1: Sharpe of s1l and cl20, difference with 95% CI, max gross of s1l, and
   "Step 1 (s1l vs cl20): <ADOPT S1|PLANT STAYS 60/40>".
6. Report-only R1 to R5, each labelled "report-only".
7. Deviations: list every deviation from the pre-registration, or "none".
8. Configuration count line: 2 decision-bearing, 6 report-only.

## Final reply (in this order)

1. Files created or changed.
2. The pytest summary line.
3. The full text of results/vol_target_report.txt.
4. Deviations or open questions, or "none".
