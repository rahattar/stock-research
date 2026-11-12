# Stock research: summary and state of the project

Split from the forex-research repo on 2026-11-11. That repo keeps the FX work (tasks 1 to 12).
This repo holds the multi-asset and levered-portfolio work (tasks 13 and 14 onward).

Read this fully before proposing a new test.

---

## 1. Goal and constraints

- Find a strategy with a positive expected return that a small personal account can run without
  screen time, and prove it to a stated standard before any larger money is used.
- Live budget: 200 USD, already written off. High risk is acceptable at this stage.
- No other money is committed until the gates in section 6 are passed. Those gates stay strict.
- Risk level: target volatility 20% (primary). 40% was tested as a secondary level (section 4).
- What "works" means here:
  - Risk behaviour (volatility on target, costs as modelled): verifiable in about a year.
  - Return: cannot be proven by testing. Sharpe 0.4 needs about 25 years for t = 2.
    The protection against a wrong return assumption is sizing, not proof.

## 2. Workflow and methodology

- Claude drafts pre-registrations and prompts, the owner runs them in Codex on a Mac, Claude
  reviews the output against the files in the repo, not against the pasted reply.
- Pre-register rules and pass criteria before looking at the data they are tested on.
- Account-free measures (Sharpe, ratios), paired benchmarks, block bootstraps.
- Confirm on data the project has not looked at. Count configurations.
- Set thresholds by simulation on synthetic data before the run, and write predictions down.
- Locked final year: 2025-11-01 to 2026-10-31. Never read. One run of a fully specified strategy.
- Preferences: concise, ASCII-only, technically rigorous, lowercase file names, chart legends
  outside the plot area.

## 3. Data (not in the repo; data/readme.md shows how to rebuild)

- data/etf/: spy, iwm, efa, eem, ief, tlt, lqd, hyg, gld, dbc, vnq, fxe, fxy, uup; yfinance daily,
  2000 to 2025-10-31.
- data/fund/: vfinx, vfitx; yfinance daily, 1991 to 2005-09-30 (fresh-fold proxies for 60/40).
- data/fred/dtb3.csv: 3-month T-bill, the cash rate.

## 4. Results

### Task 13: multi-asset ETFs, 2005-10 to 2025-10 (verified against files)

| Book | Excess Sharpe [95% CI] | Vol | Max drawdown |
|---|---|---|---|
| S1: 14 ETFs, inverse volatility | 0.49 [0.06, 0.93] | 5.0% | 12.1% |
| T1: S1 with trend long/flat | 0.51 [0.13, 0.92] | 3.6% | 5.9% |
| T2: S1 with trend long/short | 0.25 [-0.15, 0.65] | 4.0% | 6.5% |
| 60/40 (spy/ief) | 0.64 [0.27, 1.07] | 11.1% | 31.4% |
| spy | 0.55 [0.20, 0.94] | 19.4% | 55.2% |

- Verdict for T1 against S1: REDUCES RISK ONLY (no Sharpe gain).
- Part of T1's drawdown cut is lower exposure (41% cash). Per unit of volatility the cut is about
  32%, not 51%, and it rests on two episodes (2008, 2022).
- S1 holds about a quarter of its weight in currency ETFs, which earned nothing.

### Task 14: levered 60/40 and volatility targeting (prereg_task14.md)

Status: results as reported by Codex. NOT yet verified against files (section 5).

Setup
- Plant: 60/40 with borrowed money, cost = cash rate + 1.5% a year on the borrowed part, gross
  cap 3.0. Folds: main 2005-10 to 2025-10 (spy/ief), fresh 1992-11 to 2005-09 (vfinx/vfitx).
- vt20: leverage = min(3, 0.20 / EWMA volatility), 10% no-trade band, two-day delay.
- cl20: constant leverage with the same average volatility as vt20 (the bar).

Gates
- G1: the leverage-1 book reproduces task 13's 60/40 to 5e-14 per day.
- G2 failed as registered (vfinx/spy daily correlation 0.967 against a guessed minimum of 0.97).
  Amended once, before any evaluation (section 13 of the pre-registration). The amended gate
  passed: 5-day correlation 0.994, as the timing model predicted.

Decisions
- Step 1: the plant stays 60/40. The levered 14-ETF book has Sharpe 0.30 against 0.59 and needs
  3.9x gross.
- Step 2: REGULATES RISK, keep the volatility target.

| vt20 against cl20 | Main | Fresh | Pooled |
|---|---|---|---|
| Risk dispersion ratio R_D | 0.62 [0.54, 0.73] | 0.55 [0.48, 0.64] | 0.59 |
| Daily tail gain T | +0.12 [-0.04, 0.21] | -0.03 [-0.10, 0.06] | +0.065 |
| Sharpe difference dS | +0.07 [-0.12, 0.25] | -0.03 [-0.17, 0.11] | +0.03 [-0.10, 0.15] |
| Max drawdown vt20 / cl20 | 42% / 51% | 50% / 48% | |
| Sharpe vt20 / cl20 | 0.66 / 0.59 | 0.46 / 0.49 | |
| Total return a year vt20 / cl20 | 13.2% / 11.7% | 11.5% / 12.2% | |

Reading
- Steadier risk is the only replicated effect: yearly realized volatility stays at 17 to 22%
  (11% in 2017, at the cap) against 7 to 40% at constant leverage.
- No replicated return or tail benefit: the folds disagree in sign, pooled about zero.
- It is not loss protection: in the 7 full years constant leverage lost money, vt20 lost more in
  5 (1994, 2000, 2001, 2002, 2018) and less in 2 (2008, 2022).
- It costs more to run: 0.9 points a year more financing in the main fold (average leverage
  2.3 against 1.7) and 0.4 in the fresh fold (2.2 against 1.9), plus about 0.25% trading. At a
  3% spread the pooled Sharpe difference is -0.004.
- In 31 full calendar years vt20 made 30% or more in 8 and lost 20% or more in 3.
- The 40% target realized only 26 to 28% volatility (cap) with drawdowns of 60 to 67% for 1 to 2
  points more return. Not recommended.
- Doubling the spread to 3% removes 0.06 to 0.10 Sharpe and 1 to 2 points of return a year.

## 5. Open issues

- Verification: the task 14 code, tests and result files are on the owner's Mac and are not in
  this repo yet (scripts/import_from_forex.sh copies them). results/vol_target_report.txt here is
  the report text as pasted. Until the files are checked, section 4 task 14 is provisional.
- Timestamp: the pre-registration was committed to git after the run. The copy saved in the
  owner's "Forex Research" Claude project predates the run (first version 2026-11-12T02:36Z).
- The 1.5% spread is an assumption. Real 2x and 3x funds have not been examined.
- 1992 to 2025 was a favourable period for US stocks and bonds. Forward results may be worse.
- Taxes are not modelled.
- Inherited code: src/daily_carry.py, daily_momentum.py, instruments.py, rates.py, loader.py and
  scripts/run_daily_carry.py are FX modules copied unmodified because src/multi_asset.py uses two
  helpers from them. A cleanup is optional and must keep gate G1 exact.

## 6. Plan

1. Verify task 14 from the files (recompute books, metrics and bootstrap from
   results/vol_target_inputs.csv and results/vol_target_returns.csv; read the code for lookahead).
2. Task 15: map the book to real 2x and 3x funds and validate the spread on their history.
3. Step 3: trend filter on top of vt20, pre-registered, judged on drawdown per unit of volatility.
4. Step 4: cost-aware or MPC rebalancing; it must beat the simple rules of steps 2 and 3.
5. Locked year: one run of the fully specified strategy, risk metrics as pass criteria.
6. Paper tracking of vt20 runs in parallel. The 200 USD goes live after the locked-year run.
7. Any larger money is the owner's decision, outside the project, with unchanged gates.

## 7. Files

- prereg_task14.md: the frozen specification of task 14, with clarifications and amendment A1.
- prompts/codex_prompt_task14.md: the prompt Codex ran.
- scripts/sim_vol_target_power.py: simulation that set the task 14 thresholds.
- src/multi_asset.py, src/multi_asset_data.py, scripts/run_multi_asset.py,
  tests/test_multi_asset.py, results/multi_asset_*: task 13, copied unmodified from
  forex-research commit 6ccc49a.
- results/vol_target_report.txt: task 14 report (as pasted, pending the files).
- conftest.py: skips the one inherited test that needs FX data.
