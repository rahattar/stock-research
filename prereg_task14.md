# Pre-registration, task 14: levered 60/40 plant and volatility targeting (ladder steps 0 to 2)

Written 2026-11-11, before any leverage or volatility-target result exists in this project.
Frozen once committed. Any deviation must be listed in the report under "Deviations".

## 1. Purpose

The project moves from FX direction (closed, see summary.md) to a levered risk-premium portfolio
run at high risk on a small account. The ladder adds one component at a time and keeps it only
if it beats the step below. This task covers:

- Step 0: the bar. 60/40 at constant leverage.
- Step 1: does the 14-ETF diversified book (S1 of task 13) replace 60/40 as the plant?
- Step 2: does a volatility target improve on constant leverage at matched average volatility?

Risk levels: target volatility 20% (primary, decision-bearing) and 40% (secondary, report-only).

## 2. What has already been seen

- Main fold (2005-10 to 2025-10): unlevered results of S1, T1, T2, S2 (60/40) and S3 are known
  from task 13. No leverage, financing or volatility-target result has been computed.
- Fresh fold (1992-11 to 2005-09, mutual fund proxies): never loaded in this project.
- The fresh fold is fresh to this project, not to the literature. Volatility-managed portfolios
  were published in 2017 (Moreira and Muir), so only 2017 onward is post-publication. That split
  is reported.
- All parameters below are fixed a priori from convention. Nothing is fitted on either fold.
- Thresholds in section 8 were set with a simulation on synthetic returns
  (scripts/sim_vol_target_power.py), not with market data.

## 3. Data and folds

Main fold
- Files: data/etf/*.csv (the 14 ETFs of task 13) and data/fred/dtb3.csv. Adj Close returns.
- Dates: exactly those of task 13. First signal 2005-09-30, first execution 2005-10-03,
  last return 2025-10-31, 5053 daily returns.
- Base book: spy 0.6, ief 0.4.
- Volatility warm-up uses spy and ief returns from the first common date (2002-07-31).

Fresh fold
- Files: data/fund/vfinx.csv (S&P 500 index fund) and data/fund/vfitx.csv (intermediate
  Treasury fund), yfinance daily, Adj Close, downloaded with end 2005-10-01. Cash: dtb3.
- Base book: vfinx 0.6, vfitx 0.4.
- First signal: the first month-end with at least 252 base-book returns (expected 1992-10-30).
  Last return 2005-09-30. The loader must reject any fund row after 2005-09-30.

Locked period: 2025-11-01 to 2026-10-31. Not read. Not downloaded for ETFs or funds.

Data gates (STOP before any evaluation if one fails)
- G1 reproduction: the base book (leverage 1) on the main fold equals task 13 S2 day by day
  (max absolute difference of daily total return below 1e-12) and reproduces the S2 "all" row:
  annual_total_return 0.08522764, annual_excess_return 0.06749890, excess_sharpe 0.64394582,
  max_drawdown 0.31393532, n 5053 (tolerance 1e-6).
- G2 proxies: vfinx against spy on 2000-01-04 to 2005-09-30: daily return correlation at least
  0.97 and annualized mean return difference within 0.5 percentage points. vfitx against ief on
  2002-07-31 to 2005-09-30: daily return correlation at least 0.85 and annualized mean return
  difference within 2.0 percentage points (catches unadjusted distributions). Both funds: identical
  trading dates after the later inception, no duplicate dates, no non-positive prices, no
  absolute daily return above 25%.
- G3 lock: no date on or after 2025-11-01 is parsed; no fund date after 2005-09-30 is parsed.

## 4. Cost and leverage model

For weights w (fractions of equity, by asset) set at an execution date and held from the next day:

    held_d   = weights_(d-1)
    total_d  = cash_d + sum_i held_i,d * (r_i,d - cash_d)
               - 0.0005 * sum_i |weights_i,d - weights_i,d-1|
               - max(sum_i held_i,d - 1, 0) * b / 252
    excess_d = total_d - cash_d

- Everything except the last term is src.multi_asset.account_returns with short_fee 0 (task 13
  convention: signal at t, execution at close t+1, returns from t+2; weights are constant between
  changes, that is, a daily-reset constant mix).
- b = 0.015 per year on the borrowed fraction, on top of the cash rate. It stands for margin or
  swap spread plus the fees of a levered fund. It is an assumption. It is stressed at b = 0.03
  here and validated against real 2x and 3x funds in task 15.
- Gross cap 3.0 (what 3x funds allow on a small account). No shorting.
- cash_d: src.multi_asset_data.cash_returns (dtb3, previous observation, /100/252).

## 5. Books

Notation: r_b,t = 0.6 * r_equity,t + 0.4 * r_bond,t (base-book daily return, total).

Volatility forecast (uses returns through t only)

    v_t     = 0.94 * v_(t-1) + 0.06 * r_b,t^2      (v at the first base return = its square;
                                                    pandas ewm(alpha=0.06, adjust=False).mean())
    sigma_t = sqrt(252 * v_t)

Volatility-target leverage, decided every trading day t from the first signal date t0

    lstar_t  = min(3.0, target / sigma_t)
    ltilde_t = lstar_t                    if t = t0 or |lstar_t / ltilde_(t-1) - 1| > 0.10
             = ltilde_(t-1)               otherwise

The weights at execution date t+1 are ltilde_t * (0.6, 0.4).

| Book | Leverage rule | Role |
|---|---|---|
| base | 1 | reproduction gate, reference |
| vt20 | target 0.20 | step 2, primary |
| cl20 | constant, matched to vt20 | step 0 bar for vt20 |
| vt40 | target 0.40 (cap-limited) | secondary |
| cl40 | constant, matched to vt40 | bar for vt40 |
| s1l | task 13 S1 weights times a constant, matched to vt20, no cap, main fold only | step 1 |

Matched constant leverage: for a volatility-target book X in a fold, the constant L_c that makes
sd(excess) of the constant book equal to sd(excess) of X over that fold (relative tolerance 1e-6).
This uses hindsight on purpose and only to equalize average risk. It is a comparison device, not
a tradable rule. L_c is computed once per fold and is not re-matched inside bootstrap resamples.

## 6. Metrics

Per book and fold, on daily returns (n days, sd with ddof 1):

- Annual geometric total and excess return, vol = sd(excess) * sqrt(252),
  Sharpe = mean(excess) / sd(excess) * sqrt(252), MDD of the total-return equity curve
  (all as src.daily_momentum.return_statistics).
- RV_m = sqrt(252 * mean of excess_d^2) for each calendar month with at least 15 returns.
- D = sd over months of ln(RV_m). Dispersion of realized risk.
- ES1n = -(mean of the k smallest daily excess returns) / sd(excess), k = ceil(0.01 * n).
  Daily tail in sigma units.
- ES5m = same on calendar-month excess returns (prod(1 + total) - prod(1 + cash); all months of
  the fold, partial months included), k = ceil(0.05 * months), divided by sd(excess) * sqrt(21).
- MDDn = MDD / (sd(total) * sqrt(252)).
- Also: worst day, worst month, share of months with RV_m above 1.5 * vol, largest calendar-year
  realized vol divided by vol, ln(vol / target) for vt books, mean/median/min/max held gross,
  trades per year, turnover per year, trading cost and financing-spread cost per year.

Comparison of a volatility-target book with its matched constant book:

- R_D = D_vt / D_cl (below 1: steadier risk)
- T = 1 - ES1n_vt / ES1n_cl (above 0: smaller daily tail)
- dS = Sharpe_vt - Sharpe_cl
- Secondary: T_m = 1 - ES5m_vt / ES5m_cl, R_MDD = MDDn_vt / MDDn_cl

## 7. Inference

- Paired circular block bootstrap over calendar months, block length 6 months, 5000 resamples.
- Months of the fold in order 0..M-1 (partial months included). B = ceil(M / 6).
  starts = numpy.random.default_rng(seed).integers(0, M, size=(5000, B)); a resample is the
  months s, s+1, ..., s+5 (mod M) for each start in order, truncated to the first M months.
  Daily returns of all books for those months are concatenated in that order.
- Seeds: main fold 0, fresh fold 1.
- All statistics are recomputed from the resampled daily series. Intervals are the 2.5 and 97.5
  percentiles of the finite draws.
- Pooling across folds: weight_f = n_f / (n_main + n_fresh). Pooled statistic = sum of
  weight_f * statistic_f. Pooled draw i = sum of weight_f * draw_f,i.
- Six-month blocks are used because volatility clusters across months; one-month blocks would
  understate the variability of D, MDD and the tail statistics.

## 8. Decision rules

Step 1 (plant), main fold only
- ADOPT S1 only if both hold: the 95% interval of Sharpe_s1l - Sharpe_cl20 lies above 0,
  and the largest gross of s1l is at most 3.0.
- Otherwise the plant stays 60/40.
- Expected: fails both. Task 13 gives Sharpe 0.49 against 0.64 before financing; the spread costs
  a low-volatility book more (drag in Sharpe units about b * (1/vol_base - 1/target): about 0.2
  for S1 against 0.06 for 60/40 at 20%), and S1 needs a gross of about 3.6.

Step 2 (volatility target), vt20 against cl20
- K1 regulation, in each fold: upper end of the 95% interval of R_D below 1.
- K2 no tail harm, pooled: T at least -0.05.
- K3 no Sharpe harm, pooled: dS at least -0.10.

| Verdict | Condition | Decision |
|---|---|---|
| ADDS VALUE | K1, K2, K3 and lower end of the 95% interval of pooled dS above 0 | keep |
| REGULATES RISK | K1, K2, K3 | keep |
| NO BENEFIT | any of K1, K2, K3 fails | drop: constant leverage |

Why no-harm margins instead of a required tail gain: at matched average volatility the textbook
gain is small against sampling noise. In simulation the monthly tail measure improves by 2 to 6%
with a standard deviation near 7% (pooled), so a required gain of 10% in both folds would pass
at most about 10% of the time even when volatility targeting works. The daily tail measure T
separates the two worlds that matter: about +0.09 when losses cluster in volatile periods, about
-0.11 to -0.22 when crashes arrive out of calm (the case where a volatility target is levered up
at the worst time).

Operating characteristics of the keep rule (1000 synthetic histories per model, same fold lengths):

| World | P(K1 both) | T mean (sd) | dS mean (sd) | P(keep) | P(ADDS VALUE) |
|---|---|---|---|---|---|
| iid, no clustering | 0.07 | -0.03 (0.02) | -0.03 (0.03) | 0.07 | 0.00 |
| GARCH, mean proportional to vol | 1.00 | +0.09 (0.05) | -0.02 (0.07) | 0.91 | 0.01 |
| GARCH, constant mean | 1.00 | +0.10 (0.05) | +0.05 (0.08) | 0.99 | 0.08 |
| GARCH plus jumps | 1.00 | -0.11 (0.06) | -0.03 (0.10) | 0.14 | 0.00 |
| GARCH plus jumps out of calm | 1.00 | -0.22 (0.05) | -0.18 (0.08) | 0.00 | 0.00 |

Consequences read off the table: the test cannot show a Sharpe gain of realistic size
(ADDS VALUE is rare even when true), and it does detect the harmful case.

## 9. Report-only items (no decision, no selection)

- R1: vt40 against cl40, same statistics and the same rule, labelled secondary.
- R2: neighbours of vt20, each against its own matched constant book, point estimates only:
  lambda 0.90 and 0.97 (band 0.10); band 0 and 0.20 (lambda 0.94).
- R3: spread stress, vt20 and cl20 with b = 0.03.
- R4: main fold split before 2017 and from 2017 (statistics only).
- R5: calendar-year returns and realized vol for base, cl20, vt20; positions and trade sizes
  implied at 200 and 2000 USD with 3x funds (equity 0.6 * L / 3, bond 0.4 * L / 3, rest cash).

Configuration count: 2 decision-bearing (s1l, vt20), 6 report-only (vt40, four neighbours,
one stress). Project total rises from about 500 to about 508.

## 10. Predictions, written before the run

- G1 passes exactly. L_c for cl20 on the main fold is 1.8 to 1.9; cl20 Sharpe about 0.58.
- vt20 realized vol 20 to 22%; 20 to 35 trades a year; trading cost 0.3 to 0.4% a year.
- R_D between 0.60 and 0.75 in both folds: K1 passes.
- Pooled T between -0.05 and +0.10. Real markets have both clustering and jumps, so K2 is the
  criterion most likely to fail.
- Pooled dS between -0.05 and +0.08.
- Verdict: REGULATES RISK about 70%, NO BENEFIT about 20%, ADDS VALUE about 10%.
- vt40 runs near the cap: realized vol 28 to 33%, well under its target.
- Step 1: the plant stays 60/40.

## 11. What follows

- Keep: vt20 is the controller under step 3 (trend filter) and paper tracking of vt20 starts.
- Drop: the plant runs at constant leverage; step 3 is tested on that.
- Either way, task 15 maps the book to real 2x and 3x funds and validates b on their history.
- The locked year is run once, after steps 3 and 4, on the fully specified strategy.
- Live trading of the 200 USD starts after that run. Gates for any larger money are unchanged.

## 12. Clarifications

Added 2026-11-11, before any evaluation. None enters K1, K2, K3 or the step 1 rule.

- C1, spread stress (R3): positions stay fixed. vt20 keeps its leverage path and cl20 keeps the
  L_c matched at b = 0.015; only the financing term changes. Do not re-match. Print the
  resulting sd ratio.
- C2, worst day and worst month: excess returns. Worst day = smallest daily excess return.
  Worst month = smallest calendar-month excess return as defined for ES5m (all months of the
  fold, partial months included).
- C3, calendar-year realized vol: RV_y = sqrt(252 * mean of excess_d^2) over the days of the
  calendar year (root mean square on excess returns, the same form as RV_m). The metric "largest
  calendar-year realized vol divided by vol" uses only years with at least 126 returns. The R5
  table prints every year with its number of days. Calendar-year returns in R5 are total
  returns, prod(1 + total) - 1.
- C4, trades per year: a trade date is an execution date on which the weight vector changes
  (sum_i |weights_i,d - weights_i,d-1| above 1e-12). Count dates, not asset orders. Exclude the
  initial entry. Divide by n / 252. Turnover per year and cost per year follow task 13 (mean
  daily value * 252) and include the initial entry.
- C5, leverage statistics (mean, median, min, max held gross): over all return days except the
  first execution date, where held is zero by construction.
- C6, positions and trades (R5): books vt20 and cl20, each fold, fixed balances of 200 and
  2000 USD (no compounding), L = held gross on the days of C5. Positions: median, min and max of
  equity fund = balance * 0.6 * L / 3, bond fund = balance * 0.4 * L / 3 and
  cash = balance * (1 - L / 3). Trades, on the trade dates of C4: number per year, and median
  and max of trade size = balance * |change in L| / 3 (both funds together).

## 13. Amendments

A1, 2026-11-11, gate G2 (vfinx against spy). Made after the registered gate failed and before
any levered book, matching, bootstrap or verdict was computed.

- Outcome as registered: daily correlation 0.967138 against a minimum of 0.97: FAIL.
  Annualized mean difference -0.16 percentage points against a limit of 0.5: pass.
  vfitx against ief: pass.
- Why the threshold was wrong: 0.97 was a guess, not a derivation. The fund is priced at the
  4:00 pm index close; the spy close is a traded price (later session close, bid-ask bounce,
  premium). Let e be the gap between the two closes. Then spy_t = index_t + e_t - e_(t-1), and
  daily correlation is about 1 - var(e) / var(daily return). A gap worth 3 to 4% of daily
  variance (about fifteen minutes of trading) gives 0.96 to 0.97. The threshold sat on the
  expected value.
- The same model says the gap telescopes: over 5 days only the two end-point gaps remain, so
  5-day correlation is about 1 - var(e) / (5 * var(daily return)), near 0.993. Independent daily
  errors (bad data) would leave the 5-day correlation near 0.967. A date shift would show as a
  large lead or lag correlation.
- Amended G2 for vfinx against spy, same window (2000-01-04 to 2005-09-30). PASS if all hold:
  (a) annualized mean return difference within 0.5 percentage points (unchanged);
  (b) daily return correlation at least 0.95;
  (c) 5-day return correlation at least 0.985, on consecutive non-overlapping blocks of 5 daily
      returns starting at the first common return date, compounded, incomplete last block dropped;
  (d) |corr(vfinx_t, spy_(t-1))| and |corr(vfinx_t, spy_(t+1))| both below 0.20.
- Prediction: (c) between 0.990 and 0.996; (d) both below 0.10.
- If the amended gate fails: STOP. This gate is not amended again. The fresh fold would be
  rebuilt on other data under a new pre-registration.
- The vfitx against ief check and everything else in sections 1 to 12 are unchanged.
- The report prints both the registered and the amended outcome and lists under Deviations:
  "D1: G2 amended (section 13, A1) after the registered gate failed".
