"""Operating characteristics of the task 14 keep rule on synthetic returns.

Run before any real data is evaluated. No market data is read.
Usage: python scripts/sim_vol_target_power.py [replications]

Books (same rules as prereg_task14.md, simplified to one base series, cash = 0):
  VT: leverage = min(cap, target / ewma_vol), 10% no-trade band, signal t held on t+2.
  CL: constant leverage matched ex post to the realized vol of VT.
Keep rule:
  K1 each fold : 97.5% bootstrap bound of D_VT / D_CL below 1
                 (D = std of log monthly realized vol; 6-month circular blocks)
  K2 pooled    : T >= -0.05, T = 1 - ES1n_VT / ES1n_CL (daily 1% expected shortfall in sigma units)
  K3 pooled    : Sharpe_VT - Sharpe_CL >= -0.10
"""

import sys

import numpy as np


lam, cap, band, spread, trade_cost = 0.94, 3.0, 0.10, 0.015, 0.0005
month_days, warm = 21, 252
folds = {"main": 5053, "fresh": 3255}
models = ("iid", "garch_prop", "garch_const", "jump_prop", "jump_calm_prop")


def simulate(model, n, reps, rng, sharpe=0.6, vol=0.11, nu=6):
    """Daily base returns. garch: GJR(1,1), t innovations. prop: mean proportional to vol.
    jump: two negative jumps a year of 3 to 5%, not scaled by vol; calm: more likely when vol is low."""
    burn = 500
    hbar = vol ** 2 / 252
    z = rng.standard_t(nu, size=(reps, n + burn)) * np.sqrt((nu - 2) / nu)
    out = np.empty((reps, n + burn))
    if model == "iid":
        out[:] = np.sqrt(hbar) * z + sharpe * vol / 252
        return out[:, burn:]
    jump = model.startswith("jump")
    alpha, gamma, beta = 0.03, 0.10, 0.90
    level = hbar * (0.80 if jump else 1.0)
    omega = level * (1 - alpha - gamma / 2 - beta)
    h, e = np.full(reps, level), np.zeros(reps)
    uniform = rng.random((reps, n + burn))
    size = 0.03 + 0.01 * np.abs(rng.standard_normal((reps, n + burn)))
    for t in range(n + burn):
        h = omega + (alpha + gamma * (e < 0)) * e * e + beta * h
        e = np.sqrt(h) * z[:, t]
        drift = sharpe * vol / 252 if model.endswith("const") else sharpe / np.sqrt(252) * np.sqrt(h)
        value = e + drift
        if jump:
            chance = 2 / 252 * (np.clip(level / h, 0.25, 4) if "calm" in model else 1.0)
            hit = uniform[:, t] < chance
            value = value - hit * size[:, t] + 2 / 252 * 0.038
            e = e - hit * size[:, t]
        out[:, t] = value
    return out[:, burn:]


def books(base, target):
    reps, n = base.shape
    squares = base * base
    variance = squares[:, :60].mean(axis=1)
    decided = np.empty((reps, n))
    previous = None
    for t in range(n):
        if t >= 60:
            variance = lam * variance + (1 - lam) * squares[:, t]
        wanted = np.minimum(cap, target / np.sqrt(252 * variance))
        current = wanted if previous is None or t <= warm else np.where(np.abs(wanted / previous - 1) > band, wanted, previous)
        decided[:, t] = current
        previous = current
    # Decided at close t, executed at close t+1, held on day t+2.
    held = decided[:, warm:n - 2]
    returns = base[:, warm + 2:]
    traded = np.abs(decided[:, warm + 1:n - 1] - decided[:, warm:n - 2])  # paid on the return day (one day late)
    vt = held * returns - trade_cost * traded - np.maximum(held - 1, 0) * spread / 252
    matched = vt.std(axis=1, ddof=1) / returns.std(axis=1, ddof=1)
    cl = matched[:, None] * returns - np.maximum(matched[:, None] - 1, 0) * spread / 252
    return vt, cl, held, traded


def month_table(values):
    reps, n = values.shape
    months = n // month_days
    cube = values[:, :months * month_days].reshape(reps, months, month_days)
    return np.sqrt(252 * (cube ** 2).mean(axis=2)), cube.sum(axis=2), (cube ** 2).sum(axis=2)


def shortfall(values, share, sd):
    count = int(np.ceil(share * values.shape[-1]))
    return -np.sort(values, axis=-1)[..., :count].mean(axis=-1) / sd


def evaluate(model, n, target, reps, rng, resamples=600):
    base = simulate(model, n + warm + 2, reps, rng)
    vt, cl, held, traded = books(base, target)
    rv_vt, sum_vt, sq_vt = month_table(vt)
    rv_cl, sum_cl, sq_cl = month_table(cl)
    months = rv_vt.shape[1]
    days = months * month_days
    sd_vt, sd_cl = vt.std(axis=1, ddof=1), cl.std(axis=1, ddof=1)
    out = {"days": days,
           "d_ratio": np.log(rv_vt).std(axis=1, ddof=1) / np.log(rv_cl).std(axis=1, ddof=1),
           "sharpe_diff": (vt.mean(axis=1) / sd_vt - cl.mean(axis=1) / sd_cl) * np.sqrt(252),
           "tail_gain": 1 - shortfall(vt, 0.01, sd_vt) / shortfall(cl, 0.01, sd_cl),
           "vol": sd_vt * np.sqrt(252), "leverage": held.mean(axis=1),
           "at_cap": (held >= cap - 1e-9).mean(axis=1), "turnover": traded.mean(axis=1) * 252,
           "trades": (traded > 0).mean(axis=1) * 252}
    blocks = int(np.ceil(months / 6))
    k1 = np.empty(reps, dtype=bool)
    diff_draws = np.empty((reps, resamples))
    for i in range(reps):
        starts = rng.integers(0, months, size=(resamples, blocks))
        index = ((starts[:, :, None] + np.arange(6)[None, None, :]) % months).reshape(resamples, -1)[:, :months]
        ratio = np.log(rv_vt[i])[index].std(axis=1, ddof=1) / np.log(rv_cl[i])[index].std(axis=1, ddof=1)
        k1[i] = np.percentile(ratio, 97.5) < 1

        def sharpe(sums, squares):
            a, b = sums[index].sum(axis=1), squares[index].sum(axis=1)
            return a / days / np.sqrt((b - a * a / days) / (days - 1)) * np.sqrt(252)

        diff_draws[i] = sharpe(sum_vt[i], sq_vt[i]) - sharpe(sum_cl[i], sq_cl[i])
    out["k1"], out["diff_draws"] = k1, diff_draws
    return out


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    rng = np.random.default_rng(14)
    print(f"{reps} replications per model; folds {folds}; lambda {lam}, cap {cap}, band {band}, spread {spread}.")
    for target in (0.20, 0.40):
        print(f"target vol {target:.0%}")
        print(f"{'model':<15} {'D ratio':>8} {'K1 both':>8} {'T mean':>7} {'T sd':>6} {'dS mean':>8} {'dS sd':>6} "
              f"{'KEEP':>6} {'ADDS':>6} {'vol':>6} {'lev':>5} {'cap%':>5} {'trades/y':>8} {'turn/y':>7}")
        for model in models:
            parts = {name: evaluate(model, n, target, reps, rng) for name, n in folds.items()}
            total = sum(part["days"] for part in parts.values())
            pooled = lambda key: sum(part["days"] * part[key] for part in parts.values()) / total
            tail, diff = pooled("tail_gain"), pooled("sharpe_diff")
            k1 = parts["main"]["k1"] & parts["fresh"]["k1"]
            keep = k1 & (tail >= -0.05) & (diff >= -0.10)
            adds = keep & (np.percentile(pooled("diff_draws"), 2.5, axis=1) > 0)
            main_part = parts["main"]
            print(f"{model:<15} {np.median(main_part['d_ratio']):>8.2f} {k1.mean():>8.2f} {tail.mean():>+7.3f} {tail.std():>6.3f} "
                  f"{diff.mean():>+8.3f} {diff.std():>6.3f} {keep.mean():>6.2f} {adds.mean():>6.2f} "
                  f"{main_part['vol'].mean():>6.1%} {main_part['leverage'].mean():>5.2f} {main_part['at_cap'].mean():>5.0%} "
                  f"{main_part['trades'].mean():>8.1f} {main_part['turnover'].mean():>7.1f}")


if __name__ == "__main__":
    main()
