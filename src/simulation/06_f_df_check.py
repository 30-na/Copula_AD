"""Which F fits R_w better when the ARMA is estimated: F(n, n0) or F(n, n0 - k)?

The data are Gaussian ARMA(1,1) phi=0.8 theta=0.6 with no anomaly. A fixed
ARMA(3,3) (no AIC) is fit on the first n0 points, so k = 7 estimated
parameters (const + 3 AR + 3 MA). One test window of n = 10 points (the
points right after training) gives one R_w per replication.

For each n0: a histogram of R_w with both theoretical F densities, the
Kolmogorov-Smirnov p-value against each, and the share of R_w above each
F's 95% point (should be 5%).

Outputs in results/simulation/f_check/df_comparison/:
  ratios.csv, summary.csv, figures/hist_n0_vs_n0_minus_k.png

Usage: python 06_f_df_check.py [--reps 2000] [--jobs 10]
"""

from __future__ import annotations

import argparse
import os
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/ (ivr.py)
from ivr import score_window
from importlib import import_module
_m = import_module("05_f_check")  # 05_f_check.py (same folder)
MODELS, fit_known_order, simulate = _m.MODELS, _m.fit_known_order, _m.simulate

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRUE_MODEL = "ARMA(1,1) phi=0.8 theta=0.6"
FIT_ORDER = (3, 0, 3)
N0_LEVELS = [50, 100, 200, 500]
WINDOW = 10
OUT = Path("results/simulation/f_check/df_comparison")


def run_rep(task):
    n0, rep, seed = task
    phi, theta = MODELS[TRUE_MODEL]
    y = simulate(phi, theta, n0 + WINDOW, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            m = fit_known_order(y[:n0], FIT_ORDER)
            r = score_window(y[n0:], m)[0] / m["reference_sigma2"]
        except Exception:
            return None
    if not np.isfinite(r):  # failed ARMA(3,3) fit
        return None
    return {"n0": n0, "rep": rep, "ratio": r, "k": len(m["fixed_params"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--replot", action="store_true", help="reuse ratios.csv, no simulation")
    args = ap.parse_args()

    if args.replot:
        ratios = pd.read_csv(OUT / "ratios.csv")
    else:
        tasks = [(n0, rep, 10_000_000 + 100_000 * ni + rep) for ni, n0 in enumerate(N0_LEVELS) for rep in range(args.reps)]
        print(f"{len(tasks)} replications on {args.jobs} processes", flush=True)
        rows = []
        with Pool(args.jobs) as pool:
            for i, r in enumerate(pool.imap_unordered(run_rep, tasks, chunksize=10), 1):
                if r is not None:
                    rows.append(r)
                if i % 500 == 0 or i == len(tasks):
                    print(f"{i}/{len(tasks)} done", flush=True)
        ratios = pd.DataFrame(rows)
    ratios = ratios[np.isfinite(ratios["ratio"])]
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    ratios.to_csv(OUT / "ratios.csv", index=False)

    summary = []
    fig, axes = plt.subplots(1, len(N0_LEVELS), figsize=(4.2 * len(N0_LEVELS), 3.6))
    for ax, n0 in zip(axes, N0_LEVELS):
        r = ratios.loc[ratios["n0"] == n0, "ratio"].to_numpy()
        k = int(ratios.loc[ratios["n0"] == n0, "k"].iloc[0])
        refs = {"F(n, n0)": stats.f(WINDOW, n0), "F(n, n0-k)": stats.f(WINDOW, n0 - k),
                # Akaike (final prediction error): R_w inflated by (n0 + k) / (n0 - k)
                "Akaike-corrected F(n, n0)": stats.f(WINDOW, n0, scale=(n0 + k) / (n0 - k))}
        xmax = np.percentile(r, 99.5)
        ax.hist(r, bins=np.linspace(0, xmax, 50), density=True, color="0.8", edgecolor="0.6", label="simulated $R_w$")
        x = np.linspace(1e-3, xmax, 400)
        for (name, dist), style in zip(refs.items(), ["-", "--", "-"]):
            ks = stats.kstest(r, dist.cdf).pvalue
            fa = float((r > dist.ppf(0.95)).mean())
            summary.append({"n0": n0, "k": k, "reference": name, "df2": dist.args[1], "reps": len(r),
                            "mean_ratio": r.mean(), "theory_mean": dist.mean(), "ks_pvalue": ks, "above_95pct_point": fa})
            ax.plot(x, dist.pdf(x), style, lw=1.8, label=f"{name}: KS p={ks:.3f}, above 95%: {fa:.1%}")
        ax.set_title(f"n0 = {n0}, n = {WINDOW}, k = {k}")
        ax.set_xlabel("$R_w$")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("density")
    fig.suptitle(f"Data: {TRUE_MODEL}; fitted ARMA(3,3), no AIC")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "hist_with_akaike.png", dpi=150)

    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / "summary.csv", index=False)
    print(summary.round(4).to_string(index=False))
    print(f"\n-> {OUT}/")


if __name__ == "__main__":
    main()
