"""Does the variance ratio R_w = sigma_w^2 / sigma0^2 really follow the F
distribution when the ARMA parameters are ESTIMATED from the training part?

The paper derives R_w ~ F(n, n0) assuming the ARMA parameters are known.
In practice phi, theta and sigma0^2 are estimated. This Monte Carlo study
checks how close the empirical distribution of R_w is to F, using the same
code as the benchmark (src/ivr.py).

Each replication simulates a Gaussian ARMA series with NO anomaly:
a training part of n0 points followed by a test part. One test window of n
points (the first n test points) is scored per replication and per n, so the
R_w values within a setting are independent. R_w is computed four ways:

  A  true phi/theta, TRUE sigma^2 as reference      -> n*R_w ~ chi2(n) exactly
  B  true phi/theta, sigma0^2 estimated on training -> R_w ~ F(n, n0) exactly
  C  phi/theta/sigma0^2 estimated, order known      -> approx. F (the question)
  D  as C, but the ARMA order chosen by AIC (p,q <= 3), as in practice
     (only for the first model, it is slow: 15 fits per replication)

For each setting it reports: the Kolmogorov-Smirnov test against the
theoretical distribution (for C and D both F(n, n0) and F(n, n0 - k), k =
number of estimated const/AR/MA parameters), the mean of R_w against the F
mean, and the empirical rejection rate (one-sided upper and two-sided) at
alpha = 0.01, 0.05, 0.10 with Wilson 95% intervals.

Outputs in results/simulation/f_check/:
  ratios.csv     every simulated R_w (model, n0, n, version, rep, ratio, k)
  summary.csv    one row per (model, n0, n, version, reference df)
  figures/qq.png               QQ plots, version C vs F(n, n0 - k), n = 20
  figures/false_alarm_vs_n0.png two-sided rejection rate at alpha = 0.05

Usage: python 05_f_check.py [--reps 2000] [--reps-aic 1000] [--jobs 10]
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
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/ (ivr.py)
from ivr import D_ORDER, fit_reference_model, score_window

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = {  # name -> (phi list, theta list)
    "ARMA(1,1) phi=0.8 theta=0.6": ([0.8], [0.6]),
    "AR(1) phi=0.95": ([0.95], []),
    "MA(1) theta=0.9": ([], [0.9]),
}
AIC_MODEL = "ARMA(1,1) phi=0.8 theta=0.6"
N0_LEVELS = [200, 500, 1000, 5000]
WINDOW_LEVELS = [20, 100]
ALPHAS = [0.01, 0.05, 0.10]
SIGMA = 1.0
OUT = Path("results/simulation/f_check")


def simulate(phi, theta, n_samples, seed):
    process = ArmaProcess(np.r_[1.0, -np.array(phi)], np.r_[1.0, np.array(theta)])
    rng = np.random.default_rng(seed)
    return process.generate_sample(nsample=n_samples, scale=SIGMA, distrvs=rng.standard_normal, burnin=500)


def true_model(phi, theta, n0):
    """Model dict in the shape src/ivr.py uses, holding the TRUE
    const (0), AR and MA parameters fixed."""
    order = (len(phi), D_ORDER, len(theta))
    names = list(ARIMA(np.zeros(50), order=order).param_names)
    names.remove("sigma2")
    return {"order": order, "param_names": names, "fixed_params": np.r_[0.0, phi, theta], "reference_df": n0}


def fit_known_order(train, order):
    """Estimate const/AR/MA/sigma0^2 at the TRUE order (version C), same
    dict shape and df convention as algorithm.fit_reference_model."""
    fit = ARIMA(train, order=order).fit()
    names, params = list(fit.param_names), np.asarray(fit.params, dtype=float)
    i = names.index("sigma2")
    fixed = np.delete(params, i)
    return {"order": order, "param_names": [n for n in names if n != "sigma2"], "fixed_params": fixed,
            "reference_sigma2": float(params[i]), "reference_df": len(train) - len(fixed)}


def run_rep(task):
    model_name, n0, rep, with_aic, seed = task
    phi, theta = MODELS[model_name]
    y = simulate(phi, theta, n0 + max(WINDOW_LEVELS), seed)
    train, test = y[:n0], y[n0:]
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tm = true_model(phi, theta, n0)
        sigma0_true_params = score_window(train, tm)[0]  # B: sigma0^2 estimated with the true phi/theta
        try:
            mc = fit_known_order(train, tm["order"])
        except Exception:
            mc = None
        try:
            md = fit_reference_model(train, 3, 3) if with_aic else None
        except Exception:
            md = None
        for n in WINDOW_LEVELS:
            w = test[:n]
            s_true = score_window(w, tm)[0]
            base = {"model": model_name, "n0": n0, "n": n, "rep": rep}
            rows.append({**base, "version": "A", "ratio": s_true / SIGMA**2, "k": 0})
            rows.append({**base, "version": "B", "ratio": s_true / sigma0_true_params, "k": 0})
            if mc is not None:
                rows.append({**base, "version": "C", "ratio": score_window(w, mc)[0] / mc["reference_sigma2"],
                             "k": len(mc["fixed_params"])})
            if md is not None:
                rows.append({**base, "version": "D", "ratio": score_window(w, md)[0] / md["reference_sigma2"],
                             "k": len(md["fixed_params"]), "aic_order": str(md["order"])})
    return rows


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return c - h, c + h


def summarize(ratios: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (model, n0, n, version), g in ratios.groupby(["model", "n0", "n", "version"]):
        r = g["ratio"].to_numpy()
        if version == "A":
            refs = [("chi2(n)/n", stats.chi2(n, scale=1.0 / n))]
        elif version == "B":
            refs = [("F(n, n0)", stats.f(n, n0))]
        else:
            k = int(round(g["k"].median()))
            refs = [("F(n, n0)", stats.f(n, n0)), (f"F(n, n0-k)", stats.f(n, n0 - k))]
        for ref_name, dist in refs:
            ks = stats.kstest(r, dist.cdf)
            row = {"model": model, "n0": n0, "n": n, "version": version, "reference": ref_name, "reps": len(r),
                   "mean_ratio": r.mean(), "theory_mean": dist.mean(), "ks_stat": ks.statistic, "ks_pvalue": ks.pvalue}
            for a in ALPHAS:
                up = r > dist.ppf(1 - a)
                two = (r < dist.ppf(a / 2)) | (r > dist.ppf(1 - a / 2))
                for name, flag in (("one_sided", up), ("two_sided", two)):
                    lo, hi = wilson(flag.sum(), len(r))
                    row[f"{name}_{a:g}"] = flag.mean()
                    row[f"{name}_{a:g}_lo"], row[f"{name}_{a:g}_hi"] = lo, hi
            out.append(row)
    return pd.DataFrame(out)


def plot_qq(ratios: pd.DataFrame, path: Path) -> None:
    models, n = list(MODELS), 20
    fig, axes = plt.subplots(len(models), len(N0_LEVELS), figsize=(3.2 * len(N0_LEVELS), 3.0 * len(models)))
    for i, model in enumerate(models):
        for j, n0 in enumerate(N0_LEVELS):
            ax = axes[i, j]
            for version, color in (("B", "#8a8a8a"), ("C", "#c0392b")):
                g = ratios[(ratios.model == model) & (ratios.n0 == n0) & (ratios.n == n) & (ratios.version == version)]
                if g.empty:
                    continue
                k = int(round(g["k"].median()))
                dist = stats.f(n, n0 - k)
                r = np.sort(g["ratio"].to_numpy())
                q = dist.ppf((np.arange(1, len(r) + 1) - 0.5) / len(r))
                ax.plot(q, r, ".", ms=2, color=color, label=f"{version}" + (" (true params)" if version == "B" else " (estimated)"))
            lim = [0.1, 4]
            ax.plot(lim, lim, "k--", lw=0.8)
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
            ax.set_title(f"{model}\nn0={n0}", fontsize=8, loc="left")
            if i == len(models) - 1:
                ax.set_xlabel("F(n, n0-k) quantile")
            if j == 0:
                ax.set_ylabel(f"empirical $R_w$ quantile (n={n})")
            if i == 0 and j == 0:
                ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_false_alarm(summary: pd.DataFrame, path: Path, alpha: float = 0.05) -> None:
    models = list(MODELS)
    fig, axes = plt.subplots(len(WINDOW_LEVELS), len(models), figsize=(4.2 * len(models), 3.4 * len(WINDOW_LEVELS)), sharey=True)
    styles = {("B", "F(n, n0)"): ("#8a8a8a", "B: true params, F(n,n0)"),
              ("C", "F(n, n0)"): ("#2c6fbb", "C: estimated, F(n,n0)"),
              ("C", "F(n, n0-k)"): ("#c0392b", "C: estimated, F(n,n0-k)"),
              ("D", "F(n, n0-k)"): ("#1a1a1a", "D: AIC order, F(n,n0-k)")}
    col = f"two_sided_{alpha:g}"
    for i, n in enumerate(WINDOW_LEVELS):
        for j, model in enumerate(models):
            ax = axes[i, j]
            ax.axhspan(0.5 * alpha, 1.5 * alpha, color="#d8e6f9", alpha=0.6, lw=0)  # Bradley (1978) liberal band
            ax.axhline(alpha, color="k", ls="--", lw=0.9)
            for (version, ref), (color, label) in styles.items():
                g = summary[(summary.model == model) & (summary.n == n) & (summary.version == version) & (summary.reference == ref)].sort_values("n0")
                if g.empty:
                    continue
                ax.errorbar(g["n0"], g[col], yerr=[g[col] - g[col + "_lo"], g[col + "_hi"] - g[col]],
                            color=color, marker="o", ms=4, lw=1.4, capsize=2, label=label)
            ax.set_xscale("log")
            ax.set_xticks(N0_LEVELS); ax.set_xticklabels([str(v) for v in N0_LEVELS]); ax.minorticks_off()
            ax.set_title(f"{model}, n={n}", fontsize=9, loc="left")
            if i == len(WINDOW_LEVELS) - 1:
                ax.set_xlabel("training length $n_0$")
            if j == 0:
                ax.set_ylabel(f"two-sided rejection rate ($\\alpha$={alpha:g})")
    axes[0, -1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--reps-aic", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)

    # fixed, reproducible seed per (model, n0, rep)
    tasks = [(m, n0, rep, m == AIC_MODEL and rep < args.reps_aic, 1_000_000 * mi + 100_000 * ni + rep)
             for mi, m in enumerate(MODELS) for ni, n0 in enumerate(N0_LEVELS) for rep in range(args.reps)]
    print(f"{len(tasks)} replications on {args.jobs} processes", flush=True)
    with Pool(args.jobs) as pool:
        rows = []
        for i, chunk in enumerate(pool.imap_unordered(run_rep, tasks, chunksize=20), 1):
            rows += chunk
            if i % 500 == 0 or i == len(tasks):
                print(f"{i}/{len(tasks)} done", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    ratios = pd.DataFrame(rows)
    ratios.to_csv(out / "ratios.csv", index=False)
    summary = summarize(ratios)
    summary.to_csv(out / "summary.csv", index=False)
    plot_qq(ratios, out / "figures" / "qq.png")
    plot_false_alarm(summary, out / "figures" / "false_alarm_vs_n0.png")

    cols = ["model", "n0", "n", "version", "reference", "reps", "mean_ratio", "theory_mean", "ks_pvalue", "two_sided_0.05", "one_sided_0.05"]
    pd.set_option("display.width", 220)
    print(summary[cols].round(4).to_string(index=False))
    print(f"\n-> {out}/")


if __name__ == "__main__":
    main()
