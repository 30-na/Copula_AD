"""Diagnostic tests for the three assumptions the innovation-variance F-test
relies on, run against real mouse data instead of just asserting they fail.

1. Model adequacy: are standardized residuals actually white noise?
   -> Ljung-Box test on the reference model's own training residuals, and
      on a sample of "normal" (non-seizure) test-window residuals.
2. Gaussian innovations: are those residuals normally distributed?
   -> Jarque-Bera test + skewness/kurtosis on the same residuals.
3. Stationary baseline: does the innovation variance drift over the
   pre-seizure period even with no seizure present?
   -> Linear regression of each window's sigma2_w (already saved in
      window_results_*.csv, no recomputation) against window index,
      restricted to actual-normal windows.

Residuals for (1)/(2) are recomputed via algorithm.score_window (cheap --
one Kalman filter pass per window), reusing the already-extracted
timeseries_seizureN.csv so the raw .dat file is never touched again.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

from algorithm import fit_reference_model, make_windows, score_window


def residual_diagnostics(residuals: np.ndarray, label: str) -> dict:
    lb = acorr_ljungbox(residuals, lags=[10], return_df=True)
    lb_pvalue = float(lb["lb_pvalue"].iloc[0])
    jb_stat, jb_pvalue = stats.jarque_bera(residuals)
    return {
        "label": label,
        "n": len(residuals),
        "ljung_box_p": lb_pvalue,
        "white_noise_ok": lb_pvalue > 0.05,
        "jarque_bera_p": float(jb_pvalue),
        "gaussian_ok": jb_pvalue > 0.05,
        "skew": float(stats.skew(residuals)),
        "kurtosis_excess": float(stats.kurtosis(residuals)),
    }


def stationarity_check(window_results: pd.DataFrame, n_blocks: int = 10) -> dict:
    normal = window_results[window_results.actual_anomaly == 0]
    x = normal["window_index"].to_numpy(dtype=float)
    y = np.log(normal["sigma2_w"].to_numpy(dtype=float))
    slope, intercept, r, p, se = stats.linregress(x, y)

    # non-monotonic drift: equal-variance test across chronological blocks
    # (catches bursty/local instability a linear trend would average out)
    blocks = np.array_split(y, n_blocks)
    levene_stat, levene_p = stats.levene(*blocks)
    block_means = [b.mean() for b in blocks]

    return {
        "n_normal_windows": len(normal),
        "log_variance_trend_slope": slope,
        "trend_p_value": p,
        "trend_significant": p < 0.05,
        "r_squared": r**2,
        "levene_p": levene_p,
        "block_variance_heterogeneous": levene_p < 0.05,
        "block_means": block_means,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="2000hz_pre60s_ntrain1000")
    parser.add_argument("--seizure-index", type=int, default=0)
    parser.add_argument("--channel", type=int, default=0, help="0-indexed (0=ch1).")
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--max-p", type=int, default=3)
    parser.add_argument("--max-q", type=int, default=3)
    parser.add_argument("--n-sample-windows", type=int, default=40, help="How many normal test windows to sample residuals from.")
    args = parser.parse_args()

    seizure_number = args.seizure_index + 1
    tag = f"seizure{seizure_number}" + (f"_ch{args.channel + 1}" if args.channel != 0 else "")

    values = pd.read_csv(f"data/mice/{args.config}/timeseries_{tag}.csv")["value"].to_numpy(float)
    window_results = pd.read_csv(f"results/mice/{args.config}/window_results_{tag}.csv")

    train_values = values[: args.n_train]
    test_values = values[args.n_train :]

    print(f"=== {args.config} / {tag} / window={args.window_size} ===\n")

    model = fit_reference_model(train_values, args.max_p, args.max_q)
    print(f"Reference model: order={model['order']}, AIC={model['aic']:.1f}\n")

    # (1) + (2): training in-sample residuals
    _, train_residuals = score_window(train_values, model)
    result_train = residual_diagnostics(train_residuals, "training (in-sample)")

    # sample residuals from normal test windows
    windows = make_windows(test_values, args.window_size)
    normal_flags = window_results["actual_anomaly"].to_numpy()[: len(windows)] == 0
    normal_indices = np.where(normal_flags)[0]
    rng = np.random.default_rng(0)
    sample_indices = rng.choice(normal_indices, size=min(args.n_sample_windows, len(normal_indices)), replace=False)
    pooled_test_residuals = np.concatenate([score_window(windows[i], model)[1] for i in sample_indices])
    result_test = residual_diagnostics(pooled_test_residuals, f"{len(sample_indices)} sampled normal test windows (pooled)")

    for r in [result_train, result_test]:
        print(f"-- {r['label']} (n={r['n']}) --")
        print(f"   Ljung-Box p={r['ljung_box_p']:.4f}  -> {'white noise OK' if r['white_noise_ok'] else 'AUTOCORRELATED (model misspecified)'}")
        print(f"   Jarque-Bera p={r['jarque_bera_p']:.4f}  -> {'Gaussian OK' if r['gaussian_ok'] else 'NOT GAUSSIAN'}  (skew={r['skew']:.2f}, excess kurtosis={r['kurtosis_excess']:.2f})")
        print()

    # (3): stationarity of the baseline variance
    stat = stationarity_check(window_results)
    print(f"-- Stationarity of innovation variance across {stat['n_normal_windows']} normal windows --")
    print(f"   log(sigma2_w) vs window index (linear trend): slope={stat['log_variance_trend_slope']:.5f}, p={stat['trend_p_value']:.4f}, R^2={stat['r_squared']:.3f}")
    print(f"   -> {'SIGNIFICANT LINEAR DRIFT' if stat['trend_significant'] else 'no significant linear trend'}")
    print(f"   Levene's test across 10 time-blocks: p={stat['levene_p']:.4f} -> {'HETEROGENEOUS (bursty/non-stationary)' if stat['block_variance_heterogeneous'] else 'homogeneous'}")
    print(f"   block mean log(sigma2_w): {[f'{m:.2f}' for m in stat['block_means']]}")


if __name__ == "__main__":
    main()
