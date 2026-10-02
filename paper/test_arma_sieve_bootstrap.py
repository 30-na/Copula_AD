"""ARMA-sieve bootstrap calibration, as a SEPARATE experiment from
test_block_bootstrap.py (whole-chunk block bootstrap), so the two can be
compared directly.

Procedure:
  1. Fit the reference model (same as always): ARMA(p,0,q) on seizure 1 /
     channel 1 training data -> phi_hat, theta_hat, sigma0^2.
  2. Get real out-of-sample residuals from a long, confirmed-normal
     stretch of test data, scored with that reference model.
  3. Fit an ARMA(p', q') -- AR AND MA, AIC-selected -- to THOSE residuals
     (the "sieve" model, characterizing the leftover dependency we found).
  4. Sieve bootstrap: resample the sieve model's own fitted innovations
     (preserves the real, measured marginal shape), run them back through
     the fitted ARMA(p', q') recursion -> synthetic dependent residual
     sequences.
  5. Feed those (rescaled to sigma0 units) through the REFERENCE model's
     own ARMA(p,0,q) recursion -> full synthetic test series under H0.
  6. Score non-overlapping window_size windows of the synthetic series
     with the reference model -> empirical null distribution of the
     variance ratio T*.
  7. Take percentiles as new control limits; apply to the REAL test
     windows (same ones test_block_bootstrap.py scored) and report
     false-alarm / detection rates for direct comparison.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA

from algorithm import fit_reference_model, make_windows, score_window
from test_block_bootstrap import CONFIG, N_TRAIN, WINDOW_SIZE, SEIZURE_NUMBER, ALPHA, f_critical_values
from test_dependency_ablation import simulate_arma_from_innovations

SYNTHETIC_LENGTH = 500_000
BURN_IN = 2000


def fit_arma_sieve(residuals: np.ndarray, max_p: int = 3, max_q: int = 3) -> dict:
    """AIC-selected ARMA(p', q') fit to the residuals -- the "sieve" model."""
    best = None
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p == 0 and q == 0:
                continue
            print(f"  fitting ARMA({p},0,{q}) sieve candidate...", flush=True)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    warnings.simplefilter("ignore", UserWarning)
                    fit = ARIMA(residuals, order=(p, 0, q)).fit()
            except Exception:
                continue
            if not np.isfinite(fit.aic):
                continue
            print(f"    AIC={fit.aic:.1f}", flush=True)
            if best is None or fit.aic < best["aic"]:
                param_names = list(fit.param_names)
                params = np.asarray(fit.params, dtype=float)
                sigma_index = param_names.index("sigma2")
                fixed_params = np.delete(params, sigma_index)
                best = {
                    "order": (p, 0, q),
                    "param_names": [n for n in param_names if n != "sigma2"],
                    "fixed_params": fixed_params,
                    "aic": float(fit.aic),
                    "innovations": np.asarray(fit.resid, dtype=float),
                }
    return best


def empirical_type1_and_detection(window_results: pd.DataFrame, lower: float, upper: float) -> dict:
    actual_anomaly = window_results["actual_anomaly"].to_numpy()
    ratio = window_results["variance_ratio"].to_numpy()
    flagged = (ratio < lower) | (ratio > upper)
    normal_mask = actual_anomaly == 0
    anomaly_mask = actual_anomaly == 1
    n_normal, n_anomaly = normal_mask.sum(), anomaly_mask.sum()
    fa = (flagged & normal_mask).sum()
    det = (flagged & anomaly_mask).sum()
    return {"false_alarms": fa, "n_normal": n_normal, "detected": det, "n_anomaly": n_anomaly}


def main() -> None:
    values = pd.read_csv(f"data/mice/{CONFIG}/timeseries_seizure{SEIZURE_NUMBER}.csv")["value"].to_numpy(float)
    labels = pd.read_csv(f"data/mice/{CONFIG}/labels_seizure{SEIZURE_NUMBER}.csv")["is_anomaly"].to_numpy(int)
    window_results = pd.read_csv(f"results/mice/{CONFIG}/window_results_seizure{SEIZURE_NUMBER}.csv")

    train_values = values[:N_TRAIN]
    test_values = values[N_TRAIN:]
    model = fit_reference_model(train_values, 3, 3)
    sigma0 = np.sqrt(model["reference_sigma2"])
    print(f"Reference model: order={model['order']}, sigma0={sigma0:.4f}", flush=True)

    normal_stretch = test_values[:100_000]
    assert labels[N_TRAIN : N_TRAIN + 100_000].sum() == 0
    _, real_residuals_full = score_window(normal_stretch, model)
    real_residuals = real_residuals_full[:20_000]  # shorter stretch for the sieve order search (MLE cost scales with length)

    sieve = fit_arma_sieve(real_residuals, max_p=3, max_q=3)
    print(f"ARMA sieve model: order={sieve['order']}, AIC={sieve['aic']:.1f}", flush=True)

    rng = np.random.default_rng(0)
    n = SYNTHETIC_LENGTH + BURN_IN
    driving_noise = rng.choice(sieve["innovations"], size=n, replace=True)

    synthetic_residuals = simulate_arma_from_innovations(
        driving_noise, sieve["order"], sieve["fixed_params"], sieve["param_names"]
    )
    zeta = synthetic_residuals * sigma0

    y = simulate_arma_from_innovations(zeta, model["order"], model["fixed_params"], model["param_names"])
    y = y[BURN_IN:]

    windows = make_windows(y, WINDOW_SIZE)
    t_star = np.empty(len(windows))
    for i, w in enumerate(windows):
        sigma2_w, _ = score_window(w, model)
        t_star[i] = sigma2_w / model["reference_sigma2"]
    np.save("results/mice/figs/arma_sieve_t_star.npy", t_star)

    empirical_lower, empirical_upper = np.percentile(t_star, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    f_lower, f_upper = f_critical_values(WINDOW_SIZE, model["reference_df"], ALPHA)

    print(f"\nARMA-sieve bootstrap null (n_windows={len(windows)}):", flush=True)
    print(f"  control limits: [{empirical_lower:.4f}, {empirical_upper:.4f}]", flush=True)
    print(f"Theoretical F-distribution control limits: [{f_lower:.4f}, {f_upper:.4f}]", flush=True)

    r_f = empirical_type1_and_detection(window_results, f_lower, f_upper)
    r_sieve = empirical_type1_and_detection(window_results, empirical_lower, empirical_upper)

    print(f"\n-- Theoretical F-distribution (current pipeline) --", flush=True)
    print(f"   false alarms: {r_f['false_alarms']}/{r_f['n_normal']} ({r_f['false_alarms']/r_f['n_normal']:.2%})", flush=True)
    print(f"   detected:     {r_f['detected']}/{r_f['n_anomaly']} ({r_f['detected']/r_f['n_anomaly']:.2%})", flush=True)
    print(f"\n-- ARMA-sieve bootstrap --", flush=True)
    print(f"   false alarms: {r_sieve['false_alarms']}/{r_sieve['n_normal']} ({r_sieve['false_alarms']/r_sieve['n_normal']:.2%})", flush=True)
    print(f"   detected:     {r_sieve['detected']}/{r_sieve['n_anomaly']} ({r_sieve['detected']/r_sieve['n_anomaly']:.2%})", flush=True)
    print(f"\n(for comparison, the earlier block bootstrap: false alarms 68/1190 (5.71%), detected 64/100 (64.00%))", flush=True)


if __name__ == "__main__":
    main()
