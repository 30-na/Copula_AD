"""Experiment: does requiring in-sample residual whiteness (Ljung-Box pass)
when picking the ARMA order -- instead of picking by lowest AIC alone --
improve anything? Tested on 3 seizures (channel 1, window=100, n_train=1000,
60s-before config) without touching algorithm.py, so the paper's core
method and all cached results are untouched. This is a standalone
before/after comparison.

For each seizure:
  - baseline: algorithm.fit_reference_model (current method, AIC only)
  - candidate: same (p,q) grid, but prefer the lowest-AIC order among those
    whose IN-SAMPLE training residuals pass Ljung-Box (p > 0.05); fall back
    to the best-available (highest Ljung-Box p) if none pass.

Both models are then scored on the same test windows, and compared on:
  - in-sample Ljung-Box p (trivially easy to pass, included for completeness)
  - out-of-sample Ljung-Box p, pooled from 40 sampled normal windows (the
    diagnostic that actually matters)
  - fraction of all test windows flagged (closer to 5% = better calibrated)
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA

from algorithm import fit_reference_model, make_windows, score_window, variance_ratio_pvalue
from diagnose_mice import residual_diagnostics

CONFIG = "2000hz_pre60s_ntrain1000"
N_TRAIN = 1000
WINDOW_SIZE = 100
MAX_P = 3
MAX_Q = 3
ALPHA = 0.05


def fit_reference_model_whiteness(train_values: np.ndarray, max_p: int, max_q: int, alpha: float) -> dict:
    """Same (p,q) grid as fit_reference_model, but select by: lowest-AIC
    among candidates whose in-sample residuals pass Ljung-Box, falling
    back to the candidate with the highest Ljung-Box p-value if none pass.
    """
    candidates = []
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p == 0 and q == 0:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    warnings.simplefilter("ignore", UserWarning)
                    fit = ARIMA(train_values, order=(p, 0, q)).fit()
            except Exception:
                continue
            if not np.isfinite(fit.aic):
                continue
            resid = np.asarray(fit.standardized_forecasts_error, dtype=float)
            resid = resid[~np.isnan(resid)]
            if len(resid) < 15:
                continue
            lb_p = float(acorr_ljungbox(resid, lags=[10], return_df=True)["lb_pvalue"].iloc[0])

            param_names = list(fit.param_names)
            params = np.asarray(fit.params, dtype=float)
            sigma_index = param_names.index("sigma2")
            fixed_params = np.delete(params, sigma_index)
            candidates.append(
                {
                    "order": (p, 0, q),
                    "param_names": [name for name in param_names if name != "sigma2"],
                    "fixed_params": fixed_params,
                    "reference_sigma2": float(params[sigma_index]),
                    "reference_df": len(train_values) - len(fixed_params),
                    "aic": float(fit.aic),
                    "lb_p": lb_p,
                }
            )

    if not candidates:
        raise RuntimeError("No ARMA(p, q) model could be fit on the training segment.")

    passing = [c for c in candidates if c["lb_p"] > alpha]
    best = min(passing, key=lambda c: c["aic"]) if passing else max(candidates, key=lambda c: c["lb_p"])
    best["in_sample_passed"] = len(passing) > 0
    return best


def evaluate_model(model: dict, test_values: np.ndarray, labels: np.ndarray, n_train: int, window_size: int, alpha: float) -> dict:
    windows = make_windows(test_values, window_size)
    p_values = []
    actual_anomaly = []
    for i, w in enumerate(windows):
        sigma2_w, _ = score_window(w, model)
        _, p_value = variance_ratio_pvalue(sigma2_w, window_size, model)
        p_values.append(p_value)
        start, end = n_train + i * window_size, n_train + (i + 1) * window_size
        actual_anomaly.append(int(labels[start:end].any()))
    p_values = np.array(p_values)
    actual_anomaly = np.array(actual_anomaly)
    flagged = p_values < alpha

    normal_idx = np.where(actual_anomaly == 0)[0]
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(normal_idx, size=min(40, len(normal_idx)), replace=False)
    pooled_resid = np.concatenate([score_window(windows[i], model)[1] for i in sample_idx])
    oos_diag = residual_diagnostics(pooled_resid, "out-of-sample")

    n_true = int(actual_anomaly.sum())
    n_normal = int((~actual_anomaly.astype(bool)).sum())
    n_detected = int((flagged & actual_anomaly.astype(bool)).sum())
    n_false_alarms = int((flagged & ~actual_anomaly.astype(bool)).sum())

    return {
        "frac_flagged_overall": flagged.mean(),
        "detected": f"{n_detected}/{n_true}",
        "false_alarms": f"{n_false_alarms}/{n_normal}",
        "oos_ljung_box_p": oos_diag["ljung_box_p"],
        "oos_white_ok": oos_diag["white_noise_ok"],
    }


def main() -> None:
    for seizure_number in [1, 2, 3]:
        values = pd.read_csv(f"data/mice/{CONFIG}/timeseries_seizure{seizure_number}.csv")["value"].to_numpy(float)
        labels = pd.read_csv(f"data/mice/{CONFIG}/labels_seizure{seizure_number}.csv")["is_anomaly"].to_numpy(int)
        train_values = values[:N_TRAIN]
        test_values = values[N_TRAIN:]

        baseline = fit_reference_model(train_values, MAX_P, MAX_Q)
        baseline_resid = np.asarray(score_window(train_values, baseline)[1])
        baseline_lb_p = float(acorr_ljungbox(baseline_resid, lags=[10], return_df=True)["lb_pvalue"].iloc[0])

        candidate = fit_reference_model_whiteness(train_values, MAX_P, MAX_Q, ALPHA)

        print(f"=== Seizure {seizure_number} ===")
        print(f"Baseline (AIC-only):      order={baseline['order']}, AIC={baseline['aic']:.1f}, in-sample LB p={baseline_lb_p:.4f}")
        print(f"Whiteness-constrained:    order={candidate['order']}, AIC={candidate['aic']:.1f}, in-sample LB p={candidate['lb_p']:.4f}, any order passed={candidate['in_sample_passed']}")

        result_baseline = evaluate_model(baseline, test_values, labels, N_TRAIN, WINDOW_SIZE, ALPHA)
        result_candidate = evaluate_model(candidate, test_values, labels, N_TRAIN, WINDOW_SIZE, ALPHA)

        print(f"{'':25s} {'baseline':>18s} {'whiteness-constrained':>22s}")
        print(f"{'out-of-sample LB p':25s} {result_baseline['oos_ljung_box_p']:>18.4f} {result_candidate['oos_ljung_box_p']:>22.4f}")
        print(f"{'out-of-sample white?':25s} {str(result_baseline['oos_white_ok']):>18s} {str(result_candidate['oos_white_ok']):>22s}")
        print(f"{'frac windows flagged':25s} {result_baseline['frac_flagged_overall']:>18.3f} {result_candidate['frac_flagged_overall']:>22.3f}")
        print(f"{'detected':25s} {result_baseline['detected']:>18s} {result_candidate['detected']:>22s}")
        print(f"{'false alarms':25s} {result_baseline['false_alarms']:>18s} {result_candidate['false_alarms']:>22s}")
        print()


if __name__ == "__main__":
    main()
