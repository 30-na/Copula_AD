"""Literally test "keep increasing model order until residuals stop being
autocorrelated" -- not capped at (3,3) like the pipeline's default search.
Pure AR(p) for p = 1..20 (q=0, to keep each fit fast and the trend easy to
read), on seizure 3 first (the one where no order <=3 passed in-sample),
then all 3 if it's informative. Tracks BOTH in-sample (trivially easier)
and out-of-sample (pooled from 40 sampled normal test windows -- the
diagnostic that actually matters) Ljung-Box p-values at every order.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA

from algorithm import make_windows, score_window
from diagnose_mice import residual_diagnostics

CONFIG = "2000hz_pre60s_ntrain1000"
N_TRAIN = 1000
WINDOW_SIZE = 100
MAX_ORDER = 20


def fit_fixed_order(train_values: np.ndarray, p: int) -> dict | None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", UserWarning)
            fit = ARIMA(train_values, order=(p, 0, 0)).fit()
    except Exception:
        return None
    if not np.isfinite(fit.aic):
        return None
    param_names = list(fit.param_names)
    params = np.asarray(fit.params, dtype=float)
    sigma_index = param_names.index("sigma2")
    fixed_params = np.delete(params, sigma_index)
    resid = np.asarray(fit.standardized_forecasts_error, dtype=float)
    resid = resid[~np.isnan(resid)]
    in_sample_lb_p = float(acorr_ljungbox(resid, lags=[10], return_df=True)["lb_pvalue"].iloc[0])
    return {
        "order": (p, 0, 0),
        "param_names": [name for name in param_names if name != "sigma2"],
        "fixed_params": fixed_params,
        "reference_sigma2": float(params[sigma_index]),
        "reference_df": len(train_values) - len(fixed_params),
        "aic": float(fit.aic),
        "in_sample_lb_p": in_sample_lb_p,
    }


def out_of_sample_lb_p(model: dict, test_values: np.ndarray, window_size: int, seed: int = 0) -> float:
    windows = make_windows(test_values, window_size)
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(len(windows), size=min(40, len(windows)), replace=False)
    pooled = np.concatenate([score_window(windows[i], model)[1] for i in sample_idx])
    return residual_diagnostics(pooled, "oos")["ljung_box_p"]


def main() -> None:
    for seizure_number in [3, 1, 2]:
        values = pd.read_csv(f"data/mice/{CONFIG}/timeseries_seizure{seizure_number}.csv")["value"].to_numpy(float)
        train_values = values[:N_TRAIN]
        test_values = values[N_TRAIN:]

        print(f"=== Seizure {seizure_number}: AR(p), q=0, p=1..{MAX_ORDER} ===")
        print(f"{'p':>3s} {'AIC':>10s} {'in-sample LB p':>16s} {'out-of-sample LB p':>20s}")
        first_in_sample_pass = None
        first_oos_pass = None
        for p in range(1, MAX_ORDER + 1):
            model = fit_fixed_order(train_values, p)
            if model is None:
                print(f"{p:>3d}  (fit failed)")
                continue
            oos_p = out_of_sample_lb_p(model, test_values, WINDOW_SIZE)
            print(f"{p:>3d} {model['aic']:>10.1f} {model['in_sample_lb_p']:>16.4f} {oos_p:>20.4f}")
            if first_in_sample_pass is None and model["in_sample_lb_p"] > 0.05:
                first_in_sample_pass = p
            if first_oos_pass is None and oos_p > 0.05:
                first_oos_pass = p
        print(f"-> first order with in-sample whiteness: {first_in_sample_pass}")
        print(f"-> first order with OUT-OF-SAMPLE whiteness: {first_oos_pass}")
        print()


if __name__ == "__main__":
    main()
