"""
    1. Fit an ARMA(p, q) model on the training segment (AIC order search),
       giving the reference parameters c_hat, phi_hat, theta_hat and the
       reference innovation variance sigma0^2 (fit_reference_model).
    2. Cut the test segment into non-overlapping windows of length n
       (make_windows).
    3. For each window, hold c_hat/phi_hat/theta_hat FIXED and re-estimate
       only that window's own innovation variance sigma_w^2 via the Kalman
       filter (score_window).
    4. Form the ratio R_w = sigma_w^2 / sigma0^2 and its two-sided p-value
       from the F(n, n0) distribution (variance_ratio_pvalue).

The differencing order is fixed at d = 0: assumption is
that the normal process is stationary and invertible, so no differencing
is needed.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA

D_ORDER = 0


def fit_reference_model(train_values: np.ndarray, max_p: int, max_q: int) -> dict:
    """AIC search over ARMA(p, q), p,q in [0, max_p]x[0, max_q].

    Args:
        train_values: anomaly-free training segment, 1D array of floats.
        max_p: largest AR order to try (search covers p = 0..max_p).
        max_q: largest MA order to try (search covers q = 0..max_q).

    Returns:
        dict with:
            order: winning (p, d, q).
            param_names: names of the frozen parameters (const/ar/ma).
            fixed_params: their fitted values, held fixed per window.
            reference_sigma2: sigma0^2, the reference innovation variance.
            reference_df: n0 minus the number of fixed params (const/ar/ma),
                the F-test denominator df.
            aic: AIC of the winning fit.
    """
    best = None
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p == 0 and q == 0:
                continue  # skip the trivial "no dynamics" model
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    warnings.simplefilter("ignore", UserWarning)
                    fit = ARIMA(train_values, order=(p, D_ORDER, q)).fit()
            except Exception:
                continue
            if not np.isfinite(fit.aic):
                continue
            if best is None or fit.aic < best["aic"]:
                param_names = list(fit.param_names)
                params = np.asarray(fit.params, dtype=float)
                sigma_index = param_names.index("sigma2")
                fixed_params = np.delete(params, sigma_index)
                best = {
                    "order": (p, D_ORDER, q),
                    "param_names": [name for name in param_names if name != "sigma2"],
                    "fixed_params": fixed_params,
                    "reference_sigma2": float(params[sigma_index]),
                    # n0 - (number of fixed params): sigma0^2 is estimated on
                    # the same data that fit c_hat/phi_hat/theta_hat, so those
                    # degrees of freedom are already "spent" and shouldn't be
                    # double-counted in the F-test denominator.
                    "reference_df": len(train_values) - len(fixed_params),
                    "aic": float(fit.aic),
                }

    if best is None:
        raise RuntimeError("No ARMA(p, q) model could be fit on the training segment.")
    return best


def make_windows(test_values: np.ndarray, window_size: int) -> list[np.ndarray]:
    """Cut the test segment into non-overlapping windows.

    Args:
        test_values: test segment, 1D array of floats.
        window_size: number of points per window (n).

    Returns:
        list of 1D arrays, each of length window_size. A trailing partial
        window (fewer than window_size points left over) is dropped --
        every window must hold exactly n points for F(n, n0) to be the
        right reference distribution.
    """
    n_windows = len(test_values) // window_size
    return [test_values[i * window_size : (i + 1) * window_size] for i in range(n_windows)]


def score_window(window_values: np.ndarray, model: dict) -> tuple[float, np.ndarray]:
    """Hold model's c_hat/phi_hat/theta_hat fixed and re-estimate only this
    window's own innovation variance, via statsmodels' concentrate_scale
    trick (the analytic MLE for sigma^2 given fixed dynamics).

    Args:
        window_values: one window's values, 1D array of length window_size.
        model: the fitted reference model, as returned by fit_reference_model.

    Returns:
        (sigma2_w, standardized_residuals):
            sigma2_w: this window's own innovation variance estimate.
            standardized_residuals: v_t / sqrt(sigma2_w * F_t^*) for each
                point in the window (1D array, length window_size) -- a
                useful byproduct for checking assumption A2 later.
    """
    window_model = ARIMA(window_values, order=model["order"], concentrate_scale=True)
    if list(window_model.param_names) != model["param_names"]:
        raise RuntimeError("Parameter mismatch between reference model and window model.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = window_model.filter(model["fixed_params"])

    sigma2_w = float(result.scale)
    standardized_residuals = np.asarray(result.filter_results.standardized_forecasts_error[0], dtype=float)
    return sigma2_w, standardized_residuals


def variance_ratio_pvalue(sigma2_w: float, window_size: int, model: dict) -> tuple[float, float]:
    """R_w = sigma_w^2 / sigma0^2 and its two-sided p-value from F(n, n0).

    Args:
        sigma2_w: this window's innovation variance estimate (from score_window).
        window_size: number of points in the window (n).
        model: the fitted reference model, as returned by fit_reference_model.

    Returns:
        (ratio, p_value):
            ratio: R_w = sigma2_w / sigma0^2.
            p_value: two-sided p-value from F(window_size, model["reference_df"]).
    """
    ratio = sigma2_w / model["reference_sigma2"]
    lower_tail = stats.f.cdf(ratio, window_size, model["reference_df"])
    upper_tail = stats.f.sf(ratio, window_size, model["reference_df"])
    p_value = min(1.0, 2.0 * min(lower_tail, upper_tail))
    return ratio, p_value


def run_algorithm(
    train_values: np.ndarray,
    test_values: np.ndarray,
    window_size: int,
    alpha: float,
    max_p: int = 3,
    max_q: int = 3,
) -> tuple[pd.DataFrame, dict, list[np.ndarray]]:
    """Run Algorithm 1 end to end: fit the reference model, then score
    every test window.

    Args:
        train_values: anomaly-free training segment, 1D array of floats.
        test_values: test segment, 1D array of floats.
        window_size: number of points per window (n).
        alpha: significance level; a window is flagged when p_value < alpha.
        max_p: largest AR order to try when fitting the reference model.
        max_q: largest MA order to try when fitting the reference model.

    Returns:
        (results, model, residuals_by_window):
            results: one row per window -- columns window_index,
                window_start, window_end, sigma2_w, variance_ratio,
                p_value, is_anomalous.
            model: the fitted reference model (see fit_reference_model).
            residuals_by_window: each window's standardized residuals
                (see score_window), aligned with the rows of `results`.
    """
    train_values = np.asarray(train_values, dtype=float)
    test_values = np.asarray(test_values, dtype=float)

    model = fit_reference_model(train_values, max_p, max_q)
    windows = make_windows(test_values, window_size)

    rows = []
    residuals_by_window = []
    for i, window in enumerate(windows):
        sigma2_w, residuals = score_window(window, model)
        ratio, p_value = variance_ratio_pvalue(sigma2_w, window_size, model)
        residuals_by_window.append(residuals)
        rows.append(
            {
                "window_index": i,
                "window_start": i * window_size,
                "window_end": i * window_size + window_size,
                "sigma2_w": sigma2_w,
                "variance_ratio": ratio,
                "p_value": p_value,
                "is_anomalous": int(p_value < alpha),
            }
        )

    return pd.DataFrame(rows), model, residuals_by_window
