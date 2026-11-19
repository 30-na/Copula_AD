"""IVR (innovation variance ratio): the method, in one shared file.

Sections:
  1. Algorithm: ARMA reference model, window innovation variance, F test
  2. Thresholds: block bootstrap and Bayesian posterior odds
  3. Fast scoring of every overlapping window (closed form of score_window)
  4. Data helper: end of the UCR training part
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import math

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats
from scipy.linalg import cholesky, solve_triangular, toeplitz
from scipy.optimize import minimize
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import arma_acovf

# =============================================================================
# 1. Algorithm
# =============================================================================
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


# =============================================================================
# 2. Thresholds: block bootstrap and Bayesian posterior odds
# =============================================================================
"""Reusable engine: given one train/test split, evaluate THREE calibration
methods on the exact same fitted reference model and test windows:
  1. F-distribution theory (the original pipeline)
  2. Block bootstrap (resample real normal chunks directly)
  3. Bayesian posterior odds (Inverse-Gamma prior fit to the SAME
     bootstrap sample, Bayes factor threshold calibrated to ~alpha)

No test labels are used for calibration: the bootstrap pool is every
window-length start position inside the anomaly-free training segment,
the same data the reference model was fit on (calibration_starts). Test
labels are used only to score the result.
"""

def score_all_windows(test_values: np.ndarray, model: dict, window_size: int, alpha: float) -> pd.DataFrame:
    """Same scoring loop as algorithm.run_algorithm, but takes an
    ALREADY-FIT model instead of fitting one -- avoids paying for the
    (possibly expensive, e.g. on 30k+-sample UCR training sets) ARMA
    order search twice per dataset.
    """
    windows = make_windows(test_values, window_size)
    rows = []
    for i, window in enumerate(windows):
        sigma2_w, _ = score_window(window, model)
        ratio, p_value = variance_ratio_pvalue(sigma2_w, window_size, model)
        rows.append({
            "window_index": i, "window_start": i * window_size, "window_end": i * window_size + window_size,
            "sigma2_w": sigma2_w, "variance_ratio": ratio, "p_value": p_value, "is_anomalous": int(p_value < alpha),
        })
    return pd.DataFrame(rows)


def f_critical_values(window_size: int, reference_df: int, alpha: float) -> tuple[float, float]:
    return (
        float(stats.f.ppf(alpha / 2, window_size, reference_df)),
        float(stats.f.ppf(1 - alpha / 2, window_size, reference_df)),
    )


def calibration_starts(train_values: np.ndarray, window_size: int) -> np.ndarray:
    """Every start index of a full window inside the training segment.
    In-sample (same data the reference model was fit on), so the null is
    slightly optimistic, but UCR training segments are too short to hold
    out a separate calibration part.
    """
    return np.arange(max(0, len(train_values) - window_size + 1))


def build_block_bootstrap_null(
    train_values: np.ndarray, starts: np.ndarray, model: dict, window_size: int, n_bootstrap: int, rng: np.random.Generator
) -> np.ndarray:
    chosen = rng.choice(starts, size=n_bootstrap, replace=True)
    cache: dict[int, float] = {}  # same start drawn twice -> same R_w; score it once
    t_star = np.empty(n_bootstrap)
    for b, s in enumerate(chosen):
        if s not in cache:
            sigma2_w, _ = score_window(train_values[s : s + window_size], model)
            cache[s] = sigma2_w / model["reference_sigma2"]
        t_star[b] = cache[s]
    return t_star


def fit_inverse_gamma_prior(t_star: np.ndarray, sigma0_sq: float, window_size: int) -> tuple[float, float]:
    sse_samples = t_star * sigma0_sq * window_size

    def neg_log_lik(params):
        a, b = params
        if a <= 0 or b <= 0:
            return np.inf
        scale = b * window_size / a
        ll = stats.f.logpdf(sse_samples / scale, window_size, 2 * a) - np.log(scale)
        return -np.sum(ll)

    result = minimize(neg_log_lik, x0=[5.0, 5.0 * sigma0_sq], method="Nelder-Mead")
    a, b = result.x
    return float(a), float(b)


def bayes_factor(sse_w: np.ndarray, sigma0_sq: float, window_size: int, a: float, b: float) -> np.ndarray:
    p_h0 = stats.chi2.pdf(sse_w / sigma0_sq, window_size) / sigma0_sq
    scale = b * window_size / a
    p_h1 = stats.f.pdf(sse_w / scale, window_size, 2 * a) / scale
    with np.errstate(divide="ignore", invalid="ignore"):
        bf = np.where(p_h0 > 0, p_h1 / p_h0, np.inf)
    return bf


def evaluate_three_methods(
    train_values: np.ndarray, test_values: np.ndarray, test_is_anomaly: np.ndarray,
    window_size: int, alpha: float, max_p: int, max_q: int, n_bootstrap: int, seed: int,
) -> dict | None:
    """Returns a dict of results for all three methods, or None if this
    dataset can't support a meaningful comparison (e.g. too few normal
    windows to bootstrap from).
    """
    model = fit_reference_model(train_values, max_p, max_q)
    results = score_all_windows(test_values, model, window_size, alpha)

    actual_anomaly = []
    for _, row in results.iterrows():
        start, end = int(row["window_start"]), int(row["window_end"])
        actual_anomaly.append(int(test_is_anomaly[start:end].any()))
    results["actual_anomaly"] = np.array(actual_anomaly)

    normal_mask = results["actual_anomaly"].to_numpy() == 0
    anomaly_mask = results["actual_anomaly"].to_numpy() == 1
    n_normal, n_anomaly = int(normal_mask.sum()), int(anomaly_mask.sum())
    if n_normal < 20 or n_anomaly == 0:
        return None

    starts = calibration_starts(train_values, window_size)
    if len(starts) < 20:
        return None

    rng = np.random.default_rng(seed)
    t_star = build_block_bootstrap_null(train_values, starts, model, window_size, n_bootstrap, rng)

    # method 1: F-distribution theory
    flagged_f = results["p_value"].to_numpy() < alpha
    fa_f, det_f = int((flagged_f & normal_mask).sum()), int((flagged_f & anomaly_mask).sum())

    # method 2: block bootstrap percentiles
    lower_bb, upper_bb = np.percentile(t_star, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    ratio = results["variance_ratio"].to_numpy()
    flagged_bb = (ratio < lower_bb) | (ratio > upper_bb)
    fa_bb, det_bb = int((flagged_bb & normal_mask).sum()), int((flagged_bb & anomaly_mask).sum())

    # method 3: Bayesian posterior odds, threshold = (1-alpha) quantile of the
    # Bayes factor over the calibration bootstrap sample (no test labels)
    a, b = fit_inverse_gamma_prior(t_star, model["reference_sigma2"], window_size)
    sse = results["sigma2_w"].to_numpy() * window_size
    bf = bayes_factor(sse, model["reference_sigma2"], window_size, a, b)
    bf_null = bayes_factor(t_star * model["reference_sigma2"] * window_size, model["reference_sigma2"], window_size, a, b)
    bf_thresh = float(np.percentile(bf_null, 100 * (1 - alpha)))
    flagged_bayes = bf > bf_thresh
    fa_bayes, det_bayes = int((flagged_bayes & normal_mask).sum()), int((flagged_bayes & anomaly_mask).sum())

    return {
        "order": model["order"], "reference_sigma2": model["reference_sigma2"],
        "n_normal": n_normal, "n_anomaly": n_anomaly,
        "f_false_alarms": fa_f, "f_detected": det_f,
        "f_false_alarm_rate": fa_f / n_normal, "f_detection_rate": det_f / n_anomaly,
        "bb_false_alarms": fa_bb, "bb_detected": det_bb,
        "bb_false_alarm_rate": fa_bb / n_normal, "bb_detection_rate": det_bb / n_anomaly,
        "bayes_false_alarms": fa_bayes, "bayes_detected": det_bayes,
        "bayes_false_alarm_rate": fa_bayes / n_normal, "bayes_detection_rate": det_bayes / n_anomaly,
        "bb_lower": lower_bb, "bb_upper": upper_bb, "bf_thresh": bf_thresh,
        "ig_a": a, "ig_b": b,
    }


# =============================================================================
# 3. Fast scoring of every overlapping window
# =============================================================================
# With the ARMA parameters fixed, sigma2_w = (x - mu)' S^-1 (x - mu) / n, where S is
# the ARMA autocovariance matrix of the window (sigma^2 = 1). Equals score_window()
# (Kalman filter, stationary start) to 1e-8 relative; ~1000x faster.

CHUNK = 20000


def window_ratios(x: np.ndarray, model: dict, w: int) -> np.ndarray:
    """R_w of every window of length w (closed form of score_window)."""
    g = dict(zip(model["param_names"], model["fixed_params"]))
    p, _, q = model["order"]
    ar = np.r_[1.0, -np.array([g[f"ar.L{i}"] for i in range(1, p + 1)])]
    ma = np.r_[1.0, np.array([g[f"ma.L{i}"] for i in range(1, q + 1)])]
    C = cholesky(toeplitz(arma_acovf(ar, ma, nobs=w, sigma2=1.0)), lower=True)
    Z = sliding_window_view(x - g.get("const", 0.0), w)
    out = np.empty(len(Z))
    for a in range(0, len(Z), CHUNK):
        out[a : a + CHUNK] = (solve_triangular(C, Z[a : a + CHUNK].T, lower=True) ** 2).mean(axis=0)
    return out / model["reference_sigma2"]


def to_points(ratio: np.ndarray, w: int) -> np.ndarray:
    return np.concatenate([[ratio[0]] * math.ceil((w - 1) / 2), ratio, [ratio[-1]] * ((w - 1) // 2)])


# =============================================================================
# 4. Data helper
# =============================================================================

UCR_RAW_DIR = Path("data/raw/ucr")


def ucr_train_end(name: str) -> int:
    """End of the anomaly-free training part of a UCR series, e.g. ucr_067_...:
    the first of the three numbers at the end of the raw UCR file name."""
    idx = re.match(r"ucr_(\d+)_", name).group(1)
    raw = next(UCR_RAW_DIR.glob(f"{idx}_UCR_Anomaly_*.txt"))
    return int(raw.stem.split("_")[-3])
