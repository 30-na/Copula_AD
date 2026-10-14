"""Reusable engine: given one train/test split, evaluate THREE calibration
methods on the exact same fitted reference model and test windows:
  1. F-distribution theory (the original pipeline)
  2. Block bootstrap (resample real normal-labeled chunks directly)
  3. Bayesian posterior odds (Inverse-Gamma prior fit to the SAME
     bootstrap sample, Bayes factor threshold calibrated to ~alpha)

Generalized from the mice-specific scripts: the "normal pool" for
bootstrapping is now ANY valid window-length start position in the test
region that doesn't overlap a labeled anomaly (works for both the mice
data, where that's a single long contiguous stretch, and UCR series,
where normal regions may be shorter/fragmented).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from algorithm import fit_reference_model, make_windows, score_window, variance_ratio_pvalue


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


def valid_normal_starts(test_values: np.ndarray, test_is_anomaly: np.ndarray, window_size: int) -> np.ndarray:
    """Every start index i such that test_is_anomaly[i:i+window_size] is
    all-normal (no overlap with a labeled anomaly), i.e. a legitimate
    chunk to bootstrap from.
    """
    n = len(test_values) - window_size + 1
    cum = np.concatenate([[0], np.cumsum(test_is_anomaly)])
    window_sums = cum[window_size:window_size + n] - cum[:n]
    return np.where(window_sums == 0)[0]


def build_block_bootstrap_null(
    test_values: np.ndarray, starts: np.ndarray, model: dict, window_size: int, n_bootstrap: int, rng: np.random.Generator
) -> np.ndarray:
    chosen = rng.choice(starts, size=n_bootstrap, replace=True)
    t_star = np.empty(n_bootstrap)
    for b, s in enumerate(chosen):
        sigma2_w, _ = score_window(test_values[s : s + window_size], model)
        t_star[b] = sigma2_w / model["reference_sigma2"]
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

    starts = valid_normal_starts(test_values, test_is_anomaly, window_size)
    if len(starts) < 20:
        return None

    rng = np.random.default_rng(seed)
    t_star = build_block_bootstrap_null(test_values, starts, model, window_size, n_bootstrap, rng)

    # method 1: F-distribution theory
    flagged_f = results["p_value"].to_numpy() < alpha
    fa_f, det_f = int((flagged_f & normal_mask).sum()), int((flagged_f & anomaly_mask).sum())

    # method 2: block bootstrap percentiles
    lower_bb, upper_bb = np.percentile(t_star, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    ratio = results["variance_ratio"].to_numpy()
    flagged_bb = (ratio < lower_bb) | (ratio > upper_bb)
    fa_bb, det_bb = int((flagged_bb & normal_mask).sum()), int((flagged_bb & anomaly_mask).sum())

    # method 3: Bayesian posterior odds, threshold calibrated to ~alpha on this dataset's own normal windows
    a, b = fit_inverse_gamma_prior(t_star, model["reference_sigma2"], window_size)
    sse = results["sigma2_w"].to_numpy() * window_size
    bf = bayes_factor(sse, model["reference_sigma2"], window_size, a, b)
    normal_bf_sorted = np.sort(bf[normal_mask])[::-1]
    target_idx = min(int(alpha * n_normal), len(normal_bf_sorted) - 1)
    bf_thresh = normal_bf_sorted[target_idx]
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
