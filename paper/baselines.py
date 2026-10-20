"""Non-deep-learning baseline detectors, each scored/thresholded using
ONLY the training segment (no test-label peeking), matching the same
fairness rule as our own bootstrap-calibrated method.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from algorithm import make_windows, score_window

ALPHA = 0.05


def per_point_squared_residuals(values: np.ndarray, model: dict) -> np.ndarray:
    """z_t^2 = v_t^2 / F_t for every individual sample in `values`, using
    the FIXED reference model (c_hat/phi_hat/theta_hat AND sigma0^2, none
    re-estimated). Under H0 (no variance change), z_t^2 ~ chi2(1), mean 1 --
    this is the standard point-by-point residual stream used by classic
    residual control charts (Alwan & Roberts 1988; Montgomery & Mastrangelo
    1991), as opposed to our own window-batched R_w.
    """
    probe = ARIMA(values, order=model["order"], concentrate_scale=False)
    sigma_idx = list(probe.param_names).index("sigma2")
    full_params = np.insert(model["fixed_params"], sigma_idx, model["reference_sigma2"])
    result = probe.filter(full_params)
    v = np.asarray(result.filter_results.forecasts_error[0], dtype=float)
    F = np.asarray(result.filter_results.forecasts_error_cov[0, 0], dtype=float)
    return (v**2) / F


def naive_rolling_std(
    train_values: np.ndarray, test_values: np.ndarray, window_size: int
) -> tuple[np.ndarray, float]:
    """Raw sliding (overlapping) rolling std, no ARMA pre-whitening.
    Threshold = (1-alpha) percentile of the training rolling std.
    """
    train_std = pd.Series(train_values).rolling(window=window_size).std().dropna().to_numpy()
    threshold = float(np.percentile(train_std, 100 * (1 - ALPHA)))
    test_std = pd.Series(test_values).rolling(window=window_size, center=True).std().to_numpy()
    return test_std, threshold


def window_ratios(values: np.ndarray, model: dict, window_size: int) -> np.ndarray:
    """R_w = sigma_w^2 / sigma0^2 for each non-overlapping window of `values`."""
    windows = make_windows(values, window_size)
    ratios = np.empty(len(windows))
    for i, w in enumerate(windows):
        sigma2_w, _ = score_window(w, model)
        ratios[i] = sigma2_w / model["reference_sigma2"]
    return ratios


def cusum_on_ratios(
    train_values: np.ndarray, test_ratios: np.ndarray, model: dict, window_size: int,
    k: float = 0.5, h: float = 5.0,
) -> tuple[np.ndarray, float]:
    """One-sided upper CUSUM on standardized window-level variance ratios.

    Standardizes (R_w - 1) by the training windows' own spread, then
    accumulates deviations past a slack k; flags where the running sum
    exceeds h. k=0.5, h=5 are the standard Montgomery (SQC) textbook
    defaults for a one-sided CUSUM (ARL0 ~ 465 under Gaussian shifts),
    not tuned to this dataset.
    """
    train_ratios = window_ratios(train_values, model, window_size)
    train_dev = train_ratios - 1.0
    sigma_train = float(np.std(train_dev, ddof=1)) if len(train_dev) > 1 else 1.0
    sigma_train = sigma_train if sigma_train > 0 else 1.0

    z = (test_ratios - 1.0) / sigma_train
    cusum = np.empty(len(z))
    running = 0.0
    for i, zi in enumerate(z):
        running = max(0.0, running + zi - k)
        cusum[i] = running
    return cusum, h


def cusum_menelaus_on_ratios(
    train_values: np.ndarray, test_ratios: np.ndarray, model: dict, window_size: int,
) -> tuple[np.ndarray, float]:
    """One-sided CUSUM via the `menelaus` package's CUSUM detector (Page
    1954), which resets its running sum to 0 after each alarm -- unlike
    our hand-rolled version above. Target/sd_hat are set from the
    training windows' own ratios (no burn-in needed, no test-label use).
    """
    from menelaus.change_detection import CUSUM as MenelausCUSUM

    train_ratios = window_ratios(train_values, model, window_size)
    sd_hat = float(np.std(train_ratios, ddof=1)) if len(train_ratios) > 1 else 1.0
    sd_hat = sd_hat if sd_hat > 0 else 1.0

    detector = MenelausCUSUM(target=1.0, sd_hat=sd_hat, burn_in=5, delta=0.005, threshold=5, direction="positive")
    curve = np.empty(len(test_ratios))
    for i, r in enumerate(test_ratios):
        detector.update(np.array([r]))
        curve[i] = detector._upper_bound[-1]
    return curve, 5.0
