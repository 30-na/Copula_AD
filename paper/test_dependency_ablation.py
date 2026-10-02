"""Controlled null ablation: isolate how much of the inflated false-alarm
rate comes from dependency alone vs. dependency + non-Gaussianity, by
generating synthetic data under H0 (no real anomaly, fixed variance
throughout) from the REAL fitted reference model (seizure 1, channel 1),
varying ONLY how the innovations are generated:

  A. iid Gaussian innovations             -- ideal assumptions (positive control)
  B. autocorrelated Gaussian innovations   -- effect of dependence alone
  C. autocorrelated, REAL-shaped marginal  -- dependence + the actual
                                              non-Gaussian shape we measured

The autocorrelation structure for B/C is not assumed -- it's an AR(k)
model fit directly to a long, contiguous stretch of the REAL out-of-sample
residuals (model order picked by AIC). For C, the AR(k) is driven by
bootstrap-resampled residuals FROM THAT SAME FIT (preserving the true
marginal shape we actually measured, including the mild non-Gaussianity
found earlier), rather than an assumed heavy-tailed distribution.

Everything else is held fixed: same reference dynamics (phi_hat, theta_hat,
c_hat), same reference sigma0^2, same window_size, same alpha, same
scoring code (algorithm.score_window / variance_ratio_pvalue) -- so any
difference in empirical Type I error between A, B, C is attributable
ONLY to how the innovations were generated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg

from algorithm import fit_reference_model, make_windows, score_window, variance_ratio_pvalue

CONFIG = "2000hz_pre60s_ntrain1000"
N_TRAIN = 1000
WINDOW_SIZE = 100
SEIZURE_NUMBER = 1
ALPHA = 0.05
SYNTHETIC_LENGTH = 500_000
BURN_IN = 2000


def fit_dependency_structure(residuals: np.ndarray, max_lag: int = 10) -> AutoReg:
    """AR(k) fit to the real residuals, order picked by AIC."""
    best = None
    for k in range(1, max_lag + 1):
        fit = AutoReg(residuals, lags=k, old_names=False).fit()
        if best is None or fit.aic < best.aic:
            best = fit
    return best


def simulate_arma_from_innovations(zeta: np.ndarray, order: tuple, fixed_params: np.ndarray, param_names: list[str]) -> np.ndarray:
    """Manual ARMA(p,0,q) recursion driven by a given innovation sequence
    zeta_t (already at the correct scale), with fixed coefficients from
    the real reference model.
    """
    p, _, q = order
    const = 0.0
    ar_coefs = np.zeros(p)
    ma_coefs = np.zeros(q)
    for name, val in zip(param_names, fixed_params):
        if name == "const":
            const = val
        elif name.startswith("ar.L"):
            ar_coefs[int(name.split("L")[1]) - 1] = val
        elif name.startswith("ma.L"):
            ma_coefs[int(name.split("L")[1]) - 1] = val

    n = len(zeta)
    y = np.zeros(n)
    for t in range(n):
        val = const
        for i in range(p):
            if t - i - 1 >= 0:
                val += ar_coefs[i] * (y[t - i - 1] - const)
        val += zeta[t]
        for j in range(q):
            if t - j - 1 >= 0:
                val += ma_coefs[j] * zeta[t - j - 1]
        y[t] = val
    return y


def empirical_type1_error(y: np.ndarray, model: dict, window_size: int, alpha: float, burn_in: int) -> tuple[float, int]:
    y = y[burn_in:]
    windows = make_windows(y, window_size)
    n_flagged = 0
    for w in windows:
        sigma2_w, _ = score_window(w, model)
        _, p_value = variance_ratio_pvalue(sigma2_w, window_size, model)
        if p_value < alpha:
            n_flagged += 1
    return n_flagged / len(windows), len(windows)


def main() -> None:
    values = pd.read_csv(f"data/mice/{CONFIG}/timeseries_seizure{SEIZURE_NUMBER}.csv")["value"].to_numpy(float)
    labels = pd.read_csv(f"data/mice/{CONFIG}/labels_seizure{SEIZURE_NUMBER}.csv")["is_anomaly"].to_numpy(int)

    train_values = values[:N_TRAIN]
    test_values = values[N_TRAIN:]
    model = fit_reference_model(train_values, 3, 3)
    sigma0 = np.sqrt(model["reference_sigma2"])
    print(f"Reference model: order={model['order']}, sigma0={sigma0:.4f}")

    # a long, contiguous, label-confirmed-normal stretch to characterize
    # the REAL residual dependency structure (first 100000 of 129000 test
    # samples; seizure occupies only the last ~10000, confirmed via labels)
    normal_stretch = test_values[:100_000]
    assert labels[N_TRAIN : N_TRAIN + 100_000].sum() == 0, "normal_stretch unexpectedly overlaps the seizure"
    _, real_residuals = score_window(normal_stretch, model)

    dep_fit = fit_dependency_structure(real_residuals, max_lag=10)
    psi = dep_fit.params[1:]  # drop the AR intercept
    k = len(psi)
    print(f"Dependency structure: AR({k}) fit to real out-of-sample residuals, AIC={dep_fit.aic:.1f}")
    print(f"  AR coefficients: {np.round(psi, 4).tolist()}")
    driving_residuals = dep_fit.resid  # the AR(k) model's own innovations (true empirical marginal)
    tau = driving_residuals.std()
    print(f"  Driving noise std (tau): {tau:.4f}, skew={pd.Series(driving_residuals).skew():.3f}, "
          f"excess kurtosis={pd.Series(driving_residuals).kurt():.3f}")
    print()

    rng = np.random.default_rng(0)
    n = SYNTHETIC_LENGTH + BURN_IN

    def generate_ar_k(driving_noise: np.ndarray) -> np.ndarray:
        out = np.zeros(len(driving_noise))
        for t in range(len(driving_noise)):
            val = driving_noise[t]
            for i in range(k):
                if t - i - 1 >= 0:
                    val += psi[i] * out[t - i - 1]
            out[t] = val
        return out

    # A: iid Gaussian innovations, scale = sigma0 (ideal assumptions)
    zeta_a = rng.normal(0.0, sigma0, size=n)

    # B: autocorrelated Gaussian innovations (dependence only)
    driving_b = rng.normal(0.0, tau, size=n)
    resid_b = generate_ar_k(driving_b)
    zeta_b = resid_b * sigma0  # real residuals are ~unit-scale standardized; rescale to innovation units

    # C: autocorrelated, REAL-shaped marginal (dependence + true non-Gaussianity)
    driving_c = rng.choice(driving_residuals, size=n, replace=True)  # bootstrap from the AR(k)'s own innovations
    resid_c = generate_ar_k(driving_c)
    zeta_c = resid_c * sigma0

    results = {}
    for label, zeta in [("A: iid Gaussian", zeta_a), ("B: autocorrelated Gaussian", zeta_b), ("C: autocorrelated, real marginal", zeta_c)]:
        y = simulate_arma_from_innovations(zeta, model["order"], model["fixed_params"], model["param_names"])
        type1, n_windows = empirical_type1_error(y, model, WINDOW_SIZE, ALPHA, BURN_IN)
        results[label] = type1
        print(f"{label:38s}: Type I error = {type1:.2%}  (n_windows={n_windows})")

    print()
    print(f"A -> B (contribution of autocorrelation alone):      {results['B: autocorrelated Gaussian'] - results['A: iid Gaussian']:+.2%}")
    print(f"B -> C (additional contribution of non-Gaussianity): {results['C: autocorrelated, real marginal'] - results['B: autocorrelated Gaussian']:+.2%}")


if __name__ == "__main__":
    main()
