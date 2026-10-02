"""Step 1: simple Bayesian posterior-odds test for a variance change,
replacing the F-test p-value with a Bayes factor.

Setup, for one test window of size n_w with observed sigma2_w (already
computed exactly as the pipeline always does):
  H0: this window's true variance = sigma0^2 (fixed reference, known)
      -> SSE_w / sigma0^2 ~ chi2(n_w)
  H1: this window's true variance = sigma_w^2, UNKNOWN, with a conjugate
      Inverse-Gamma(a, b) prior
      -> marginalizing over the prior gives a closed form:
         SSE_w ~ (b * n_w / a) * F(n_w, 2a)   [derived from the standard
         Gamma/Gamma ratio -> F relationship; see comment in bayes_factor()]

The prior (a, b) is NOT arbitrary -- it's fit (via MLE) directly to the
SAME empirical block-bootstrap sample (t_star) already built in
test_block_bootstrap.py, so H1 reflects the REAL measured spread of
normal-window variances, not a generic/vague prior.

Decision rule: posterior odds = Bayes factor (prior odds = 1, i.e. no
prior preference for H0 vs H1) -- flag anomaly when BF > 1 (H1 more
probable than H0 given the data), no separate alpha needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from algorithm import fit_reference_model
from test_block_bootstrap import CONFIG, N_TRAIN, WINDOW_SIZE, SEIZURE_NUMBER


def fit_inverse_gamma_prior(t_star: np.ndarray, sigma0_sq: float, window_size: int) -> tuple[float, float]:
    """MLE fit of (a, b) so that SSE ~ (b*n_w/a) * F(n_w, 2a) matches the
    empirical distribution of SSE = t_star * sigma0_sq * window_size.
    """
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


def bayes_factor(sse_w: float, sigma0_sq: float, window_size: int, a: float, b: float) -> float:
    """BF = p(SSE_w | H1) / p(SSE_w | H0).

    p(SSE_w | H0): SSE_w/sigma0^2 ~ chi2(n_w), known variance.
    p(SSE_w | H1): SSE_w ~ (b*n_w/a) * F(n_w, 2a), derived as follows --
    let tau = 1/sigma_w^2 ~ Gamma(a, rate=b) (Inverse-Gamma(a,b) prior on
    sigma_w^2). Given tau, SSE_w*tau ~ chi2(n_w) =: Z. Let U = 2*b*tau ~
    chi2(2a). Then (Z/n_w)/(U/2a) ~ F(n_w, 2a) by the standard
    chi2-ratio-to-F relationship, and SSE_w = Z/tau = Z*2b/U, which
    rearranges to SSE_w = (b*n_w/a) * F(n_w, 2a).
    """
    p_h0 = stats.chi2.pdf(sse_w / sigma0_sq, window_size) / sigma0_sq
    scale = b * window_size / a
    p_h1 = stats.f.pdf(sse_w / scale, window_size, 2 * a) / scale
    return p_h1 / p_h0 if p_h0 > 0 else np.inf


def main() -> None:
    window_results = pd.read_csv(f"results/mice/{CONFIG}/window_results_seizure{SEIZURE_NUMBER}.csv")
    values = pd.read_csv(f"data/mice/{CONFIG}/timeseries_seizure{SEIZURE_NUMBER}.csv")["value"].to_numpy(float)
    train_values = values[:N_TRAIN]
    model = fit_reference_model(train_values, 3, 3)
    sigma0_sq = model["reference_sigma2"]

    t_star = np.load("results/mice/figs/block_bootstrap_t_star.npy")
    a, b = fit_inverse_gamma_prior(t_star, sigma0_sq, WINDOW_SIZE)
    print(f"Calibrated Inverse-Gamma(a={a:.3f}, b={b:.5f}) prior for H1, fit to {len(t_star)} bootstrap samples")
    print(f"Implied prior mean of sigma_w^2 under H1: {b/(a-1) if a > 1 else float('nan'):.4f} (reference sigma0^2={sigma0_sq:.4f})")

    sse = window_results["sigma2_w"].to_numpy() * WINDOW_SIZE
    bf = np.array([bayes_factor(s, sigma0_sq, WINDOW_SIZE, a, b) for s in sse])

    actual_anomaly = window_results["actual_anomaly"].to_numpy()
    normal_mask = actual_anomaly == 0
    anomaly_mask = actual_anomaly == 1
    n_normal, n_anomaly = normal_mask.sum(), anomaly_mask.sum()

    print(f"\n-- Bayesian posterior odds, by decision threshold on BF --")
    print(f"{'threshold':>12s} {'false alarms':>18s} {'detected':>18s}  (Jeffreys label)")
    for thresh, jeffreys in [(1, "BF>1: weak"), (3, "BF>3: substantial"), (10, "BF>10: strong"), (30, "BF>30: very strong"), (100, "BF>100: decisive")]:
        flagged = bf > thresh
        fa = (flagged & normal_mask).sum()
        det = (flagged & anomaly_mask).sum()
        print(f"{thresh:>12d} {f'{fa}/{n_normal} ({fa/n_normal:.2%})':>18s} {f'{det}/{n_anomaly} ({det/n_anomaly:.2%})':>18s}  {jeffreys}")

    # also: what threshold would hit ~5% false alarms, for apples-to-apples comparison?
    normal_bf = np.sort(bf[normal_mask])[::-1]
    target_idx = int(0.05 * n_normal)
    matched_thresh = normal_bf[target_idx]
    flagged_matched = bf > matched_thresh
    fa_m = (flagged_matched & normal_mask).sum()
    det_m = (flagged_matched & anomaly_mask).sum()
    print(f"\nThreshold calibrated to ~5% false alarms: BF > {matched_thresh:.2f}")
    print(f"   false alarms: {fa_m}/{n_normal} ({fa_m/n_normal:.2%})")
    print(f"   detected:     {det_m}/{n_anomaly} ({det_m/n_anomaly:.2%})")

    print(f"\n(for comparison: F-theory 408/1190 (34.29%) det 97/100 (97%); "
          f"block bootstrap 68/1190 (5.71%) det 64/100 (64%); "
          f"ARMA-sieve bootstrap 550/1190 (46.22%) det 100/100 (100%))")


if __name__ == "__main__":
    main()
