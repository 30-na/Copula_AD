"""Step 2: Albert & Chib (1993)-style regime-switching extension.

Rather than one fixed reference variance (F-test) or one static
null distribution (block bootstrap / simple Bayes), this fits a
Markov-switching Gaussian model to log(variance ratio) across
non-overlapping windows of the REAL normal baseline -- explicitly
modeling the bursty, regime-like heterogeneity we found (Levene's
test: no linear trend, but significant block-to-block heterogeneity).

Procedure:
  1. Score every non-overlapping window_size window across a long,
     confirmed-normal stretch (same data used throughout) -> a sequence
     of log(T) values in chronological order.
  2. Fit a k-regime Markov-switching Gaussian model (statsmodels
     MarkovRegression, switching_variance=True) to that sequence --
     each regime has its own mean/variance of log(T), with a fitted
     transition matrix between regimes.
  3. Compute the model's STATIONARY regime probabilities (the long-run
     fraction of time spent in each regime), and build the implied
     marginal null distribution of T as a mixture of regimes.
  4. Draw many iid samples from that mixture (ignoring temporal order --
     we only need the marginal for a threshold) to get empirical
     percentiles, exactly as in test_block_bootstrap.py, so results are
     directly comparable.
  5. Apply to the REAL test windows and report false-alarm/detection
     rates alongside every method tried so far.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from algorithm import fit_reference_model, make_windows, score_window
from test_block_bootstrap import CONFIG, N_TRAIN, WINDOW_SIZE, SEIZURE_NUMBER, ALPHA


def build_logT_sequence(normal_stretch: np.ndarray, model: dict, window_size: int) -> np.ndarray:
    windows = make_windows(normal_stretch, window_size)
    t = np.empty(len(windows))
    for i, w in enumerate(windows):
        sigma2_w, _ = score_window(w, model)
        t[i] = sigma2_w / model["reference_sigma2"]
    return np.log(t)


def stationary_distribution(transition_matrix: np.ndarray) -> np.ndarray:
    """Stationary probabilities of a Markov chain. statsmodels'
    regime_transition is column-stochastic: T[i,j] = P(next=i | current=j),
    columns sum to 1. The stationary pi (column vector) satisfies T @ pi =
    pi, i.e. pi is a RIGHT eigenvector of T for eigenvalue 1 (not a left
    eigenvector / not of T.T).
    """
    eigvals, eigvecs = np.linalg.eig(transition_matrix)
    idx = np.argmin(np.abs(eigvals - 1.0))
    vec = np.real(eigvecs[:, idx])
    return vec / vec.sum()


def main() -> None:
    values = pd.read_csv(f"data/mice/{CONFIG}/timeseries_seizure{SEIZURE_NUMBER}.csv")["value"].to_numpy(float)
    labels = pd.read_csv(f"data/mice/{CONFIG}/labels_seizure{SEIZURE_NUMBER}.csv")["is_anomaly"].to_numpy(int)
    window_results = pd.read_csv(f"results/mice/{CONFIG}/window_results_seizure{SEIZURE_NUMBER}.csv")

    train_values = values[:N_TRAIN]
    test_values = values[N_TRAIN:]
    model = fit_reference_model(train_values, 3, 3)
    print(f"Reference model: order={model['order']}, sigma0^2={model['reference_sigma2']:.4f}")

    normal_stretch = test_values[:100_000]
    assert labels[N_TRAIN : N_TRAIN + 100_000].sum() == 0
    logT = build_logT_sequence(normal_stretch, model, WINDOW_SIZE)
    print(f"Built log(T) sequence: {len(logT)} non-overlapping windows from the real normal baseline")

    best = None
    for k in [2, 3]:
        fit = MarkovRegression(logT, k_regimes=k, trend="c", switching_variance=True).fit()
        # params layout (verified via fit.model.param_names): first k*(k-1)
        # entries are transition probabilities p[i->j], THEN const[0..k-1],
        # THEN sigma2[0..k-1] -- NOT [means..., variances...] as a naive
        # slice from the front would assume (that grabbed transition
        # probabilities instead of regime means on the first pass).
        n_trans = k * (k - 1)
        means = fit.params[n_trans : n_trans + k]
        variances = fit.params[n_trans + k : n_trans + 2 * k]
        print(f"\nk_regimes={k}: log-likelihood={fit.llf:.2f}, AIC={fit.aic:.2f}")
        for i in range(k):
            print(f"  regime {i}: mean={means[i]:.4f}, variance={variances[i]:.4f}")
        if best is None or fit.aic < best.aic:
            best = fit
            best_k = k
            best_n_trans = n_trans

    print(f"\nSelected k_regimes={best_k} (lowest AIC)")
    transition_matrix = best.regime_transition.squeeze(axis=-1)
    print(f"Transition matrix:\n{transition_matrix}")
    stationary = stationary_distribution(transition_matrix)
    print(f"Stationary regime probabilities: {np.round(stationary, 4)}")

    regime_means = best.params[best_n_trans : best_n_trans + best_k]
    regime_vars = best.params[best_n_trans + best_k : best_n_trans + 2 * best_k]
    print(f"Regime means (log T): {np.round(regime_means, 4)}")
    print(f"Regime variances (log T): {np.round(regime_vars, 4)}")

    rng = np.random.default_rng(0)
    n_samples = 50_000
    regime_choice = rng.choice(best_k, size=n_samples, p=stationary)
    mixture_logT = rng.normal(regime_means[regime_choice], np.sqrt(regime_vars[regime_choice]))
    mixture_t = np.exp(mixture_logT)

    empirical_lower, empirical_upper = np.percentile(mixture_t, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    print(f"\nRegime-switching mixture control limits: [{empirical_lower:.4f}, {empirical_upper:.4f}]")

    actual_anomaly = window_results["actual_anomaly"].to_numpy()
    ratio = window_results["variance_ratio"].to_numpy()
    flagged = (ratio < empirical_lower) | (ratio > empirical_upper)
    normal_mask = actual_anomaly == 0
    anomaly_mask = actual_anomaly == 1
    n_normal, n_anomaly = normal_mask.sum(), anomaly_mask.sum()
    fa = (flagged & normal_mask).sum()
    det = (flagged & anomaly_mask).sum()

    print(f"\n-- Regime-switching mixture --")
    print(f"   false alarms: {fa}/{n_normal} ({fa/n_normal:.2%})")
    print(f"   detected:     {det}/{n_anomaly} ({det/n_anomaly:.2%})")
    print(f"\n(for comparison: F-theory 408/1190 (34.29%) det 97/100 (97%); "
          f"block bootstrap 68/1190 (5.71%) det 64/100 (64%); "
          f"ARMA-sieve bootstrap 550/1190 (46.22%) det 100/100 (100%); "
          f"Bayesian posterior odds @ matched 5%: 59/1190 (4.96%) det 75/100 (75%))")


if __name__ == "__main__":
    main()
