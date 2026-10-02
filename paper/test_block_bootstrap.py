"""Block bootstrap: stop trusting the F(window_size, reference_df) theory
and estimate the null distribution of the variance-ratio statistic
directly from real "normal" EEG, preserving local dependency structure.

Procedure:
  1. Pool: a long, label-confirmed-normal stretch of real test data
     (seizure 1, channel 1), scored with the SAME fixed reference model
     used everywhere else.
  2. To build one synthetic null window (length = window_size), sample
     several short consecutive BLOCKS (length < window_size) from the
     pool with replacement and stitch them together -- preserves local
     autocorrelation within each block, while the stitching points break
     up any single long-range structure (appropriate since the pool
     itself isn't literally iid across its length).
  3. Score each synthetic window with score_window (the exact same
     scoring code the real pipeline uses) to get T* = sigma2_w / sigma0^2.
  4. Repeat B times -> empirical null distribution of T*. Take its
     2.5th/97.5th percentiles as empirical control limits (replacing the
     F-distribution's theoretical quantiles).
  5. Apply THESE empirical limits to the REAL test windows (already
     scored, cached in window_results) and recompute the false-alarm
     rate and detection rate -- does it actually fix the real problem?
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from algorithm import fit_reference_model, make_windows, score_window

CONFIG = "2000hz_pre60s_ntrain1000"
N_TRAIN = 1000
WINDOW_SIZE = 100
SEIZURE_NUMBER = 1
ALPHA = 0.05
BLOCK_LENGTH = WINDOW_SIZE  # = window_size: each synthetic window is one real contiguous
# chunk, no internal stitching seams (sub-window block stitching creates
# artificial jumps at the seams that the model misreads as huge residuals --
# this inflates the synthetic null's variance far above real windows', an
# artifact discovered empirically on the first run of this script)
N_BOOTSTRAP = 5000


def f_critical_values(window_size: int, reference_df: int, alpha: float) -> tuple[float, float]:
    return (
        float(stats.f.ppf(alpha / 2, window_size, reference_df)),
        float(stats.f.ppf(1 - alpha / 2, window_size, reference_df)),
    )


def build_null_distribution(pool: np.ndarray, model: dict, window_size: int, block_length: int, n_bootstrap: int, rng: np.random.Generator) -> np.ndarray:
    n_blocks = window_size // block_length
    max_start = len(pool) - block_length
    t_star = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        starts = rng.integers(0, max_start, size=n_blocks)
        synthetic_window = np.concatenate([pool[s : s + block_length] for s in starts])
        sigma2_w, _ = score_window(synthetic_window, model)
        t_star[b] = sigma2_w / model["reference_sigma2"]
    return t_star


def main() -> None:
    values = pd.read_csv(f"data/mice/{CONFIG}/timeseries_seizure{SEIZURE_NUMBER}.csv")["value"].to_numpy(float)
    labels = pd.read_csv(f"data/mice/{CONFIG}/labels_seizure{SEIZURE_NUMBER}.csv")["is_anomaly"].to_numpy(int)
    window_results = pd.read_csv(f"results/mice/{CONFIG}/window_results_seizure{SEIZURE_NUMBER}.csv")

    train_values = values[:N_TRAIN]
    test_values = values[N_TRAIN:]
    model = fit_reference_model(train_values, 3, 3)
    print(f"Reference model: order={model['order']}, sigma0^2={model['reference_sigma2']:.4f}, reference_df={model['reference_df']}")

    normal_stretch = test_values[:100_000]
    assert labels[N_TRAIN : N_TRAIN + 100_000].sum() == 0, "pool unexpectedly overlaps the seizure"

    rng = np.random.default_rng(0)
    t_star = build_null_distribution(normal_stretch, model, WINDOW_SIZE, BLOCK_LENGTH, N_BOOTSTRAP, rng)
    np.save("results/mice/figs/block_bootstrap_t_star.npy", t_star)

    empirical_lower, empirical_upper = np.percentile(t_star, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    f_lower, f_upper = f_critical_values(WINDOW_SIZE, model["reference_df"], ALPHA)

    print(f"\nEmpirical null (block bootstrap, B={N_BOOTSTRAP}, block_length={BLOCK_LENGTH}):")
    print(f"  control limits: [{empirical_lower:.4f}, {empirical_upper:.4f}]")
    print(f"Theoretical F-distribution control limits: [{f_lower:.4f}, {f_upper:.4f}]")
    print()

    actual_anomaly = window_results["actual_anomaly"].to_numpy()
    ratio = window_results["variance_ratio"].to_numpy()
    normal_mask = actual_anomaly == 0
    anomaly_mask = actual_anomaly == 1

    flagged_f = (window_results["p_value"].to_numpy() < ALPHA)
    flagged_empirical = (ratio < empirical_lower) | (ratio > empirical_upper)

    n_normal = normal_mask.sum()
    n_anomaly = anomaly_mask.sum()
    fa_f = (flagged_f & normal_mask).sum()
    fa_emp = (flagged_empirical & normal_mask).sum()
    det_f = (flagged_f & anomaly_mask).sum()
    det_emp = (flagged_empirical & anomaly_mask).sum()

    print(f"-- Using theoretical F-distribution control limits (current pipeline) --")
    print(f"   false alarms: {fa_f}/{n_normal} ({fa_f/n_normal:.2%})")
    print(f"   detected:     {det_f}/{n_anomaly} ({det_f/n_anomaly:.2%})")
    print()
    print(f"-- Using empirical block-bootstrap control limits --")
    print(f"   false alarms: {fa_emp}/{n_normal} ({fa_emp/n_normal:.2%})")
    print(f"   detected:     {det_emp}/{n_anomaly} ({det_emp/n_anomaly:.2%})")


if __name__ == "__main__":
    main()
