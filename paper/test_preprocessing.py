"""Test two preprocessing ideas against the current no-preprocessing
baseline, on seizure 1 / channel 1 / n_train=1000 / window_size=100:

  1. Slow-window variance normalization: divide the raw signal by a
     trailing rolling standard deviation (window=20000 raw samples = 10s,
     much longer than the ~5s seizure portion) BEFORE fitting/scoring.
     Should absorb slow baseline drift while leaving fast seizure-onset
     changes detectable.
  2. Standard EEG cleanup filters: 0.5 Hz high-pass (removes slow drift)
     + 60 Hz notch (powerline noise), zero-phase (filtfilt), applied to
     the raw signal before fitting/scoring.

Same diagnostics as before: out-of-sample Ljung-Box (dependency), Levene
heterogeneity across 10 time-blocks (stationarity), plus the practical
outcome (fraction flagged, detected, false alarms).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch

from algorithm import fit_reference_model, make_windows, run_algorithm, score_window
from diagnose_mice import residual_diagnostics, stationarity_check

CONFIG = "2000hz_pre60s_ntrain1000"
N_TRAIN = 1000
WINDOW_SIZE = 100
SEIZURE_NUMBER = 1
SAMPLE_RATE_HZ = 2000
ALPHA = 0.05


def slow_window_normalize(values: np.ndarray, window: int) -> np.ndarray:
    """Divide by a trailing (causal) rolling std, window=`window` samples.
    Expanding window for the warm-up period before enough history exists.
    """
    out = np.empty_like(values)
    cumsum = np.cumsum(values, dtype=float)
    cumsum_sq = np.cumsum(values**2, dtype=float)
    for i in range(len(values)):
        start = max(0, i - window + 1)
        n = i - start + 1
        s = cumsum[i] - (cumsum[start - 1] if start > 0 else 0.0)
        sq = cumsum_sq[i] - (cumsum_sq[start - 1] if start > 0 else 0.0)
        mean = s / n
        var = max(sq / n - mean**2, 1e-6)
        std = np.sqrt(var)
        out[i] = values[i] / std
    return out


def eeg_cleanup_filter(values: np.ndarray, fs: int) -> np.ndarray:
    b_hp, a_hp = butter(4, 0.5, btype="highpass", fs=fs)
    filtered = filtfilt(b_hp, a_hp, values)
    b_notch, a_notch = iirnotch(60.0, Q=30, fs=fs)
    filtered = filtfilt(b_notch, a_notch, filtered)
    return filtered


def evaluate(values: np.ndarray, labels: np.ndarray, label: str) -> None:
    train_values = values[:N_TRAIN]
    test_values = values[N_TRAIN:]

    model = fit_reference_model(train_values, 3, 3)
    results, _, _ = run_algorithm(train_values, test_values, WINDOW_SIZE, ALPHA, 3, 3)

    actual_anomaly = []
    for _, row in results.iterrows():
        start, end = int(row["window_start"]), int(row["window_end"])
        actual_anomaly.append(int(labels[N_TRAIN + start : N_TRAIN + end].any()))
    results["actual_anomaly"] = actual_anomaly

    windows = make_windows(test_values, WINDOW_SIZE)
    normal_idx = np.where(np.array(actual_anomaly) == 0)[0]
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(normal_idx, size=min(40, len(normal_idx)), replace=False)
    pooled_residuals = np.concatenate([score_window(windows[i], model)[1] for i in sample_idx])
    resid_stats = residual_diagnostics(pooled_residuals, "oos")
    stat = stationarity_check(results)

    n_true = int((results["actual_anomaly"] == 1).sum())
    n_normal = int((results["actual_anomaly"] == 0).sum())
    n_detected = int(((results["actual_anomaly"] == 1) & (results["p_value"] < ALPHA)).sum())
    n_false_alarms = int(((results["actual_anomaly"] == 0) & (results["p_value"] < ALPHA)).sum())

    print(f"=== {label} ===")
    print(f"  order={model['order']}, AIC={model['aic']:.1f}")
    print(f"  out-of-sample Ljung-Box p={resid_stats['ljung_box_p']:.4f} ({'white noise OK' if resid_stats['white_noise_ok'] else 'AUTOCORRELATED'})")
    print(f"  Levene p={stat['levene_p']:.4f} ({'HETEROGENEOUS' if stat['block_variance_heterogeneous'] else 'homogeneous'})")
    print(f"  detected={n_detected}/{n_true}, false_alarms={n_false_alarms}/{n_normal}, frac_flagged={(n_detected+n_false_alarms)/len(results):.3f}")
    print()


def main() -> None:
    values = pd.read_csv(f"data/mice/{CONFIG}/timeseries_seizure{SEIZURE_NUMBER}.csv")["value"].to_numpy(float)
    labels = pd.read_csv(f"data/mice/{CONFIG}/labels_seizure{SEIZURE_NUMBER}.csv")["is_anomaly"].to_numpy(int)

    evaluate(values, labels, "Baseline (no preprocessing)")

    normalized = slow_window_normalize(values, window=20000)
    evaluate(normalized, labels, "Slow-window variance normalization (10s trailing window)")

    filtered = eeg_cleanup_filter(values, SAMPLE_RATE_HZ)
    evaluate(filtered, labels, "EEG cleanup filters (0.5Hz high-pass + 60Hz notch)")

    differenced = np.diff(values)
    evaluate(differenced, labels[1:], "First-differenced (d=1)")


if __name__ == "__main__":
    main()
