"""Run all 6 methods (ours, MERLIN, naive rolling-std, CUSUM, Autoencoder,
LSTM) on all 23 UCR noise-category datasets, at a fixed window_size.

Output per dataset: one CSV with ONE ROW PER RAW SAMPLE (time = sample
index in the test region). Per-sample methods (MERLIN, naive rolling std)
keep their native per-sample value; per-window methods (ours, CUSUM, AE,
LSTM) repeat the SAME value across every sample in that window.

Also writes a separate thresholds table (dataset x method -> threshold),
since the final decision threshold per method is still open.

Crash-safe: each dataset's CSV is written as soon as it's done, and
already-completed datasets (file already on disk) are skipped on re-run.
"""

from __future__ import annotations

# stumpy (numba) must be imported before numpy/pandas/torch -- importing
# numpy+pandas first causes a native DLL conflict with torch on Windows.
import stumpy

import csv
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from algorithm import fit_reference_model, make_windows, score_window, variance_ratio_pvalue
from baselines import cusum_on_ratios, naive_rolling_std
from compare_methods import build_block_bootstrap_null, valid_normal_starts
from dl_baselines import autoencoder_scores, lstm_scores
from run_ucr import find_train_end

ALPHA = 0.05
N_BOOTSTRAP = 2000
WINDOW_SIZE = 100

NOISE_DIR = Path("data/ucr_prepared/noise")
RAW_DIR = Path("data/ucr_raw")
OUT_TS_DIR = Path("results/ucr/comparison/all_methods_timeseries")
OUT_THRESH_PATH = Path("results/ucr/comparison/all_methods_thresholds.csv")
THRESH_FIELDS = ["dataset", "method", "threshold"]


def broadcast_window_values(window_values: np.ndarray, window_size: int, n_samples: int) -> np.ndarray:
    out = np.full(n_samples, np.nan)
    for i, v in enumerate(window_values):
        start, end = i * window_size, min((i + 1) * window_size, n_samples)
        out[start:end] = v
    return out


def append_thresholds(rows: list[dict]) -> None:
    write_header = not OUT_THRESH_PATH.exists()
    OUT_THRESH_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_THRESH_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=THRESH_FIELDS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_dataset(dataset_name: str, window_size: int) -> None:
    dataset_dir = NOISE_DIR / dataset_name
    train_end = find_train_end(RAW_DIR, dataset_name)

    ts = pd.read_csv(dataset_dir / "timeseries.csv")
    labels = pd.read_csv(dataset_dir / "labels.csv")
    values = ts["value"].to_numpy(dtype=float)
    is_anomaly = labels["is_anomaly"].to_numpy(dtype=int)

    train_values = values[:train_end]
    test_values = values[train_end:]
    test_is_anomaly = is_anomaly[train_end:]
    n = len(test_values)

    print(f"  [{dataset_name}] MERLIN...", flush=True)
    stumpy.config.STUMPY_EXCL_ZONE_DENOM = 1
    mp = stumpy.stump(test_values, m=window_size)
    merlin_profile = mp[:, 0].astype(float)
    merlin_full = np.full(n, np.nan)
    merlin_full[: len(merlin_profile)] = merlin_profile
    merlin_threshold = float(np.percentile(merlin_profile, 100 * (1 - ALPHA)))

    print(f"  [{dataset_name}] our method (fit + bootstrap)...", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = fit_reference_model(train_values, 3, 3)
        windows = make_windows(test_values, window_size)
        ratios = np.empty(len(windows))
        for i, w in enumerate(windows):
            sigma2_w, _ = score_window(w, model)
            ratios[i], _ = variance_ratio_pvalue(sigma2_w, window_size, model)

        starts = valid_normal_starts(test_values, test_is_anomaly, window_size)
        rng = np.random.default_rng(0)
        t_star = build_block_bootstrap_null(test_values, starts, model, window_size, N_BOOTSTRAP, rng)
    bb_upper = float(np.percentile(t_star, 100 * (1 - ALPHA)))
    ours_full = broadcast_window_values(ratios, window_size, n)

    print(f"  [{dataset_name}] naive rolling std...", flush=True)
    naive_curve, naive_threshold = naive_rolling_std(train_values, test_values, window_size)

    print(f"  [{dataset_name}] CUSUM...", flush=True)
    cusum_vals, cusum_h = cusum_on_ratios(train_values, ratios, model, window_size)
    cusum_full = broadcast_window_values(cusum_vals, window_size, n)

    print(f"  [{dataset_name}] autoencoder...", flush=True)
    ae_err, ae_threshold = autoencoder_scores(train_values, test_values, window_size)
    ae_full = broadcast_window_values(ae_err, window_size, n)

    print(f"  [{dataset_name}] LSTM...", flush=True)
    lstm_err, lstm_threshold = lstm_scores(train_values, test_values, window_size)
    lstm_full = broadcast_window_values(lstm_err, window_size, n)

    out = pd.DataFrame({
        "time": np.arange(n),
        "value": test_values,
        "is_anomaly": test_is_anomaly,
        "ours_variance_ratio": ours_full,
        "merlin_nn_distance": merlin_full,
        "naive_rolling_std": naive_curve,
        "cusum": cusum_full,
        "autoencoder_error": ae_full,
        "lstm_error": lstm_full,
    })
    OUT_TS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_TS_DIR / f"{dataset_name}.csv", index=False)

    append_thresholds([
        {"dataset": dataset_name, "method": "ours", "threshold": bb_upper},
        {"dataset": dataset_name, "method": "merlin", "threshold": merlin_threshold},
        {"dataset": dataset_name, "method": "naive_rolling_std", "threshold": naive_threshold},
        {"dataset": dataset_name, "method": "cusum", "threshold": cusum_h},
        {"dataset": dataset_name, "method": "autoencoder", "threshold": ae_threshold},
        {"dataset": dataset_name, "method": "lstm", "threshold": lstm_threshold},
    ])
    print(f"  [{dataset_name}] done -> {OUT_TS_DIR / f'{dataset_name}.csv'}", flush=True)


def main() -> None:
    dataset_names = sorted(p.name for p in NOISE_DIR.iterdir() if p.is_dir())
    total = len(dataset_names)
    print(f"Running 6 methods on {total} noise-category datasets, window_size={WINDOW_SIZE}", flush=True)

    for idx, name in enumerate(dataset_names, start=1):
        out_path = OUT_TS_DIR / f"{name}.csv"
        if out_path.exists():
            print(f"[{idx}/{total}] {name}: already done, skipping", flush=True)
            continue
        print(f"[{idx}/{total}] {name}: starting", flush=True)
        try:
            run_dataset(name, WINDOW_SIZE)
        except Exception as exc:
            print(f"[{idx}/{total}] {name}: FAILED ({exc})", flush=True)
            continue

    print("\nAll datasets processed.", flush=True)
    print(f"Per-dataset time series: {OUT_TS_DIR}/", flush=True)
    print(f"Thresholds table: {OUT_THRESH_PATH}", flush=True)


if __name__ == "__main__":
    main()
