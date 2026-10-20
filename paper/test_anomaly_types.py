"""Test whether the bootstrap-calibrated, one-sided variance-ratio method
works on anomaly types OTHER than "noise" (the only type tested so far).

Targets 4 anomaly types predicted (from their definitions) to cause a
genuine variance increase: outlier, steep_increase, amplitude_change,
unusual_pattern. Window size is set PER DATASET to that dataset's own
published anomaly length (not tuned against detection results -- the
anomaly start/end is part of the archive's filename metadata, known
before running any detector), rather than one fixed w for every dataset
regardless of scale.

Writes each result row immediately (crash-safe, resumable).
"""

from __future__ import annotations

import csv
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from algorithm import fit_reference_model, make_windows, score_window, variance_ratio_pvalue
from compare_methods import build_block_bootstrap_null, valid_normal_starts
from run_ucr import find_train_end

ALPHA = 0.05
N_BOOTSTRAP = 2000
MIN_WINDOW = 10

CATEGORIES = ["outlier", "steep_increase", "amplitude_change", "unusual_pattern"]
RAW_DIR = Path("../data/ucr_raw")
PREPARED_DIR = Path("../data/ucr_prepared")
OUT_PATH = "../results/ucr/comparison/anomaly_type_test.csv"
FIELDS = ["category", "dataset", "window_size", "anomaly_length", "n_normal", "n_anom", "fa_rate", "det_rate"]


def anomaly_length(dataset_name: str) -> int:
    m = re.match(r"ucr_(\d+)_", dataset_name)
    idx = m.group(1)
    matches = list(RAW_DIR.glob(f"{idx}_UCR_Anomaly_*.txt"))
    m2 = re.search(r"_(\d+)_(\d+)_(\d+)\.txt$", matches[0].name)
    train_end, a_start, a_end = (int(g) for g in m2.groups())
    return a_end - a_start


def load_done() -> set[tuple[str, str]]:
    if not os.path.exists(OUT_PATH):
        return set()
    df = pd.read_csv(OUT_PATH)
    return set(zip(df["category"], df["dataset"]))


def run_one(category: str, dataset_name: str) -> dict | None:
    dataset_dir = PREPARED_DIR / category / dataset_name
    train_end = find_train_end(RAW_DIR, dataset_name)
    a_len = anomaly_length(dataset_name)
    window_size = max(MIN_WINDOW, a_len)

    ts = pd.read_csv(dataset_dir / "timeseries.csv")
    labels = pd.read_csv(dataset_dir / "labels.csv")
    values = ts["value"].to_numpy(dtype=float)
    is_anomaly = labels["is_anomaly"].to_numpy(dtype=int)

    train_values = values[:train_end]
    test_values = values[train_end:]
    test_is_anomaly = is_anomaly[train_end:]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = fit_reference_model(train_values, 3, 3)
        windows = make_windows(test_values, window_size)
        rows = []
        for i, w in enumerate(windows):
            sigma2_w, _ = score_window(w, model)
            ratio, _ = variance_ratio_pvalue(sigma2_w, window_size, model)
            start, end = i * window_size, (i + 1) * window_size
            rows.append({"ratio": ratio, "anomaly": int(test_is_anomaly[start:end].any())})
        res = pd.DataFrame(rows)

        starts = valid_normal_starts(test_values, test_is_anomaly, window_size)
        if len(starts) < 20:
            return None
        rng = np.random.default_rng(0)
        t_star = build_block_bootstrap_null(test_values, starts, model, window_size, N_BOOTSTRAP, rng)

    normal = res["anomaly"] == 0
    anom = res["anomaly"] == 1
    n_normal, n_anom = int(normal.sum()), int(anom.sum())
    if n_normal == 0 or n_anom == 0:
        return None
    bb_upper = np.percentile(t_star, 100 * (1 - ALPHA))
    ratio = res["ratio"].to_numpy()
    flagged = ratio > bb_upper
    fa, det = int((flagged & normal).sum()), int((flagged & anom).sum())
    return {
        "category": category, "dataset": dataset_name, "window_size": window_size, "anomaly_length": a_len,
        "n_normal": n_normal, "n_anom": n_anom, "fa_rate": fa / n_normal, "det_rate": det / n_anom,
    }


def main() -> None:
    done = load_done()
    write_header = not os.path.exists(OUT_PATH)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
            f.flush()

        for category in CATEGORIES:
            dataset_dirs = sorted(p.name for p in (PREPARED_DIR / category).iterdir() if p.is_dir())
            for name in dataset_dirs:
                if (category, name) in done:
                    continue
                try:
                    r = run_one(category, name)
                except Exception as exc:
                    print(f"{category}/{name}: FAILED ({exc})", flush=True)
                    continue
                if r is None:
                    print(f"{category}/{name}: skipped", flush=True)
                    continue
                writer.writerow(r)
                f.flush()
                print(f"{category}/{name}: w={r['window_size']} FA={r['fa_rate']:.1%} det={r['det_rate']:.1%}", flush=True)

    print(f"\nDone. Results in {OUT_PATH}")


if __name__ == "__main__":
    main()
