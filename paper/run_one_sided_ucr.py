"""One-sided F-theory vs. one-sided block bootstrap on the UCR benchmark
("noise" category), mirroring run_one_sided_test.py's mice version.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from algorithm import fit_reference_model, make_windows, score_window, variance_ratio_pvalue
from compare_methods import build_block_bootstrap_null, valid_normal_starts
from run_ucr import find_train_end

WINDOW_SIZE = 100
ALPHA = 0.05
N_BOOTSTRAP = 2000


def run_one_dataset(dataset_dir: Path, raw_dir: Path) -> dict | None:
    name = dataset_dir.name
    train_end = find_train_end(raw_dir, name)
    ts = pd.read_csv(dataset_dir / "timeseries.csv")
    labels = pd.read_csv(dataset_dir / "labels.csv")
    values = ts["value"].to_numpy(dtype=float)
    is_anomaly = labels["is_anomaly"].to_numpy(dtype=int)

    if is_anomaly[:train_end].sum() > 0:
        return None

    train_values = values[:train_end]
    test_values = values[train_end:]
    test_is_anomaly = is_anomaly[train_end:]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = fit_reference_model(train_values, 3, 3)
        windows = make_windows(test_values, WINDOW_SIZE)
        rows = []
        for i, w in enumerate(windows):
            sigma2_w, _ = score_window(w, model)
            ratio, _ = variance_ratio_pvalue(sigma2_w, WINDOW_SIZE, model)
            start, end = i * WINDOW_SIZE, (i + 1) * WINDOW_SIZE
            rows.append({"variance_ratio": ratio, "actual_anomaly": int(test_is_anomaly[start:end].any())})
        results = pd.DataFrame(rows)

        starts = valid_normal_starts(test_values, test_is_anomaly, WINDOW_SIZE)
        if len(starts) < 20:
            return None
        rng = np.random.default_rng(0)
        t_star = build_block_bootstrap_null(test_values, starts, model, WINDOW_SIZE, N_BOOTSTRAP, rng)

    normal_mask = results["actual_anomaly"].to_numpy() == 0
    anomaly_mask = results["actual_anomaly"].to_numpy() == 1
    n_normal, n_anomaly = int(normal_mask.sum()), int(anomaly_mask.sum())
    if n_normal == 0 or n_anomaly == 0:
        return None
    ratio = results["variance_ratio"].to_numpy()

    f_upper_1s = float(stats.f.ppf(1 - ALPHA, WINDOW_SIZE, model["reference_df"]))
    flagged_f_1s = ratio > f_upper_1s
    fa_f_1s, det_f_1s = int((flagged_f_1s & normal_mask).sum()), int((flagged_f_1s & anomaly_mask).sum())

    bb_upper_1s = np.percentile(t_star, 100 * (1 - ALPHA))
    flagged_bb_1s = ratio > bb_upper_1s
    fa_bb_1s, det_bb_1s = int((flagged_bb_1s & normal_mask).sum()), int((flagged_bb_1s & anomaly_mask).sum())

    return {
        "dataset": name,
        "f1s_false_alarm_rate": fa_f_1s / n_normal, "f1s_detection_rate": det_f_1s / n_anomaly,
        "bb1s_false_alarm_rate": fa_bb_1s / n_normal, "bb1s_detection_rate": det_bb_1s / n_anomaly,
    }


def main() -> None:
    data_dir = Path("data/ucr_prepared/noise")
    raw_dir = Path("data/ucr_raw")
    dataset_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir())

    rows = []
    for dataset_dir in dataset_dirs:
        try:
            r = run_one_dataset(dataset_dir, raw_dir)
        except Exception as exc:
            print(f"{dataset_dir.name}: failed ({exc})", flush=True)
            continue
        if r is None:
            print(f"{dataset_dir.name}: skipped", flush=True)
            continue
        rows.append(r)
        print(f"{dataset_dir.name}: F_1s={r['f1s_false_alarm_rate']:.2%}/{r['f1s_detection_rate']:.2%}  "
              f"BB_1s={r['bb1s_false_alarm_rate']:.2%}/{r['bb1s_detection_rate']:.2%}", flush=True)

    one_sided = pd.DataFrame(rows)
    two_sided = pd.read_csv("results/ucr/comparison/comparison_table.csv")[["dataset", "f_false_alarm_rate", "f_detection_rate", "bb_false_alarm_rate", "bb_detection_rate"]]
    merged = two_sided.merge(one_sided, on="dataset")
    merged.to_csv("results/ucr/comparison/one_sided_vs_two_sided.csv", index=False)

    print("\n=== Mean across all UCR datasets: two-sided vs one-sided ===")
    print(f"F-theory   two-sided: FA={merged['f_false_alarm_rate'].mean():.2%}  det={merged['f_detection_rate'].mean():.2%}")
    print(f"F-theory   one-sided: FA={merged['f1s_false_alarm_rate'].mean():.2%}  det={merged['f1s_detection_rate'].mean():.2%}")
    print(f"Bootstrap  two-sided: FA={merged['bb_false_alarm_rate'].mean():.2%}  det={merged['bb_detection_rate'].mean():.2%}")
    print(f"Bootstrap  one-sided: FA={merged['bb1s_false_alarm_rate'].mean():.2%}  det={merged['bb1s_detection_rate'].mean():.2%}")
    print(f"\nSaved results/ucr/comparison/one_sided_vs_two_sided.csv")


if __name__ == "__main__":
    main()
