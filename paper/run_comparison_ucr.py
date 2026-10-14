"""Same three-method comparison (F-theory / block bootstrap / Bayesian
posterior odds), applied to the UCR anomaly benchmark ("noise" category)
instead of the mice data -- the "another dataset" check requested
alongside the mice comparison.

Outputs: results/ucr/comparison/comparison_table.csv, one row per dataset.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import pandas as pd

from compare_methods import evaluate_three_methods
from run_ucr import find_train_end

WINDOW_SIZE = 100
ALPHA = 0.05
N_BOOTSTRAP = 2000


def main() -> None:
    import sys
    data_dir = Path("data/ucr_prepared/noise")
    raw_dir = Path("data/ucr_raw")
    dataset_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir())
    output_path = "results/ucr/comparison/comparison_table.csv"
    if len(sys.argv) > 1:
        only_names = set(sys.argv[1].split(","))
        dataset_dirs = [d for d in dataset_dirs if d.name in only_names]
        output_path = "results/ucr/comparison/comparison_table_resume.csv"

    rows = []
    for dataset_dir in dataset_dirs:
        name = dataset_dir.name
        t0 = time.time()
        try:
            train_end = find_train_end(raw_dir, name)
            ts = pd.read_csv(dataset_dir / "timeseries.csv")
            labels = pd.read_csv(dataset_dir / "labels.csv")
            values = ts["value"].to_numpy(dtype=float)
            is_anomaly = labels["is_anomaly"].to_numpy(dtype=int)

            if is_anomaly[:train_end].sum() > 0:
                print(f"{name}: skipped (dirty training segment)")
                continue

            train_values = values[:train_end]
            test_values = values[train_end:]
            test_is_anomaly = is_anomaly[train_end:]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = evaluate_three_methods(
                    train_values, test_values, test_is_anomaly, WINDOW_SIZE, ALPHA, 3, 3, N_BOOTSTRAP, seed=0
                )
            elapsed = time.time() - t0
            if result is None:
                print(f"{name}: skipped (insufficient normal/anomaly windows)")
                continue
            result["dataset"] = name
            rows.append(result)
            print(
                f"{name} ({elapsed:.1f}s): "
                f"F={result['f_false_alarm_rate']:.2%}/{result['f_detection_rate']:.2%}  "
                f"BB={result['bb_false_alarm_rate']:.2%}/{result['bb_detection_rate']:.2%}  "
                f"Bayes={result['bayes_false_alarm_rate']:.2%}/{result['bayes_detection_rate']:.2%}",
                flush=True,
            )
        except Exception as exc:
            print(f"{name}: failed ({exc})", flush=True)

    table = pd.DataFrame(rows)
    Path("results/ucr/comparison").mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    print(f"\nSaved {output_path} ({len(table)} rows)")

    if len(table):
        print("\n=== Mean across all UCR datasets ===")
        for method, label in [("f", "F-distribution theory"), ("bb", "Block bootstrap"), ("bayes", "Bayesian posterior odds")]:
            fa = table[f"{method}_false_alarm_rate"].mean()
            det = table[f"{method}_detection_rate"].mean()
            print(f"{label:28s}: false alarms {fa:.2%}, detected {det:.2%}")


if __name__ == "__main__":
    main()
