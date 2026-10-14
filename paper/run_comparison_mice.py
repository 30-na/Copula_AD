"""Run the three-method comparison (F-theory / block bootstrap / Bayesian
posterior odds) across all 7 seizures x 4 channels, config
2000hz_pre60s_ntrain1000 (window=100, n_train=1000, 60s-before extraction).

Outputs: results/mice/comparison/comparison_table.csv (one row per
seizure x channel x method) and figures/comparison_summary.png.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from compare_methods import evaluate_three_methods

CONFIG = "2000hz_pre60s_ntrain1000"
N_TRAIN = 1000
WINDOW_SIZE = 100
ALPHA = 0.05
N_BOOTSTRAP = 2000


def main() -> None:
    import sys
    only_combos = None
    output_path = "results/mice/comparison/comparison_table.csv"
    if len(sys.argv) > 1:
        # e.g. "6:3,6:4,7:1,7:2,7:3,7:4"
        only_combos = [(int(a), int(b)) for a, b in (p.split(":") for p in sys.argv[1].split(","))]
        output_path = "results/mice/comparison/comparison_table_resume.csv"

    rows = []
    for seizure_number in range(1, 8):
        for channel in range(1, 5):
            if only_combos is not None and (seizure_number, channel) not in only_combos:
                continue
            tag = f"seizure{seizure_number}" + (f"_ch{channel}" if channel != 1 else "")
            t0 = time.time()
            values = pd.read_csv(f"data/mice/{CONFIG}/timeseries_{tag}.csv")["value"].to_numpy(float)
            labels = pd.read_csv(f"data/mice/{CONFIG}/labels_{tag}.csv")["is_anomaly"].to_numpy(int)

            train_values = values[:N_TRAIN]
            test_values = values[N_TRAIN:]
            test_is_anomaly = labels[N_TRAIN:]

            result = evaluate_three_methods(
                train_values, test_values, test_is_anomaly, WINDOW_SIZE, ALPHA, 3, 3, N_BOOTSTRAP, seed=0
            )
            elapsed = time.time() - t0
            if result is None:
                print(f"seizure{seizure_number} ch{channel}: skipped (insufficient data)")
                continue
            result["seizure"] = seizure_number
            result["channel"] = channel
            rows.append(result)
            print(
                f"seizure{seizure_number} ch{channel} ({elapsed:.1f}s): "
                f"F={result['f_false_alarm_rate']:.2%}/{result['f_detection_rate']:.2%}  "
                f"BB={result['bb_false_alarm_rate']:.2%}/{result['bb_detection_rate']:.2%}  "
                f"Bayes={result['bayes_false_alarm_rate']:.2%}/{result['bayes_detection_rate']:.2%}",
                flush=True,
            )

    table = pd.DataFrame(rows)
    table.to_csv(output_path, index=False)
    print(f"\nSaved {output_path} ({len(table)} rows)")

    print("\n=== Mean across all seizure x channel combos ===")
    for method, label in [("f", "F-distribution theory"), ("bb", "Block bootstrap"), ("bayes", "Bayesian posterior odds")]:
        fa = table[f"{method}_false_alarm_rate"].mean()
        det = table[f"{method}_detection_rate"].mean()
        print(f"{label:28s}: false alarms {fa:.2%}, detected {det:.2%}")


if __name__ == "__main__":
    main()
