"""One-sided variant: put the full alpha budget into the upper tail only
(seizures only INCREASE variance, never decrease it -- same justification
Sohn & Farrar use for their one-sided F-test). Compares against the
two-sided numbers already in real_data_summary_table.csv.

F-theory one-sided: flag when ratio > F.ppf(1-alpha, n, ref_df)
                     (vs. two-sided: ratio outside [F.ppf(alpha/2,...), F.ppf(1-alpha/2,...)])
Bootstrap one-sided: flag when ratio > percentile(t_star, 100*(1-alpha))
                     (vs. two-sided: outside [2.5th, 97.5th] percentile)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from algorithm import fit_reference_model, make_windows, score_window, variance_ratio_pvalue
from compare_methods import build_block_bootstrap_null, valid_normal_starts

CONFIG = "2000hz_pre60s_ntrain1000"
N_TRAIN = 1000
WINDOW_SIZE = 100
ALPHA = 0.05
N_BOOTSTRAP = 2000


def run_one_combo(seizure_number: int, channel: int) -> dict | None:
    tag = f"seizure{seizure_number}" + (f"_ch{channel}" if channel != 1 else "")
    values = pd.read_csv(f"data/mice/{CONFIG}/timeseries_{tag}.csv")["value"].to_numpy(float)
    labels = pd.read_csv(f"data/mice/{CONFIG}/labels_{tag}.csv")["is_anomaly"].to_numpy(int)

    train_values = values[:N_TRAIN]
    test_values = values[N_TRAIN:]
    test_is_anomaly = labels[N_TRAIN:]

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
    ratio = results["variance_ratio"].to_numpy()

    # one-sided F-theory: full alpha in the upper tail
    f_upper_1s = float(stats.f.ppf(1 - ALPHA, WINDOW_SIZE, model["reference_df"]))
    flagged_f_1s = ratio > f_upper_1s
    fa_f_1s, det_f_1s = int((flagged_f_1s & normal_mask).sum()), int((flagged_f_1s & anomaly_mask).sum())

    # one-sided bootstrap: full alpha in the upper tail
    bb_upper_1s = np.percentile(t_star, 100 * (1 - ALPHA))
    flagged_bb_1s = ratio > bb_upper_1s
    fa_bb_1s, det_bb_1s = int((flagged_bb_1s & normal_mask).sum()), int((flagged_bb_1s & anomaly_mask).sum())

    return {
        "seizure": seizure_number, "channel": channel,
        "f1s_false_alarm_rate": fa_f_1s / n_normal, "f1s_detection_rate": det_f_1s / n_anomaly,
        "bb1s_false_alarm_rate": fa_bb_1s / n_normal, "bb1s_detection_rate": det_bb_1s / n_anomaly,
    }


def main() -> None:
    rows = []
    for seizure_number in range(1, 8):
        for channel in range(1, 5):
            r = run_one_combo(seizure_number, channel)
            if r is not None:
                rows.append(r)
            print(f"seizure{seizure_number} ch{channel}: "
                  f"F_1s={r['f1s_false_alarm_rate']:.2%}/{r['f1s_detection_rate']:.2%}  "
                  f"BB_1s={r['bb1s_false_alarm_rate']:.2%}/{r['bb1s_detection_rate']:.2%}", flush=True)

    one_sided = pd.DataFrame(rows).sort_values(["seizure", "channel"])
    two_sided = pd.read_csv("results/mice/comparison/real_data_summary_table.csv")
    merged = two_sided.merge(one_sided, on=["seizure", "channel"])
    merged.to_csv("results/mice/comparison/one_sided_vs_two_sided.csv", index=False)

    print("\n=== Mean across all 28 combos: two-sided vs one-sided ===")
    print(f"F-theory   two-sided: FA={merged['f_false_alarm_rate'].mean():.2%}  det={merged['f_detection_rate'].mean():.2%}")
    print(f"F-theory   one-sided: FA={merged['f1s_false_alarm_rate'].mean():.2%}  det={merged['f1s_detection_rate'].mean():.2%}")
    print(f"Bootstrap  two-sided: FA={merged['bb_false_alarm_rate'].mean():.2%}  det={merged['bb_detection_rate'].mean():.2%}")
    print(f"Bootstrap  one-sided: FA={merged['bb1s_false_alarm_rate'].mean():.2%}  det={merged['bb1s_detection_rate'].mean():.2%}")
    print(f"\nSaved results/mice/comparison/one_sided_vs_two_sided.csv")


if __name__ == "__main__":
    main()
