"""ACF comparison: global fixed reference model vs. local adaptive
(refit right before each window) -- same pooled residuals computed in
test_local_adaptive_ar.py, just visualized instead of printed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from algorithm import fit_reference_model, make_windows, score_window
from diagnose_mice import residual_diagnostics
from plot_best_order_acf import acf_panel
from test_local_adaptive_ar import CONFIG, N_TRAIN, WINDOW_SIZE, SEIZURE_NUMBER, N_SAMPLE_WINDOWS


def main() -> None:
    values = pd.read_csv(f"data/mice/{CONFIG}/timeseries_seizure{SEIZURE_NUMBER}.csv")["value"].to_numpy(float)
    labels = pd.read_csv(f"data/mice/{CONFIG}/labels_seizure{SEIZURE_NUMBER}.csv")["is_anomaly"].to_numpy(int)

    train_values = values[:N_TRAIN]
    test_values = values[N_TRAIN:]
    windows = make_windows(test_values, WINDOW_SIZE)

    actual_anomaly = []
    for i in range(len(windows)):
        start, end = N_TRAIN + i * WINDOW_SIZE, N_TRAIN + (i + 1) * WINDOW_SIZE
        actual_anomaly.append(int(labels[start:end].any()))
    normal_idx = np.where(np.array(actual_anomaly) == 0)[0]

    rng = np.random.default_rng(0)
    sample_idx = rng.choice(normal_idx, size=min(N_SAMPLE_WINDOWS, len(normal_idx)), replace=False)

    global_model = fit_reference_model(train_values, 3, 3)
    global_residuals = np.concatenate([score_window(windows[i], global_model)[1] for i in sample_idx])
    global_diag = residual_diagnostics(global_residuals, "global fixed")

    local_residuals_list = []
    for i in sample_idx:
        abs_window_start = N_TRAIN + i * WINDOW_SIZE
        local_train = values[abs_window_start - N_TRAIN : abs_window_start]
        local_model = fit_reference_model(local_train, 3, 3)
        _, resid = score_window(windows[i], local_model)
        local_residuals_list.append(resid)
    local_residuals = np.concatenate(local_residuals_list)
    local_diag = residual_diagnostics(local_residuals, "local adaptive")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))

    gp, _, gq = global_model["order"]
    acf_panel(
        axes[0], global_residuals,
        f"Global fixed model (p={gp}, q={gq}), Seizure {SEIZURE_NUMBER}\n"
        f"Ljung-Box p={global_diag['ljung_box_p']:.4f} ({'white noise OK' if global_diag['white_noise_ok'] else 'AUTOCORRELATED'}), "
        f"Jarque-Bera p={global_diag['jarque_bera_p']:.4f}",
    )
    acf_panel(
        axes[1], local_residuals,
        f"Local adaptive (refit per window), Seizure {SEIZURE_NUMBER}\n"
        f"Ljung-Box p={local_diag['ljung_box_p']:.4f} ({'white noise OK' if local_diag['white_noise_ok'] else 'AUTOCORRELATED'}), "
        f"Jarque-Bera p={local_diag['jarque_bera_p']:.4f}",
    )

    fig.tight_layout()
    output_path = "results/mice/figs/local_adaptive_acf_seizure1.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
