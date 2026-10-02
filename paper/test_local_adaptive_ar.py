"""Direct test: does refitting LOCALLY (on the 1000 samples immediately
preceding each test window, instead of one frozen model from the very
start of the recording) remove the residual autocorrelation we keep
finding?

For the same 40 sampled "normal" test windows used throughout, each
window gets its OWN freshly-fit reference model (same (p,q) AIC search,
same n_train=1000, just re-anchored right before that window instead of
at t=0). Residuals are pooled and run through the same Ljung-Box /
Levene diagnostics as the global-fixed-model baseline, so the comparison
is apples to apples.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from algorithm import fit_reference_model, make_windows, score_window
from diagnose_mice import residual_diagnostics

CONFIG = "2000hz_pre60s_ntrain1000"
N_TRAIN = 1000
WINDOW_SIZE = 100
SEIZURE_NUMBER = 1
N_SAMPLE_WINDOWS = 40


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

    # (A) global fixed model, same as every other diagnostic (reference point)
    global_model = fit_reference_model(train_values, 3, 3)
    global_residuals = np.concatenate([score_window(windows[i], global_model)[1] for i in sample_idx])
    global_diag = residual_diagnostics(global_residuals, "global fixed")

    # (B) local adaptive: refit on the 1000 samples immediately before EACH window
    local_residuals_list = []
    orders_used = []
    skipped = 0
    for i in sample_idx:
        abs_window_start = N_TRAIN + i * WINDOW_SIZE
        local_train_start = abs_window_start - N_TRAIN
        if local_train_start < 0:
            skipped += 1
            continue
        local_train = values[local_train_start:abs_window_start]
        try:
            local_model = fit_reference_model(local_train, 3, 3)
        except Exception:
            skipped += 1
            continue
        orders_used.append(local_model["order"])
        _, resid = score_window(windows[i], local_model)
        local_residuals_list.append(resid)

    local_residuals = np.concatenate(local_residuals_list)
    local_diag = residual_diagnostics(local_residuals, "local adaptive")

    print(f"Windows used: global={len(sample_idx)}, local={len(local_residuals_list)} (skipped={skipped})")
    print(f"Local orders used (first 10): {orders_used[:10]}")
    print()
    print(f"-- Global fixed model (one reference for the whole recording) --")
    print(f"   order={global_model['order']}")
    print(f"   Ljung-Box p={global_diag['ljung_box_p']:.4f}  ({'white noise OK' if global_diag['white_noise_ok'] else 'AUTOCORRELATED'})")
    print(f"   Jarque-Bera p={global_diag['jarque_bera_p']:.4f}")
    print()
    print(f"-- Local adaptive model (refit right before each window) --")
    print(f"   Ljung-Box p={local_diag['ljung_box_p']:.4f}  ({'white noise OK' if local_diag['white_noise_ok'] else 'AUTOCORRELATED'})")
    print(f"   Jarque-Bera p={local_diag['jarque_bera_p']:.4f}")


if __name__ == "__main__":
    main()
