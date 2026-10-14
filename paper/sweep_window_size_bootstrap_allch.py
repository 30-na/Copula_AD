"""Bootstrap-only window-size sweep across ALL 4 channels x 7 seizures
for the single mouse (AC75a-5) in data/mice/2000hz_pre60s_ntrain1000.

The ARIMA reference model only depends on (channel, seizure) -- NOT on
window_size -- so it's fit once per (channel, seizure) and reused across
all window sizes, instead of refitting it from scratch 14x per combo.

Writes each (channel, seizure, window_size) result to CSV immediately
(append + flush) so a crash mid-sweep still leaves usable partial
results. Safe to re-run: already-computed combos are skipped.
"""

from __future__ import annotations

import csv
import os
import warnings

import numpy as np
import pandas as pd

from algorithm import fit_reference_model, make_windows, score_window, variance_ratio_pvalue
from compare_methods import build_block_bootstrap_null, valid_normal_starts

N_TRAIN = 1000
ALPHA = 0.05
N_BOOTSTRAP = 2000
WINDOW_SIZES = [10, 20, 30, 50, 75, 100, 150, 200, 250, 300, 400, 500, 750, 1000]
CHANNELS = [1, 2, 3, 4]
SEIZURES = list(range(1, 8))

OUT_PATH = "../results/mice/comparison/window_size_sweep_bootstrap_allch.csv"
FIELDS = ["channel", "seizure", "window_size", "n_normal", "n_anom", "fa_rate", "det_rate"]


def load_done() -> set[tuple[int, int, int]]:
    if not os.path.exists(OUT_PATH):
        return set()
    df = pd.read_csv(OUT_PATH)
    return set(zip(df["channel"], df["seizure"], df["window_size"]))


def score_at_window_size(test_values, test_is_anomaly, model, window_size: int) -> dict | None:
    windows = make_windows(test_values, window_size)
    rows = []
    for i, w in enumerate(windows):
        s2, _ = score_window(w, model)
        r, _ = variance_ratio_pvalue(s2, window_size, model)
        start, end = i * window_size, (i + 1) * window_size
        rows.append({"ratio": r, "anomaly": int(test_is_anomaly[start:end].any())})
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
        "window_size": window_size, "n_normal": n_normal, "n_anom": n_anom,
        "fa_rate": fa / n_normal, "det_rate": det / n_anom,
    }


def main() -> None:
    done = load_done()
    write_header = not os.path.exists(OUT_PATH)
    with open(OUT_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
            f.flush()

        for channel in CHANNELS:
            for s in SEIZURES:
                pending = [ws for ws in WINDOW_SIZES if (channel, s, ws) not in done]
                if not pending:
                    continue

                tag = f"seizure{s}" + (f"_ch{channel}" if channel != 1 else "")
                try:
                    values = pd.read_csv(f"../data/mice/2000hz_pre60s_ntrain1000/timeseries_{tag}.csv")["value"].to_numpy(float)
                    is_anomaly = pd.read_csv(f"../data/mice/2000hz_pre60s_ntrain1000/labels_{tag}.csv")["is_anomaly"].to_numpy(int)
                except FileNotFoundError as exc:
                    print(f"ch{channel} s{s}: FAILED to load ({exc})", flush=True)
                    continue
                train_values = values[:N_TRAIN]
                test_values = values[N_TRAIN:]
                test_is_anomaly = is_anomaly[N_TRAIN:]

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = fit_reference_model(train_values, 3, 3)

                    for ws in pending:
                        try:
                            r = score_at_window_size(test_values, test_is_anomaly, model, ws)
                        except Exception as exc:
                            print(f"ch{channel} s{s} w{ws}: FAILED ({exc})", flush=True)
                            continue
                        if r is None:
                            print(f"ch{channel} s{s} w{ws}: skipped", flush=True)
                            continue
                        row = {"channel": channel, "seizure": s, **r}
                        writer.writerow(row)
                        f.flush()
                        print(f"ch{channel} s{s} w{ws}: FA={r['fa_rate']:.1%} det={r['det_rate']:.1%}", flush=True)

    print(f"\nDone. Results in {OUT_PATH}")


if __name__ == "__main__":
    main()
