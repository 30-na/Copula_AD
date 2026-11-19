"""Naive baseline on all 250 UCR time series: rolling standard deviation
of the raw values (no ARMA model), under the same TSB-UAD rules as our
method IVR (02_run_ivr.py):
  - window length = TSB-UAD's find_length (read from ivr_scores/<ds>.json)
  - overlapping windows sliding by 1 point, whole series
  - each window's std goes to its middle point (TSB-UAD placement)
Computed with pandas (Series.rolling(w).std()).

Writes results/ucr/rolling_std_scores/<ds>.npz (point_score).
Usage: python 03_run_rolling_std.py   (after 02_run_ivr.py)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from importlib import import_module
_m = import_module("02_run_ivr")  # 02_run_ivr.py (same folder)
IVR_DIR, PREPARED, to_points = _m.OUT_DIR, _m.PREPARED, _m.to_points

OUT_DIR = Path("results/ucr/rolling_std_scores")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for jf in tqdm(sorted(IVR_DIR.glob("*.json")), ascii=True):
        meta = json.loads(jf.read_text())
        w = meta["window_length"]
        x = pd.read_csv(PREPARED / meta["category"] / meta["dataset"] / "timeseries.csv")["value"]
        per_window = x.rolling(w).std().to_numpy()[w - 1 :]  # one std per window start
        np.savez_compressed(OUT_DIR / f"{meta['dataset']}.npz", point_score=to_points(per_window, w))


if __name__ == "__main__":
    main()
