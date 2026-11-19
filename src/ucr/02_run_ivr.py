"""IVR (Innovation Variance Ratio, our method) on ALL 250 UCR anomaly time series (TSB-UAD's KDD21
collection), under the TSB-UAD rules, so it can be compared with the
results TSB-UAD published per time series
(baselines/TSB-UAD/result/accuracy_table/).

Rules (TSB-UAD):
  - window length = TSB-UAD's find_length (their code)
  - overlapping windows sliding by 1 point over the WHOLE series
  - each window's R_w goes to its middle point (TSB-UAD placement)
  - ARMA fit on the anomaly-free training part (AIC search up to (3,3))

Window scoring uses the closed form of the Kalman-filter variance: with
the ARMA parameters fixed, sigma2_w = (x - mu)' S^-1 (x - mu) / n, where
S is the ARMA autocovariance matrix of the window (sigma^2 = 1). This
equals score_window() (statsmodels Kalman filter, stationary start) to
1e-8 relative on all 23 noise series, and is ~1000x faster.

Saved per series (nothing needs a rerun):
  <ds>.npz   ratio (R_w per window start), point_ratio (per point), t_star
             (bootstrap sample of training-window R_w)
  <ds>.json  category, n, train_end, anomaly, window length, ARMA order and
             parameters, sigma0^2, df, runtimes, error (if any)

Usage: python 02_run_ivr.py   (from the project root)
Crash-safe: a series is skipped if its .json exists.
"""

from __future__ import annotations

import os

# one numerical thread per worker, so the free CPUs really stay free
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import math
import re
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.linalg import cholesky, solve_triangular, toeplitz
from statsmodels.tsa.arima_process import arma_acovf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "baselines" / "TSB-UAD" / "TSB_UAD" / "utils"))
from slidingWindows import find_length  # noqa: E402  (TSB-UAD's own function)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/ (ivr.py)
from ivr import fit_reference_model, to_points, window_ratios  # noqa: E402
from ivr import ucr_train_end as get_train_end  # noqa: E402

PREPARED = Path("data/processed/ucr")
OUT_DIR = Path("results/ucr/ivr_scores")
N_JOBS = max(1, (os.cpu_count() or 2) - 4)  # keep 4 logical CPUs free
N_BOOTSTRAP = 2000
SEED = 0


def all_series() -> list[Path]:
    return sorted(PREPARED.glob("*/ucr_*"), key=lambda p: p.name)


def run_one(path: Path) -> str:
    name = path.name
    x = pd.read_csv(path / "timeseries.csv")["value"].to_numpy(dtype=float)
    lab = pd.read_csv(path / "labels.csv")["is_anomaly"].to_numpy(dtype=int)
    te = get_train_end(name)
    meta = {"dataset": name, "category": path.parent.name, "n": len(x), "train_end": te,
            "anomaly_start": int(np.flatnonzero(lab)[0]), "anomaly_end": int(np.flatnonzero(lab)[-1])}
    try:
        w = int(find_length(x))
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = fit_reference_model(x[:te], 3, 3)
        fit_s = time.time() - t0
        t0 = time.time()
        ratio = window_ratios(x, model, w)
        train_ratio = ratio[: te - w + 1]
        t_star = np.random.default_rng(SEED).choice(train_ratio, N_BOOTSTRAP, replace=True)
        np.savez_compressed(OUT_DIR / f"{name}.npz", ratio=ratio, point_ratio=to_points(ratio, w), t_star=t_star)
        meta.update({
            "window_length": w, "n_bootstrap": N_BOOTSTRAP, "seed": SEED,
            "arma_order": list(model["order"]), "param_names": model["param_names"],
            "fixed_params": [float(v) for v in model["fixed_params"]],
            "reference_sigma2": model["reference_sigma2"], "reference_df": int(model["reference_df"]),
            "fit_runtime_s": fit_s, "score_runtime_s": time.time() - t0, "error": None,
        })
        msg = "ok"
    except Exception as e:  # recorded, counted as a failure in the evaluation
        meta["error"] = repr(e)
        msg = f"FAILED {e!r}"
    (OUT_DIR / f"{name}.json").write_text(json.dumps(meta, indent=2))  # written last: marks done
    return f"{name}: {msg}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    todo = [p for p in all_series() if not (OUT_DIR / f"{p.name}.json").exists()]
    # longest training parts first, so the slow ARMA fits don't end up last
    todo.sort(key=lambda p: -get_train_end(p.name))
    print(f"{len(todo)} of {len(all_series())} series to run, {N_JOBS} processes -> {OUT_DIR}/", flush=True)
    with Pool(N_JOBS) as pool:
        bar = tqdm(pool.imap_unordered(run_one, todo), total=len(todo), ascii=True, mininterval=10, file=sys.stdout)
        for msg in bar:
            if "FAILED" in msg:
                tqdm.write(msg, file=sys.stdout)
    print("done", flush=True)


if __name__ == "__main__":
    main()
