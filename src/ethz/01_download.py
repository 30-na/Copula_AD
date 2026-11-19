"""Download N random ETHZ test-group time series (seed 0) that have a P pick.

Raw waveforms as stored by SeisBench (3 channels ZNE, counts, original sampling rate,
not filtered, not resampled), read remotely from the SeisBench mirror.

Run (seis env): python src/ethz/01_download.py all   (or N: N random ones, for a quick test)
Needs SeisBench's ETHZ metadata (downloaded here if missing): data/raw/ethz/seisbench_metadata.csv
Output: data/raw/ethz/metadata.csv, waves_part<c>.npz (500 time series each, key = row
index; saved as they finish, finished parts are skipped on a restart). Time series whose name
does not have the usual "bucketX$i,:3,:n" form (1 in the test group) are left out.
"""
import re
import sys
from pathlib import Path

import fsspec
import h5py
import numpy as np
import pandas as pd

ARG = sys.argv[1] if len(sys.argv) > 1 else "100"  # a number, or "all"
URL = "https://hifis-storage.desy.de/Helmholtz/HelmholtzAI/SeisBench/datasets/ethz/waveforms.hdf5"
OUT = Path("data/raw/ethz") if ARG == "all" else Path(f"data/raw/ethz_test{ARG}")
META = Path("data/raw/ethz/seisbench_metadata.csv")
META_URL = "https://hifis-storage.desy.de/Helmholtz/HelmholtzAI/SeisBench/datasets/ethz/metadata.csv"
P_COLS = ["trace_P1_arrival_sample", "trace_Pg_arrival_sample", "trace_Pn_arrival_sample",
          "trace_P_arrival_sample", "trace_PmP_arrival_sample"]
S_COLS = ["trace_S1_arrival_sample", "trace_Sg_arrival_sample", "trace_Sn_arrival_sample", "trace_SmS_arrival_sample"]


def main():
    if not META.exists():
        META.parent.mkdir(parents=True, exist_ok=True)
        with fsspec.open(META_URL, "rb") as f:
            META.write_bytes(f.read())
    m = pd.read_csv(META, low_memory=False)
    m["first_p_sample"] = m[P_COLS].min(axis=1)
    m["first_s_sample"] = m[S_COLS].min(axis=1)
    ok = m["trace_name"].str.match(r"^bucket\d+\$\d+,:3,:\d+$")
    sel = m[(m["split"] == "test") & m["first_p_sample"].notna() & ok]
    sel = (sel if ARG == "all" else sel.sample(int(ARG), random_state=0)).reset_index(drop=True)
    N = len(sel)
    OUT.mkdir(parents=True, exist_ok=True)
    sel.to_csv(OUT / "metadata.csv", index=False)
    h5 = h5py.File(fsspec.open(URL, "rb", block_size=2**20).open(), "r")
    waves = {}
    for k, name in enumerate(sel["trace_name"]):
        part = OUT / f"waves_part{k // 500}.npz"
        if part.exists():
            continue
        b, i, n = re.match(r"(bucket\d+)\$(\d+),:3,:(\d+)", name).groups()
        for attempt in range(5):
            try:
                waves[str(k)] = np.asarray(h5["data"][b][int(i), :, :int(n)], dtype=np.float32)
                break
            except Exception as err:  # noqa: BLE001
                print(f"retry {attempt + 1} for {name}: {type(err).__name__}", flush=True)
                h5 = h5py.File(fsspec.open(URL, "rb", block_size=2**20).open(), "r")
        else:
            raise RuntimeError(f"could not download {name}")
        if (k + 1) % 500 == 0 or k + 1 == N:
            np.savez_compressed(part, **waves)
            waves = {}
            print(f"downloaded {k + 1}/{N}", flush=True)


if __name__ == "__main__":
    main()
