"""Compare IVR (our method) on the 250 UCR time series with the 12 models whose
per-series results TSB-UAD published
(baselines/TSB-UAD/result/accuracy_table/mergedTable_<metric>.csv).
Reads saved outputs only (02_run_ivr.py), runs nothing.

Our metrics are computed exactly as in TSB-UAD's code
(TSB_UAD/vus/basic_metrics.py, metric_new / metric_PR):
  AUC_ROC   sklearn roc_auc_score, per point, all points
  AUC_PR    sklearn average_precision_score
  Precision, Recall, F   threshold mean + 3 std of the scores (all points)
A series where our method failed gets AUC 0.5 and the rest 0.

Writes results/ucr/
  per_series.csv   one row per series: our metrics + the published ones
  summary.csv      mean of each metric per model, over the series that
                   both have (also per anomaly category, summary_by_category.csv)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PUB_DIR = Path("baselines/TSB-UAD/result/accuracy_table")
RUN_DIR = Path("results/ucr/ivr_scores")
STD_DIR = Path("results/ucr/rolling_std_scores")  # naive baseline, 03_run_rolling_std.py
OUT_DIR = Path("results/ucr")
PREPARED = Path("data/processed/ucr")
# UCR_LABELS=tsb: score with TSB-UAD's own labels (data/raw/tsb_uad/.../KDD21/*.out, anomaly
# begin..end-1) instead of ours (begin..end); outputs get the suffix _tsblabels. Temporary option.
TSB_LABELS = os.environ.get("UCR_LABELS") == "tsb"
TSB_DIR = Path("data/raw/tsb_uad/TSB-UAD-Public/KDD21")
SUFFIX = "_tsblabels" if TSB_LABELS else ""


def labels(name: str, category: str) -> np.ndarray:
    if TSB_LABELS:
        f = next(TSB_DIR.glob(f"{name.split('_')[1]}_*.out"))
        return np.loadtxt(f, delimiter=",", usecols=1).astype(int)
    return pd.read_csv(PREPARED / category / name / "labels.csv")["is_anomaly"].to_numpy(dtype=int)
MODELS = ["NORMA", "MP", "LOF", "AE", "CNN", "IFOREST", "LSTM", "OCSVM", "HBOS", "IFOREST1", "POLY", "PCA"]
METRICS = ["AUC_ROC", "AUC_PR", "F", "Precision", "Recall", "Precision@k", "RF"]
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "baselines" / "TSB-UAD" / "TSB_UAD" / "vus"))
from basic_metrics import basic_metricor  # noqa: E402  (TSB-UAD's metric code)

METRICOR = basic_metricor()


def two_sided(ratio: np.ndarray, df1: int, df2: int) -> np.ndarray:
    """-log two-sided F p-value of R_w; p underflowing to 0 ranked by |log R_w|."""
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = np.minimum(0.0, np.log(2.0) + np.minimum(stats.f.logcdf(ratio, df1, df2), stats.f.logsf(ratio, df1, df2)))
        key = -lp
        inf = ~np.isfinite(key) & np.isfinite(ratio)
        if inf.any():
            key[inf] = np.nanmax(np.where(np.isfinite(key), key, np.nan)) + 1.0 + np.abs(np.log(ratio[inf]))
    return key


def our_metrics(score: np.ndarray, y: np.ndarray) -> dict:
    """TSB-UAD's own metric code (basic_metricor.metric_new / metric_PR) on the
    score min-max scaled to [0, 1], as in their pipeline. The scaling changes
    nothing except Precision@k, whose TSB-UAD implementation compares the
    0/1 flags with a score percentile and so depends on the score scale."""
    s = (score - score.min()) / (score.max() - score.min())
    auc, prec, rec, f, rrec, _, _, rprec, rf, p_at_k = METRICOR.metric_new(y, s)
    return {"AUC_ROC": auc, "AUC_PR": METRICOR.metric_PR(y, s)[2], "F": f, "Precision": prec, "Recall": rec,
            "Precision@k": p_at_k, "RF": rf}


def ivr() -> pd.DataFrame:
    rows = []
    for jf in sorted(RUN_DIR.glob("*.json")):
        meta = json.loads(jf.read_text())
        name = meta["dataset"]
        y = labels(name, meta["category"])
        if not y.any():  # resperation2 (begin == end): no anomaly under TSB-UAD's labels, no published result
            continue
        base = {"id": name.split("_")[1], "dataset": name, "category": meta["category"]}
        if meta.get("error"):
            fail = {"AUC_ROC": 0.5, "AUC_PR": float(y.mean()), "F": 0.0, "Precision": 0.0, "Recall": 0.0,
                    "Precision@k": 0.0, "RF": 0.0}
            for side in ("IVR_1s", "IVR_2s"):
                rows.append({**base, "model": side, **fail, "failed": 1})
            continue
        r = np.load(RUN_DIR / f"{name}.npz")["point_ratio"]
        for side, s in (("IVR_1s", r), ("IVR_2s", two_sided(r, meta["window_length"], meta["reference_df"]))):
            rows.append({**base, "model": side, **our_metrics(s, y), "failed": 0})
        rs = STD_DIR / f"{name}.npz"
        if rs.exists():
            rows.append({**base, "model": "ROLLING_STD", **our_metrics(np.load(rs)["point_score"], y), "failed": 0})
    return pd.DataFrame(rows)


def published() -> pd.DataFrame:
    parts = []
    for m in METRICS:
        t = pd.read_csv(PUB_DIR / f"mergedTable_{m}.csv")
        t = t[t["dataset"] == "KDD21"]
        t = t.assign(id=t["filename"].str[:3]).melt(id_vars="id", value_vars=MODELS, var_name="model", value_name=m)
        parts.append(t.set_index(["id", "model"]))
    return pd.concat(parts, axis=1).reset_index()


def main() -> None:
    o = ivr()
    pub = published()
    common = sorted(set(o["id"]) & set(pub["id"]))
    o = o[o["id"].isin(common)]
    cat = o.drop_duplicates("id").set_index("id")["category"]
    pub = pub[pub["id"].isin(common)].assign(category=lambda d: d["id"].map(cat))
    allm = pd.concat([o.drop(columns=["dataset", "failed"]), pub], ignore_index=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    allm.to_csv(OUT_DIR / f"per_series{SUFFIX}.csv", index=False)
    summary = allm.groupby("model")[METRICS].mean().sort_values("AUC_ROC", ascending=False)
    summary.insert(0, "n_series", allm.groupby("model")["id"].nunique())
    # how often our two-sided score beats each model on AUC-ROC
    piv = allm.pivot(index="id", columns="model", values="AUC_ROC")
    summary["IVR_2s_wins_AUC_ROC"] = [(piv["IVR_2s"] > piv[m]).mean() if m in piv else np.nan for m in summary.index]
    summary.to_csv(OUT_DIR / f"summary{SUFFIX}.csv")
    by_cat = allm.groupby(["category", "model"])["AUC_ROC"].mean().unstack()
    by_cat.insert(0, "n_series", allm.groupby("category")["id"].nunique())
    by_cat.to_csv(OUT_DIR / f"summary_by_category{SUFFIX}.csv")

    pd.set_option("display.width", 250)
    print(f"{len(common)} series in both (IVR failed on {int(o.loc[o.model == 'IVR_2s', 'failed'].sum())})\n")
    print(summary.round(3).to_string())
    print("\nAUC-ROC by anomaly category:")
    print(by_cat.round(2).to_string())
    print(f"\n-> {OUT_DIR}/")


if __name__ == "__main__":
    main()
