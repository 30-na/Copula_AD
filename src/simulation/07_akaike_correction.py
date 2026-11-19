"""Does Akaike's factor fix the too-large R_w seen in 06_f_df_check.py?

Corrected ratio: R_w * (n0 - k) / (n0 + k)  (Akaike's final prediction
error: out-of-sample / in-sample residual variance ~ (n0 + k) / (n0 - k)).
Compares false alarms at the 95% point of F(n, n0) and the KS p-value,
before and after the correction. Reuses df_comparison/ratios.csv.

Output: results/simulation/f_check/df_comparison/correction.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from importlib import import_module
_m = import_module("06_f_df_check")  # 06_f_df_check.py (same folder)
OUT, WINDOW = _m.OUT, _m.WINDOW


def main() -> None:
    ratios = pd.read_csv(OUT / "ratios.csv")
    rows = []
    for n0, g in ratios.groupby("n0"):
        k = int(g["k"].iloc[0])
        dist = stats.f(WINDOW, n0)
        lo, hi = dist.ppf(0.025), dist.ppf(0.975)
        for name, r in [("raw", g["ratio"].to_numpy()), ("corrected", g["ratio"].to_numpy() * (n0 - k) / (n0 + k))]:
            rows.append({"n0": n0, "k": k, "ratio": name, "reps": len(r), "mean_ratio": r.mean(), "theory_mean": dist.mean(),
                         "one_sided_0.05": float((r > dist.ppf(0.95)).mean()),
                         "two_sided_0.05": float(((r < lo) | (r > hi)).mean()),
                         "ks_pvalue": stats.kstest(r, dist.cdf).pvalue})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "correction.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
