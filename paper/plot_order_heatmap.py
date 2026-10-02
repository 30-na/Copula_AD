"""Heatmap: Ljung-Box statistic (out-of-sample) over the full (AR order p,
MA order q) grid, for one seizure/channel. x=AR(p), y=MA(q).

Setup (same as the rest of the mouse diagnostics):
  - config: 2000hz_pre60s_ntrain1000 (60s before onset, raw 2000 Hz, channel 1)
  - n_train = 1000 (reference-model training samples)
  - window_size (n_w) = 100 (test-scoring window)
  - out-of-sample residuals: pooled from 40 randomly sampled NORMAL
    (non-seizure-labeled) test windows, scored with the fixed reference
    params via algorithm.score_window -- same scoring the pipeline uses,
    just run across a (p,q) grid instead of the AIC-selected single order.
  - Ljung-Box computed at lags=10 (same as every other diagnostic so far).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from algorithm import fit_reference_model, make_windows, score_window

CONFIG = "2000hz_pre60s_ntrain1000"
N_TRAIN = 1000
WINDOW_SIZE = 100
SEIZURE_NUMBER = 1
MAX_P = 8
MAX_Q = 8
N_SAMPLE_WINDOWS = 40
LAGS = 10


def fit_pq(train_values: np.ndarray, p: int, q: int) -> dict | None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", UserWarning)
            fit = ARIMA(train_values, order=(p, 0, q)).fit()
    except Exception:
        return None
    if not np.isfinite(fit.aic):
        return None
    param_names = list(fit.param_names)
    params = np.asarray(fit.params, dtype=float)
    sigma_index = param_names.index("sigma2")
    fixed_params = np.delete(params, sigma_index)
    return {
        "order": (p, 0, q),
        "param_names": [name for name in param_names if name != "sigma2"],
        "fixed_params": fixed_params,
        "reference_sigma2": float(params[sigma_index]),
        "reference_df": len(train_values) - len(fixed_params),
        "aic": float(fit.aic),
    }


def out_of_sample_lb_stat(model: dict, test_values: np.ndarray, window_size: int, seed: int = 0) -> float | None:
    windows = make_windows(test_values, window_size)
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(len(windows), size=min(N_SAMPLE_WINDOWS, len(windows)), replace=False)
    try:
        pooled = np.concatenate([score_window(windows[i], model)[1] for i in sample_idx])
    except Exception:
        return None
    lb = acorr_ljungbox(pooled, lags=[LAGS], return_df=True)
    return float(lb["lb_stat"].iloc[0])


def main() -> None:
    values = pd.read_csv(f"data/mice/{CONFIG}/timeseries_seizure{SEIZURE_NUMBER}.csv")["value"].to_numpy(float)
    train_values = values[:N_TRAIN]
    test_values = values[N_TRAIN:]

    baseline = fit_reference_model(train_values, 3, 3)
    print(f"AIC-method baseline order: {baseline['order']}")

    p_values_range = list(range(0, MAX_P + 1))
    q_values_range = list(range(0, MAX_Q + 1))
    heat = np.full((len(q_values_range), len(p_values_range)), np.nan)

    for i, q in enumerate(q_values_range):
        for j, p in enumerate(p_values_range):
            if p == 0 and q == 0:
                continue
            model = fit_pq(train_values, p, q)
            if model is None:
                continue
            stat = out_of_sample_lb_stat(model, test_values, WINDOW_SIZE)
            if stat is not None:
                heat[i, j] = stat
            print(f"p={p} q={q}: LB stat={stat}")

    np.save("results/mice/figs/pq_heatmap_seizure1_data.npy", heat)

    from matplotlib.colors import LogNorm

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(heat, origin="lower", aspect="auto", cmap="viridis", norm=LogNorm(vmin=np.nanmin(heat), vmax=np.nanmax(heat)),
                    extent=(p_values_range[0] - 0.5, p_values_range[-1] + 0.5, q_values_range[0] - 0.5, q_values_range[-1] + 0.5))
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Ljung-Box statistic (out-of-sample, lags=10, log scale)")
    ax.set_xticks(p_values_range)
    ax.set_yticks(q_values_range)
    ax.set_xlabel("AR order (p)")
    ax.set_ylabel("MA order (q)")

    bp, _, bq = baseline["order"]
    ax.scatter([bp], [bq], marker="*", s=300, color="red", edgecolors="white", linewidths=0.8, zorder=5,
               label=f"AIC-method choice: order=({bp},0,{bq})")
    ax.legend(loc="upper right", fontsize=9, frameon=True, facecolor="white")

    ax.set_title(
        f"Out-of-sample Ljung-Box statistic over (p,q) grid — Seizure {SEIZURE_NUMBER}, channel 1\n"
        f"n_train={N_TRAIN}, window_size={WINDOW_SIZE}, {N_SAMPLE_WINDOWS} sampled normal windows",
        fontsize=11,
    )
    fig.tight_layout()
    output_path = "results/mice/figs/pq_heatmap_seizure1_ljungbox.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"\nSaved {output_path}")


if __name__ == "__main__":
    main()
