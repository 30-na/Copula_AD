"""ACF plot of out-of-sample residuals for the single best (p,q) found in
the heatmap sweep (plot_order_heatmap.py): p=4, q=4, lowest Ljung-Box
statistic (77.1) of the whole 9x9 grid. Same setup: seizure 1, channel 1,
n_train=1000, window_size=100, 40 sampled normal test windows (seed=0).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from algorithm import fit_reference_model, make_windows, score_window
from diagnose_mice import residual_diagnostics
from plot_order_heatmap import CONFIG, N_TRAIN, WINDOW_SIZE, SEIZURE_NUMBER, N_SAMPLE_WINDOWS, fit_pq

COLOR_INK = "#0b0b0b"
COLOR_MUTED = "#6b6a65"
COLOR_LIMIT = "#8a8a8a"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 10.5,
        "axes.labelsize": 10.5,
        "axes.titlesize": 10.5,
        "axes.linewidth": 0.8,
    }
)


def acf_panel(ax, residuals: np.ndarray, title: str) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_MUTED)
    ax.spines["bottom"].set_color(COLOR_MUTED)
    ax.tick_params(colors=COLOR_MUTED)

    nlags = 20
    n = len(residuals)
    acf_vals = acf(residuals, nlags=nlags, fft=True)
    conf = 1.96 / np.sqrt(n)
    ax.bar(np.arange(nlags + 1), acf_vals, color=COLOR_INK, width=0.6)
    ax.axhline(conf, color=COLOR_LIMIT, linewidth=1.2, linestyle="--", label="95% confidence band")
    ax.axhline(-conf, color=COLOR_LIMIT, linewidth=1.2, linestyle="--")
    ax.axhline(0, color=COLOR_MUTED, linewidth=0.7)
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
    ax.set_title(title, fontsize=10.5)
    ax.legend(loc="upper right", fontsize=8.5, frameon=False)


def pooled_residuals_for(model: dict, test_values: np.ndarray, window_size: int) -> np.ndarray:
    windows = make_windows(test_values, window_size)
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(windows), size=min(N_SAMPLE_WINDOWS, len(windows)), replace=False)
    return np.concatenate([score_window(windows[i], model)[1] for i in sample_idx])


def main() -> None:
    best_p, best_q = 4, 4

    values = pd.read_csv(f"data/mice/{CONFIG}/timeseries_seizure{SEIZURE_NUMBER}.csv")["value"].to_numpy(float)
    train_values = values[:N_TRAIN]
    test_values = values[N_TRAIN:]

    best_model = fit_pq(train_values, best_p, best_q)
    best_residuals = pooled_residuals_for(best_model, test_values, WINDOW_SIZE)
    best_diag = residual_diagnostics(best_residuals, "out-of-sample")

    aic_model = fit_reference_model(train_values, 3, 3)
    aic_residuals = pooled_residuals_for(aic_model, test_values, WINDOW_SIZE)
    aic_diag = residual_diagnostics(aic_residuals, "out-of-sample")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))

    ap, _, aq = aic_model["order"]
    acf_panel(
        axes[0], aic_residuals,
        f"AIC-method order (p={ap}, q={aq}), Seizure {SEIZURE_NUMBER}\n"
        f"Ljung-Box p={aic_diag['ljung_box_p']:.4f} ({'white noise OK' if aic_diag['white_noise_ok'] else 'AUTOCORRELATED'}), n={len(aic_residuals)}",
    )
    acf_panel(
        axes[1], best_residuals,
        f"Best grid order (p={best_p}, q={best_q}), Seizure {SEIZURE_NUMBER}\n"
        f"Ljung-Box p={best_diag['ljung_box_p']:.4f} ({'white noise OK' if best_diag['white_noise_ok'] else 'AUTOCORRELATED'}), n={len(best_residuals)}",
    )

    fig.tight_layout()
    output_path = "results/mice/figs/best_order_acf_seizure1.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
