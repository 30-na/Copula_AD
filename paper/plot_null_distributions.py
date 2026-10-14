"""Null-distribution comparison: theoretical F pdf vs. bootstrap empirical
histogram of T* = sigma_w^2 / sigma0^2, for two illustrative UCR datasets
(ucr_067 at window=20, ucr_095 at window=100), with one-sided alpha=0.05
control limits marked for both.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from algorithm import fit_reference_model
from compare_methods import build_block_bootstrap_null, valid_normal_starts
from run_ucr import find_train_end

ALPHA = 0.05
N_BOOTSTRAP = 20000

COLOR_INK = "#0b0b0b"
COLOR_MUTED = "#6b6a65"
COLOR_THEORY = "#1a1a1a"
COLOR_BOOT = "#8a8a8a"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 10.5,
        "axes.labelsize": 10.5,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 8.5,
        "axes.linewidth": 0.8,
    }
)


def _style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_MUTED)
    ax.spines["bottom"].set_color(COLOR_MUTED)
    ax.tick_params(colors=COLOR_MUTED)


def load_dataset(dataset_name: str):
    dataset_dir = Path("data/ucr_prepared/noise") / dataset_name
    raw_dir = Path("data/ucr_raw")
    train_end = find_train_end(raw_dir, dataset_name)
    ts = pd.read_csv(dataset_dir / "timeseries.csv")
    labels = pd.read_csv(dataset_dir / "labels.csv")
    values = ts["value"].to_numpy(dtype=float)
    is_anomaly = labels["is_anomaly"].to_numpy(dtype=int)
    return values[:train_end], values[train_end:], is_anomaly[train_end:]


def panel_data(dataset_name: str, window_size: int):
    train_values, test_values, test_is_anomaly = load_dataset(dataset_name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = fit_reference_model(train_values, 3, 3)
        starts = valid_normal_starts(test_values, test_is_anomaly, window_size)
        rng = np.random.default_rng(0)
        t_star = build_block_bootstrap_null(test_values, starts, model, window_size, N_BOOTSTRAP, rng)
    f_upper = float(stats.f.ppf(1 - ALPHA, window_size, model["reference_df"]))
    bb_upper = float(np.percentile(t_star, 100 * (1 - ALPHA)))
    return t_star, model["reference_df"], f_upper, bb_upper


def plot_panel(ax, dataset_name: str, window_size: int, t_star: np.ndarray, ref_df: int, f_upper: float, bb_upper: float) -> None:
    _style_axis(ax)
    x_max = max(np.percentile(t_star, 99), f_upper, bb_upper) * 1.1
    xs = np.linspace(1e-6, x_max, 400)

    kde = stats.gaussian_kde(t_star)
    ax.fill_between(xs, kde(xs), color=COLOR_BOOT, alpha=0.35, linewidth=0)
    ax.plot(xs, kde(xs), color=COLOR_BOOT, linewidth=1.6, label="Bootstrap empirical null (KDE)")

    pdf = stats.f.pdf(xs, window_size, ref_df)
    ax.plot(xs, pdf, color=COLOR_THEORY, linewidth=1.4, label="Theoretical $F$ pdf")

    ax.axvline(f_upper, color=COLOR_THEORY, linewidth=1.1, linestyle="--", label=f"F-theory limit ($\\alpha$={ALPHA})")
    ax.axvline(bb_upper, color=COLOR_BOOT, linewidth=1.6, linestyle="--", label=f"Bootstrap limit ($\\alpha$={ALPHA})")

    ax.set_xlim(0, x_max)
    ax.set_xlabel("Variance ratio $T$")
    ax.set_ylabel("Density")
    ax.set_title(f"{dataset_name}\n(window={window_size}, one-sided)", loc="left", fontsize=9.5, color=COLOR_INK)
    ax.legend(loc="upper right", frameon=False)


def main() -> None:
    datasets = [
        ("ucr_067_DISTORTEDinsectEPG3", 20),
        ("ucr_095_NOISECIMIS44AirTemperature4", 20),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for ax, (name, window_size) in zip(axes, datasets):
        t_star, ref_df, f_upper, bb_upper = panel_data(name, window_size)
        plot_panel(ax, name, window_size, t_star, ref_df, f_upper, bb_upper)
        print(f"{name}: F-theory limit={f_upper:.3f}  Bootstrap limit={bb_upper:.3f}")

    fig.tight_layout()
    output_path = Path("results/ucr/comparison/figures/null_distributions_theory_vs_bootstrap.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
