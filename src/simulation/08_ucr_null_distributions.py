"""Null-distribution comparison: theoretical F pdf vs. bootstrap empirical
histogram of T* = sigma_w^2 / sigma0^2, for two illustrative UCR datasets
(ucr_067 and ucr_095, both at window=20), with one-sided alpha=0.05
control limits marked for both. Two corrected F curves are added:
  Akaike  F(n, n0) scaled by (n0 + k) / (n0 - k)  (estimated ARMA parameters)
  Box     F(n_eff, n0_eff), n_eff = 2n / (kappa - 1), n0_eff = 2 n0 / (kappa - 1),
          kappa = kurtosis of the training residuals (3 for normal noise)
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

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/ (ivr.py)
from ivr import fit_reference_model, score_window
from ivr import build_block_bootstrap_null, calibration_starts
from ivr import ucr_train_end as get_train_end

ALPHA = 0.05
N_BOOTSTRAP = 20000

COLOR_INK = "#0b0b0b"
COLOR_MUTED = "#6b6a65"
COLOR_THEORY = "#1a1a1a"
COLOR_BOOT = "#8a8a8a"
COLOR_AKAIKE = "#c0392b"
COLOR_BOX = "#2166ac"
COLOR_SANSO = "#1b7837"

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
    dataset_dir = Path("data/processed/ucr/noise") / dataset_name
    train_end = get_train_end(dataset_name)
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
        starts = calibration_starts(train_values, window_size)
        rng = np.random.default_rng(0)
        t_star = build_block_bootstrap_null(train_values, starts, model, window_size, N_BOOTSTRAP, rng)
    f_upper = float(stats.f.ppf(1 - ALPHA, window_size, model["reference_df"]))
    bb_upper = float(np.percentile(t_star, 100 * (1 - ALPHA)))
    # Akaike (final prediction error): R_w inflated by (n0 + k) / (n0 - k)
    n0, k = len(train_values), len(model["fixed_params"])
    akaike = stats.f(window_size, n0, scale=(n0 + k) / (n0 - k))
    # Box (1953): heavy-tailed noise -> the window variance varies more, as if
    # the window had fewer points
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        resid = score_window(train_values, model)[1]
    kappa = float(stats.kurtosis(resid[np.isfinite(resid)], fisher=False))
    box = stats.f(2 * window_size / (kappa - 1), 2 * n0 / (kappa - 1))
    # Sanso, Arago & Carrion (2004), as in their corrected ICSS: replace kappa - 1
    # by the long-run variance of z^2 (Bartlett kernel, Newey-West bandwidth),
    # which also covers clustering of large residuals
    omega = long_run_variance(resid[np.isfinite(resid)])
    sanso = stats.f(2 * window_size / omega, 2 * n0 / omega)
    return t_star, model["reference_df"], f_upper, bb_upper, akaike, box, kappa, sanso, omega


def long_run_variance(resid: np.ndarray) -> float:
    """Long-run variance of z_t^2 - 1, z = resid / std(resid); equals kappa - 1
    when the squared residuals are uncorrelated (2 for normal noise)."""
    u = (resid / resid.std()) ** 2 - 1
    T = len(u)
    m = int(np.floor(4 * (T / 100) ** (2 / 9)))
    omega = u @ u / T
    for j in range(1, m + 1):
        omega += 2 * (1 - j / (m + 1)) * (u[j:] @ u[:-j]) / T
    return float(omega)


def plot_panel(ax, dataset_name: str, window_size: int, t_star: np.ndarray, ref_df: int, f_upper: float, bb_upper: float, akaike, box, kappa: float, sanso, omega: float) -> None:
    _style_axis(ax)
    x_max = max(np.percentile(t_star, 99), f_upper, bb_upper) * 1.1
    xs = np.linspace(1e-6, x_max, 400)

    kde = stats.gaussian_kde(t_star)
    ax.fill_between(xs, kde(xs), color=COLOR_BOOT, alpha=0.35, linewidth=0)
    ax.plot(xs, kde(xs), color=COLOR_BOOT, linewidth=1.6, label="Bootstrap empirical null (KDE)")

    pdf = stats.f.pdf(xs, window_size, ref_df)
    ax.plot(xs, pdf, color=COLOR_THEORY, linewidth=1.4, label="Theoretical $F$ pdf")
    # Akaike curve not drawn: on these long training parts it lies on the F curve
    ax.plot(xs, box.pdf(xs), color=COLOR_BOX, linewidth=1.4,
            label=f"Kurtosis-corrected $F$ ($\\kappa$={kappa:.1f}, $n_{{eff}}$={box.args[0]:.1f})")
    ax.plot(xs, sanso.pdf(xs), color=COLOR_SANSO, linewidth=1.4,
            label=f"Sanso-corrected $F$ ($\\omega$={omega:.1f}, $n_{{eff}}$={sanso.args[0]:.1f})")

    ax.axvline(f_upper, color=COLOR_THEORY, linewidth=1.1, linestyle="--", label=f"F-theory limit ($\\alpha$={ALPHA})")
    ax.axvline(box.ppf(1 - ALPHA), color=COLOR_BOX, linewidth=1.1, linestyle=":", label="Kurtosis-corrected limit")
    ax.axvline(sanso.ppf(1 - ALPHA), color=COLOR_SANSO, linewidth=1.1, linestyle=":", label="Sanso-corrected limit")
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

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, (name, window_size) in zip(axes, datasets):
        t_star, ref_df, f_upper, bb_upper, akaike, box, kappa, sanso, omega = panel_data(name, window_size)
        plot_panel(ax, name, window_size, t_star, ref_df, f_upper, bb_upper, akaike, box, kappa, sanso, omega)
        print(f"{name}: F-theory limit={f_upper:.3f}  "
              f"Akaike limit={akaike.ppf(1 - ALPHA):.3f}  Bootstrap limit={bb_upper:.3f}  factor={akaike.kwds['scale']:.4f}  "
              f"kappa={kappa:.2f}  n_eff={box.args[0]:.1f}  kurtosis limit={box.ppf(1 - ALPHA):.3f}  "
              f"omega={omega:.2f}  n_eff={sanso.args[0]:.1f}  Sanso limit={sanso.ppf(1 - ALPHA):.3f}")

    fig.tight_layout()
    output_path = Path("results/simulation/f_check/figures/ucr_null_distributions_corrected.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
