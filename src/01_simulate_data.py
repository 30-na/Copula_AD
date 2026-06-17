"""Simulate clean multichannel AR(1) time-series data and plot it.

Outputs:
  - data/simulated_clean.csv
  - results/figures/simulated_clean.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def simulate_multichannel_ar1(
    n_samples: int,
    n_channels: int,
    phi: float,
    noise_std: float,
    correlation: float,
    seed: int,
) -> pd.DataFrame:
    """Generate correlated multichannel AR(1) signals."""
    rng = np.random.default_rng(seed)

    covariance = np.full((n_channels, n_channels), correlation * noise_std**2)
    np.fill_diagonal(covariance, noise_std**2)

    noise = rng.multivariate_normal(
        mean=np.zeros(n_channels),
        cov=covariance,
        size=n_samples,
    )

    values = np.zeros((n_samples, n_channels))
    values[0] = noise[0]

    for t in range(1, n_samples):
        values[t] = phi * values[t - 1] + noise[t]

    columns = [f"CH{i + 1}" for i in range(n_channels)]
    index = pd.date_range("2026-01-01", periods=n_samples, freq="s", name="time")
    return pd.DataFrame(values, index=index, columns=columns)


def plot_timeseries(df: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, axes = plt.subplots(
        nrows=len(df.columns),
        ncols=1,
        figsize=(12, 2.2 * len(df.columns)),
        sharex=True,
    )

    if len(df.columns) == 1:
        axes = [axes]

    for ax, column in zip(axes, df.columns):
        ax.plot(df.index, df[column], color="black", linewidth=0.8)
        ax.set_ylabel(column)
        ax.grid(True, linestyle="--", alpha=0.3)

    axes[0].set_title(title)
    axes[-1].set_xlabel("Time")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate clean AR(1) time-series data.")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--n-channels", type=int, default=3)
    parser.add_argument("--phi", type=float, default=0.8)
    parser.add_argument("--noise-std", type=float, default=1.0)
    parser.add_argument("--correlation", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", default="data/simulated_clean.csv")
    parser.add_argument("--plot", default="results/figures/simulated_clean.png")
    args = parser.parse_args()

    df = simulate_multichannel_ar1(
        n_samples=args.n_samples,
        n_channels=args.n_channels,
        phi=args.phi,
        noise_std=args.noise_std,
        correlation=args.correlation,
        seed=args.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)

    plot_timeseries(df, Path(args.plot), "Clean Simulated AR(1) Time Series")
    print(f"Saved clean data to {output_path}")
    print(f"Saved plot to {args.plot}")


if __name__ == "__main__":
    main()
