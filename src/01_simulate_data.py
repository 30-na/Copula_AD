"""Simulate clean one-dimensional AR(1) time-series data and plot it.

Outputs:
  - data/simulation/simulated_clean.csv
  - results/simulation/figures/simulated_clean.png
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


def simulate_ar1(
    n_samples: int,
    phi: float,
    noise_std: float,
    seed: int,
) -> pd.DataFrame:
    """Generate one AR(1) signal."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=noise_std, size=n_samples)
    values = np.zeros(n_samples)
    values[0] = noise[0]

    for t in range(1, n_samples):
        values[t] = phi * values[t - 1] + noise[t]

    index = pd.date_range("2026-01-01", periods=n_samples, freq="s", name="time")
    return pd.DataFrame({"value": values}, index=index)


def plot_timeseries(df: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.index, df["value"], color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("value")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate clean AR(1) time-series data.")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--phi", type=float, default=0.8)
    parser.add_argument("--noise-std", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", default="data/simulation/simulated_clean.csv")
    parser.add_argument("--plot", default="results/simulation/figures/simulated_clean.png")
    args = parser.parse_args()

    df = simulate_ar1(
        n_samples=args.n_samples,
        phi=args.phi,
        noise_std=args.noise_std,
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
