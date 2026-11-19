"""Simulate clean one-dimensional ARMA(p, q) time-series data and plot it.

Both the autoregressive coefficients (phi) and the moving average
coefficients (theta) are configurable from the command line, so different
ARMA(p, q) combinations can be tried (e.g. --phi 0.8 for AR(1), or
--phi 0.5,0.2 --theta 0.4 for an ARMA(2,1) process).

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
from statsmodels.tsa.arima_process import ArmaProcess

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_coeffs(text: str) -> list[float]:
    """Parse a comma-separated list of coefficients, e.g. "0.5,0.2" -> [0.5, 0.2]."""
    text = text.strip()
    if not text:
        return []
    return [float(value) for value in text.split(",")]


def simulate_arma(
    n_samples: int,
    phi: list[float],
    theta: list[float],
    noise_std: float,
    seed: int,
) -> pd.DataFrame:
    """Generate one ARMA(p, q) signal, y_t = phi_1 y_{t-1} + ... + zeta_t +
    theta_1 zeta_{t-1} + ..., using the same phi/theta sign convention as
    statsmodels' own ARIMA model (so a model fit back to this data recovers
    phi and theta directly, with no sign flip).
    """
    # statsmodels' polynomial convention: ar = [1, -phi_1, -phi_2, ...],
    # ma = [1, theta_1, theta_2, ...].
    ar = np.r_[1.0, -np.asarray(phi, dtype=float)] if phi else np.array([1.0])
    ma = np.r_[1.0, np.asarray(theta, dtype=float)] if theta else np.array([1.0])

    process = ArmaProcess(ar, ma)
    if not process.isstationary:
        raise ValueError(f"phi={phi} is not stationary; choose different AR coefficients.")
    if not process.isinvertible:
        raise ValueError(f"theta={theta} is not invertible; choose different MA coefficients.")

    rng = np.random.default_rng(seed)
    values = process.generate_sample(
        nsample=n_samples,
        scale=noise_std,
        distrvs=lambda size: rng.standard_normal(size=size),
        burnin=500,
    )

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
    parser = argparse.ArgumentParser(description="Simulate clean ARMA(p, q) time-series data.")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument(
        "--phi", type=str, default="0.8",
        help="Comma-separated AR coefficients, e.g. '0.8' or '0.5,0.2'. Empty string for no AR terms.",
    )
    parser.add_argument(
        "--theta", type=str, default="",
        help="Comma-separated MA coefficients, e.g. '0.4' or '0.4,0.3'. Empty string for no MA terms.",
    )
    parser.add_argument("--noise-std", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", default="data/simulation/simulated_clean.csv")
    parser.add_argument("--plot", default="results/simulation/figures/simulated_clean.png")
    args = parser.parse_args()

    phi = parse_coeffs(args.phi)
    theta = parse_coeffs(args.theta)

    df = simulate_arma(
        n_samples=args.n_samples,
        phi=phi,
        theta=theta,
        noise_std=args.noise_std,
        seed=args.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)

    title = f"Clean Simulated ARMA({len(phi)},{len(theta)}) Time Series"
    plot_timeseries(df, Path(args.plot), title)
    print(f"phi={phi}, theta={theta}, noise_std={args.noise_std}")
    print(f"Saved clean data to {output_path}")
    print(f"Saved plot to {args.plot}")


if __name__ == "__main__":
    main()
