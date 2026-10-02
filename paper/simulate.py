"""Generate a synthetic ARMA(p, q) series with an optional variance-burst
anomaly (paper Eq. arma: y_t = c + phi_1 y_{t-1} + ... + zeta_t + theta_1
zeta_{t-1} + ...).

The first n0 points are guaranteed anomaly-free (this is the paper's
training segment length n0); the anomaly, if any, is only ever placed at or
after n0.

Outputs (under --output-dir, default data/simulation):
  - simulated_clean.csv          the ARMA series, no anomaly
  - simulated_with_anomaly.csv   the same series with the anomaly injected
  - simulated_anomaly_labels.csv one is_anomaly flag per point
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import ArmaProcess


def parse_coeffs(text: str) -> list[float]:
    """Parse a comma-separated coefficient string from the command line.

    Args:
        text: e.g. "0.5,0.2", or "" for no coefficients.

    Returns:
        list of floats, e.g. [0.5, 0.2]; [] if text is empty.
    """
    text = text.strip()
    return [float(v) for v in text.split(",")] if text else []


def simulate_arma(n_total: int, phi: list[float], theta: list[float], sigma: float, seed: int) -> np.ndarray:
    """Generate one ARMA(p, q) series (paper Eq. arma), y_t = c + phi_1
    y_{t-1} + ... + zeta_t + theta_1 zeta_{t-1} + ..., zeta_t ~ N(0, sigma^2).

    Args:
        n_total: number of points to generate.
        phi: AR coefficients [phi_1, ..., phi_p] ([] for no AR terms).
        theta: MA coefficients [theta_1, ..., theta_q] ([] for no MA terms).
        sigma: innovation standard deviation.
        seed: random seed for the innovations.

    Returns:
        1D array of length n_total: the simulated series.
    """
    # statsmodels' polynomial convention: ar = [1, -phi_1, ...], ma = [1, theta_1, ...].
    ar = np.r_[1.0, -np.asarray(phi, dtype=float)] if phi else np.array([1.0])
    ma = np.r_[1.0, np.asarray(theta, dtype=float)] if theta else np.array([1.0])

    process = ArmaProcess(ar, ma)
    if not process.isstationary:
        raise ValueError(f"phi={phi} is not stationary; choose different AR coefficients.")
    if not process.isinvertible:
        raise ValueError(f"theta={theta} is not invertible; choose different MA coefficients.")

    rng = np.random.default_rng(seed)
    return process.generate_sample(
        nsample=n_total, scale=sigma, distrvs=lambda size: rng.standard_normal(size=size), burnin=500
    )


def add_variance_burst(
    values: np.ndarray, start: int, duration: int, strength: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Inflate the noise over [start, start+duration) by adding extra
    N(0, (strength*std)^2) noise on top of the existing values.

    Args:
        values: the clean series, 1D array of length n_total.
        start: index where the anomaly window begins.
        duration: number of points the anomaly window covers.
        strength: extra noise std, as a multiple of the series' own std.
        seed: random seed for the extra noise.

    Returns:
        (anomaly_values, is_anomaly):
            anomaly_values: values with the burst added, same length as
                `values`.
            is_anomaly: 0/1 flag per point, 1 only inside
                [start, start+duration).
    """
    rng = np.random.default_rng(seed)
    anomaly_values = values.copy()
    baseline_std = float(values.std())
    burst = rng.normal(loc=0.0, scale=strength * baseline_std, size=duration)
    anomaly_values[start : start + duration] += burst

    is_anomaly = np.zeros(len(values), dtype=int)
    is_anomaly[start : start + duration] = 1
    return anomaly_values, is_anomaly


def main() -> None:
    """Read the CLI flags, run simulate_arma + add_variance_burst, and
    write the three output CSVs to --output-dir. Takes no arguments
    (reads sys.argv via argparse); returns nothing.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-total", type=int, default=9000, help="Total series length.")
    parser.add_argument("--n0", type=int, default=3600, help="Anomaly-free segment length (paper's n0).")
    parser.add_argument("--phi", type=str, default="0.6", help="Comma-separated AR coefficients.")
    parser.add_argument("--theta", type=str, default="0.3", help="Comma-separated MA coefficients.")
    parser.add_argument("--sigma", type=float, default=1.0, help="Innovation standard deviation.")
    parser.add_argument("--anomaly-duration", type=int, default=600)
    parser.add_argument("--anomaly-strength", type=float, default=1.0)
    parser.add_argument(
        "--anomaly-start", type=int, default=None,
        help="Defaults to centering the anomaly in [n0, n_total).",
    )
    parser.add_argument("--seed", type=int, default=123, help="Seed for the ARMA series itself.")
    parser.add_argument("--anomaly-seed", type=int, default=456, help="Seed for the anomaly noise burst.")
    parser.add_argument("--output-dir", default="data/simulation")
    args = parser.parse_args()

    if not 0 < args.n0 < args.n_total:
        raise ValueError("n0 must be strictly between 0 and n_total.")

    post_n0_length = args.n_total - args.n0
    if args.anomaly_duration > post_n0_length:
        raise ValueError(
            f"anomaly_duration ({args.anomaly_duration}) is longer than the test segment "
            f"(n_total - n0 = {post_n0_length}); lower --anomaly-duration or raise --n-total."
        )

    anomaly_start = args.anomaly_start
    if anomaly_start is None:
        anomaly_start = args.n0 + (post_n0_length - args.anomaly_duration) // 2
    if anomaly_start < args.n0:
        raise ValueError(f"anomaly_start ({anomaly_start}) must be >= n0 ({args.n0}).")
    if anomaly_start + args.anomaly_duration > args.n_total:
        raise ValueError("anomaly_start + anomaly_duration exceeds n_total.")

    phi = parse_coeffs(args.phi)
    theta = parse_coeffs(args.theta)
    clean_values = simulate_arma(args.n_total, phi, theta, args.sigma, args.seed)
    anomaly_values, is_anomaly = add_variance_burst(
        clean_values, anomaly_start, args.anomaly_duration, args.anomaly_strength, args.anomaly_seed
    )

    index = pd.RangeIndex(args.n_total, name="t")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"value": clean_values}, index=index).to_csv(output_dir / "simulated_clean.csv")
    pd.DataFrame({"value": anomaly_values}, index=index).to_csv(output_dir / "simulated_with_anomaly.csv")
    pd.DataFrame({"is_anomaly": is_anomaly}, index=index).to_csv(output_dir / "simulated_anomaly_labels.csv")

    print(f"phi={phi}, theta={theta}, sigma={args.sigma}")
    print(f"n_total={args.n_total}, n0={args.n0}")
    print(f"anomaly: [{anomaly_start}, {anomaly_start + args.anomaly_duration}), strength={args.anomaly_strength}")
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
