"""Sensitivity analysis: does Algorithm 1 hold its nominal Type I error
when the Gaussian-innovations assumption (A2) is violated?

Overlays the P value plot (Davidson & MacKinnon 1998) for the Gaussian
baseline against several Student-t innovation distributions at decreasing
degrees of freedom (heavier tails = bigger violation), all on one figure --
following Davidson & MacKinnon's own comparative use of the plot: "P value
plots allow us to distinguish at a glance among test statistics that
systematically over-reject... under-reject... or reject about the right
proportion of the time." Here the thing being compared is not different
test statistics but the SAME test under different innovation distributions.

Each Student-t distribution is rescaled to unit variance before use, so
every distribution has the SAME true innovation variance -- the only thing
that changes across curves is tail shape/kurtosis (Box 1953), not overall
noise level.

Type I error only (no anomaly injected, no power) -- this is purely a
calibration/robustness check.

Outputs, under --output-dir (default results/sensitivity_distribution):
  - summary.csv         one row per distribution: empirical Type I error
                         with exact (Clopper-Pearson) 95% confidence interval
  - figures/summary.png one P value plot, one curve per distribution
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.arima_process import ArmaProcess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from algorithm import run_algorithm
from simulate import parse_coeffs

COLOR_INK = "#0b0b0b"
COLOR_MUTED = "#6b6a65"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 10.5,
        "axes.labelsize": 10.5,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "axes.linewidth": 0.8,
    }
)


def _style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_MUTED)
    ax.spines["bottom"].set_color(COLOR_MUTED)
    ax.tick_params(colors=COLOR_MUTED)


def simulate_arma_innovations(
    n_total: int, phi: list[float], theta: list[float], sigma: float, df: float | None, seed: int
) -> np.ndarray:
    """Same ARMA(p, q) generator as simulate.simulate_arma, but the
    innovations can optionally come from a Student-t distribution instead
    of Gaussian.

    Args:
        df: degrees of freedom for the Student-t innovations. None means
            Gaussian (the paper's assumption A2, satisfied exactly).
            Otherwise the t-distribution is rescaled to unit variance
            first (t has variance df/(df-2) for df>2), so sigma controls
            the true innovation variance the same way regardless of df --
            only the tail shape changes.

    Returns:
        1D array of length n_total.
    """
    ar = np.r_[1.0, -np.asarray(phi, dtype=float)] if phi else np.array([1.0])
    ma = np.r_[1.0, np.asarray(theta, dtype=float)] if theta else np.array([1.0])
    process = ArmaProcess(ar, ma)
    if not process.isstationary:
        raise ValueError(f"phi={phi} is not stationary.")
    if not process.isinvertible:
        raise ValueError(f"theta={theta} is not invertible.")

    rng = np.random.default_rng(seed)
    if df is None:
        distrvs = lambda size: rng.standard_normal(size=size)
    else:
        unit_scale = np.sqrt((df - 2) / df)  # rescale Student-t to unit variance
        distrvs = lambda size: rng.standard_t(df, size=size) * unit_scale

    return process.generate_sample(nsample=n_total, scale=sigma, distrvs=distrvs, burnin=500)


def run_one_replication(
    n_total: int, n0: int, phi: list[float], theta: list[float], sigma: float, df: float | None,
    window_size: int, alpha: float, max_p: int, max_q: int, seed: int,
) -> pd.DataFrame | None:
    """One replication under H0 only (no anomaly): fresh data, fresh fit,
    fresh scoring. Returns a DataFrame with column p_value (one row per
    test window), or None if the ARMA fit failed for this draw.
    """
    values = simulate_arma_innovations(n_total, phi, theta, sigma, df, seed)
    train_values = values[:n0]
    test_values = values[n0:]
    try:
        results, _, _ = run_algorithm(train_values, test_values, window_size, alpha, max_p, max_q)
    except Exception:
        return None
    return results[["p_value"]]


def exact_ci(successes: int, n: int) -> tuple[float, float]:
    """Clopper-Pearson exact 95% confidence interval for a proportion."""
    if n == 0:
        return (0.0, 1.0)
    low = stats.beta.ppf(0.025, successes, n - successes + 1) if successes > 0 else 0.0
    high = stats.beta.ppf(0.975, successes + 1, n - successes) if successes < n else 1.0
    return float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-total", type=int, default=2000)
    parser.add_argument("--n0", type=int, default=600, help="Anomaly-free training length (paper's n0).")
    parser.add_argument("--phi", type=str, default="0.6")
    parser.add_argument("--theta", type=str, default="0.3")
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--max-p", type=int, default=3)
    parser.add_argument("--max-q", type=int, default=3)
    parser.add_argument("--n-reps", type=int, default=50, help="Replications per distribution.")
    parser.add_argument("--t-df", type=str, default="10,5,3", help="Comma-separated Student-t degrees of freedom to test.")
    parser.add_argument("--seed", type=int, default=123, help="Base seed; replication i uses seed + i.")
    parser.add_argument("--output-dir", default="results/sensitivity_distribution")
    args = parser.parse_args()

    phi = parse_coeffs(args.phi)
    theta = parse_coeffs(args.theta)
    t_dfs = [float(x) for x in args.t_df.split(",") if x.strip()]

    # (label, df) -- df=None is the Gaussian baseline, always first/solid.
    distributions: list[tuple[str, float | None]] = [("Gaussian", None)] + [
        (f"Student-t (df={int(df) if df == int(df) else df})", df) for df in t_dfs
    ]
    linestyles = ["-", "--", "-.", ":"] + ["-"] * max(0, len(distributions) - 4)

    summary_rows = []
    curves: dict[str, np.ndarray] = {}

    for label, df in distributions:
        print(f"Running {args.n_reps} replications for {label}...")
        all_p = []
        n_failed = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(args.n_reps):
                rep = run_one_replication(
                    args.n_total, args.n0, phi, theta, args.sigma, df,
                    args.window_size, args.alpha, args.max_p, args.max_q,
                    seed=args.seed + i,
                )
                if rep is None:
                    n_failed += 1
                    continue
                all_p.append(rep["p_value"].to_numpy())
        if n_failed:
            print(f"  Warning: {n_failed}/{args.n_reps} replications failed to fit and were skipped.")

        pooled_p = np.concatenate(all_p)
        curves[label] = pooled_p

        n_normal = len(pooled_p)
        n_false_positive = int((pooled_p < args.alpha).sum())
        type1_rate = n_false_positive / n_normal
        ci_low, ci_high = exact_ci(n_false_positive, n_normal)
        print(f"  Type I error: {type1_rate:.2%} [{ci_low:.2%}, {ci_high:.2%}], pooled over {n_normal} windows")

        summary_rows.append(
            {
                "distribution": label, "t_df": df if df is not None else "",
                "n_reps": args.n_reps - n_failed, "n_normal_windows": n_normal,
                "n_false_positive": n_false_positive, "type1_rate": type1_rate,
                "type1_ci_low": ci_low, "type1_ci_high": ci_high,
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            pd.DataFrame(summary_rows).to_csv(output_dir / "summary.csv", index=False)
            break
        except PermissionError:
            if attempt == 2:
                raise
            time.sleep(0.5)

    # P value plot, one curve per distribution (Davidson & MacKinnon 1998),
    # same canvas-width-to-LaTeX-width convention as monte_carlo.py's
    # summary.png (5.6in canvas at width=0.7\linewidth) so fonts match.
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    _style_axis(ax)
    ax.plot([0, 1], [0, 1], color=COLOR_MUTED, linewidth=1.0, linestyle=":", label="Uniform(0,1)")
    for (label, _), linestyle in zip(distributions, linestyles):
        sorted_p = np.sort(curves[label])
        ecdf = np.arange(1, len(sorted_p) + 1) / len(sorted_p)
        ax.plot(sorted_p, ecdf, color=COLOR_INK, linewidth=1.4, linestyle=linestyle, label=label)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("alpha (nominal significance level)")
    ax.set_ylabel("Fraction of normal windows flagged")
    ax.set_title("P Value Plot: Sensitivity to Non-Normal Innovations", loc="left", fontsize=10.5, color=COLOR_INK)
    ax.legend(loc="center left", bbox_to_anchor=(1.06, 0.5), fontsize=8.5, frameon=False)
    fig.subplots_adjust(left=0.13, right=0.58, top=0.90, bottom=0.14)

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            fig.savefig(figures_dir / "summary.png", dpi=220)
            break
        except OSError:
            if attempt == 2:
                raise
            time.sleep(0.5)
    plt.close(fig)

    print(f"Results saved under {output_dir}")


if __name__ == "__main__":
    main()
