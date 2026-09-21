"""Does the test really hold its nominal level, and when does it stop?

Script 09 runs Algorithm 1 once. One run is one draw, so it cannot tell you
whether the test keeps its promise. This script runs it many times on data
with NO ANOMALY, so every window is a normal window and every flag is a
false alarm. If the theorem is right, flagging should happen alpha of the
time.

It then breaks three of the theorem's assumptions on purpose:

  --grid-df       Gaussian innovations (assumption A2). Replaced by Student
                  t, which has heavier tails. Smaller df = heavier.
  --grid-window   Window length n.
  --grid-n0       Training length n0.

Example:
  python src/10_null_calibration_monte_carlo.py --n-seeds 20 \
      --grid-df "3,5,10,30,normal" --grid-window "5,10,25,50,100"

Outputs, under --output-dir:
  cells.csv                  one row per setting
  null_windows.csv           one row per window
  figures/sensitivity.png
"""

import argparse
import importlib.util
import os
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Script 09 holds the algorithm itself. Its filename starts with a digit, so
# a normal `import` will not work and we load it by path instead. Doing this
# rather than copying the code keeps ONE version of the algorithm.
_spec = importlib.util.spec_from_file_location(
    "paper_algorithm", Path(__file__).resolve().parent / "09_run_paper_algorithm_simulation.py"
)
algorithm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(algorithm)

GAUSSIAN = "normal"


# ---------------------------------------------------------------------------
# Simulate one series. df=None gives Gaussian innovations, a number gives
# Student t with that many degrees of freedom.
# ---------------------------------------------------------------------------

def simulate_series(n_samples, phi, theta, noise_std, df, seed):
    ar = np.r_[1.0, -np.array(phi)] if phi else np.array([1.0])
    ma = np.r_[1.0, np.array(theta)] if theta else np.array([1.0])
    process = ArmaProcess(ar, ma)

    rng = np.random.default_rng(seed)
    if df is None:
        def draw(size):
            return rng.standard_normal(size)
    else:
        # A t with df degrees of freedom has variance df/(df-2). Divide it
        # out so the innovation variance stays 1 and ONLY the tail weight
        # changes between settings.
        scale = np.sqrt(df / (df - 2.0))

        def draw(size):
            return rng.standard_t(df, size) / scale

    values = process.generate_sample(
        nsample=n_samples, scale=noise_std, distrvs=draw, burnin=500
    )
    index = pd.date_range("2026-01-01", periods=n_samples, freq="s")
    return pd.Series(values, index=index)


# ---------------------------------------------------------------------------
# Fit the reference model on the training segment, at a known order.
# Returns the same dictionary shape script 09 uses.
# ---------------------------------------------------------------------------

def fit_reference_model(train_segment, order):
    values = train_segment.to_numpy(dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ARIMA(values, order=order).fit()

    names = list(fit.param_names)
    params = np.array(fit.params, dtype=float)
    sigma_index = names.index("sigma2")

    return {
        "order": order,
        "param_names": [name for name in names if name != "sigma2"],
        "fixed_params": np.delete(params, sigma_index),
        "reference_sigma2": float(params[sigma_index]),
        "reference_df": len(values),
    }


# ---------------------------------------------------------------------------
# One run: simulate, fit, score every window. Returns one row per window.
# ---------------------------------------------------------------------------

def run_once(setting):
    df, window_size, n0, seed, options = setting
    innovation_df = None if df == GAUSSIAN else float(df)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        series = simulate_series(
            n_samples=n0 + options["test_length"],
            phi=options["phi"],
            theta=options["theta"],
            noise_std=options["noise_std"],
            df=innovation_df,
            seed=seed,
        )
        train_segment = series.iloc[:n0]
        test_segment = series.iloc[n0:]

        try:
            model = fit_reference_model(train_segment, options["order"])
        except Exception:
            return None

        rows = []
        for window in algorithm.make_nonoverlapping_windows(test_segment, window_size):
            sigma2_w, _ = algorithm.fixed_dynamics_window_variance(window, model)
            ratio, p_value = algorithm.variance_ratio_two_sided_pvalue(
                sigma2_w, window_size, model
            )
            rows.append(
                {
                    "df": df,
                    "n0": n0,
                    "window_size": window_size,
                    "seed": seed,
                    "variance_ratio": ratio,
                    "p_value": p_value,
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Turn all the windows of one setting into one summary row.
# ---------------------------------------------------------------------------

def summarize(cell, alphas):
    row = {
        "df": cell["df"].iloc[0],
        "n0": int(cell["n0"].iloc[0]),
        "window_size": int(cell["window_size"].iloc[0]),
        "n_seeds": cell["seed"].nunique(),
        "n_windows": len(cell),
        "mean_variance_ratio": cell["variance_ratio"].mean(),
    }

    for alpha in alphas:
        name = f"size_{alpha:g}"
        row[name] = (cell["p_value"] < alpha).mean()

        # For the error bars, work out each run's own false alarm rate first,
        # then see how much those rates differ from run to run. Windows inside
        # one run share a single sigma0^2, so pooling them all together as if
        # they were independent would make the error bars far too small.
        per_run = cell.groupby("seed")["p_value"].apply(lambda s: (s < alpha).mean())
        margin = 1.96 * per_run.std(ddof=1) / np.sqrt(len(per_run))
        row[name + "_lo"] = per_run.mean() - margin
        row[name + "_hi"] = per_run.mean() + margin

    return row


# ---------------------------------------------------------------------------
# Figure. The y axis is "how many times more false alarms than asked for",
# so 1 means correct. Using the ratio instead of the raw rate puts all the
# alpha panels on the same scale, which makes them comparable.
# ---------------------------------------------------------------------------

# Heavier tails get a darker blue. Gaussian is the case where the assumption
# holds, so it is gray to set it apart from the three violations.
COLORS = {3.0: "#104281", 5.0: "#2a78d6", 10.0: "#5598e7", 30.0: "#86b6ef"}
GRAY = "#52514e"


def plot_sensitivity(cells, alphas, output_path):
    # Plot against whichever of window_size / n0 was actually varied.
    if cells["window_size"].nunique() > 1:
        x_column, x_label = "window_size", "Window length $n$"
    else:
        x_column, x_label = "n0", "Training length $n_0$"

    # Heaviest tails first, Gaussian last, so the legend matches the lines.
    levels = sorted(cells["df"].unique(), key=lambda v: 99 if v == GAUSSIAN else float(v))

    fig, axes = plt.subplots(1, len(alphas), figsize=(4.5 * len(alphas), 4.3), sharey=True)
    if len(alphas) == 1:
        axes = [axes]

    for ax, alpha in zip(axes, alphas):
        name = f"size_{alpha:g}"

        # Bradley's (1978) acceptable range, 0.5 to 1.5 times alpha.
        ax.axhspan(0.5, 1.5, color="#d8e6f9", alpha=0.55, linewidth=0)
        ax.axhline(1.0, color="black", linewidth=1.1, linestyle="--")

        for level in levels:
            part = cells[cells["df"] == level].sort_values(x_column)
            color = GRAY if level == GAUSSIAN else COLORS[float(level)]
            label = "Gaussian" if level == GAUSSIAN else f"t({float(level):g})"
            ax.plot(part[x_column], part[name] / alpha, marker="o", ms=5.5,
                    linewidth=2.0, color=color, label=label)
            ax.fill_between(part[x_column], part[name + "_lo"] / alpha,
                            part[name + "_hi"] / alpha, color=color, alpha=0.18, linewidth=0)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(sorted(cells[x_column].unique()))
        ax.set_yticks([0.5, 1, 2, 5, 10, 20, 40])
        ax.set_xticklabels([str(v) for v in sorted(cells[x_column].unique())])
        ax.set_yticklabels(["0.5x", "1x", "2x", "5x", "10x", "20x", "40x"])
        ax.minorticks_off()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel(x_label)
        ax.set_title(f"alpha = {alpha:g}", loc="left")

    axes[0].set_ylabel("False alarms, compared to what was asked for")
    axes[-1].legend(frameon=False, loc="lower right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--test-length", type=int, default=10000)
    parser.add_argument("--grid-df", default="normal")
    parser.add_argument("--grid-window", default="10")
    parser.add_argument("--grid-n0", default="1000")
    parser.add_argument("--phi", default="0.8")
    parser.add_argument("--theta", default="0.6")
    parser.add_argument("--noise-std", type=float, default=1.0)
    parser.add_argument("--alphas", default="0.01,0.05,0.10")
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--output-dir", default="results/null_calibration")
    args = parser.parse_args()

    phi = [float(v) for v in args.phi.split(",") if v.strip()]
    theta = [float(v) for v in args.theta.split(",") if v.strip()]
    alphas = sorted(float(v) for v in args.alphas.split(","))

    df_levels = [v if v == GAUSSIAN else float(v) for v in args.grid_df.split(",")]
    window_levels = [int(v) for v in args.grid_window.split(",")]
    n0_levels = [int(v) for v in args.grid_n0.split(",")]

    options = {
        "test_length": args.test_length,
        "phi": phi,
        "theta": theta,
        "noise_std": args.noise_std,
        # The order is fixed at the truth, not chosen by AIC. If AIC could
        # choose, a bad result might come from the heavy tails OR from
        # picking the wrong order, and you could not tell which.
        "order": (len(phi), 0, len(theta)),
    }

    settings = []
    for df in df_levels:
        for window_size in window_levels:
            for n0 in n0_levels:
                for seed in range(10000, 10000 + args.n_seeds):
                    settings.append((df, window_size, n0, seed, options))

    n_cells = len(df_levels) * len(window_levels) * len(n0_levels)
    print(f"{n_cells} settings x {args.n_seeds} seeds = {len(settings)} runs")

    with Pool(args.jobs) as pool:
        results = pool.map(run_once, settings)

    windows = pd.concat([r for r in results if r is not None], ignore_index=True)

    cells = []
    for _, cell in windows.groupby(["df", "n0", "window_size"], sort=False):
        cells.append(summarize(cell, alphas))
    cells = pd.DataFrame(cells).sort_values(["df", "n0", "window_size"]).reset_index(drop=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    windows.to_csv(output_dir / "null_windows.csv", index=False)
    cells.to_csv(output_dir / "cells.csv", index=False)
    if n_cells > 1:
        plot_sensitivity(cells, alphas, output_dir / "figures" / "sensitivity.png")

    columns = ["df", "n0", "window_size", "n_windows"] + [f"size_{a:g}" for a in alphas]
    print()
    print(cells[columns].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
