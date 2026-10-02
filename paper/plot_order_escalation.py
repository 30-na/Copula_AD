"""Visualize the order-escalation experiment (test_order_escalation.py)
against the pipeline's actual AIC-based method (algorithm.fit_reference_model,
bounded at p,q<=3), for seizures 1-3, channel 1, window=100 config.

3 columns (one per seizure), 3 rows:
  1. AIC vs AR(p) order -- dashed line marks the AIC-method's chosen AIC
     (which may include MA terms, so it's a value reference, not a point
     on this AR-only curve)
  2. In-sample Ljung-Box p-value vs order (log scale, 0.05 threshold line)
  3. Out-of-sample Ljung-Box p-value vs order (log scale, 0.05 threshold line)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from algorithm import fit_reference_model
from test_order_escalation import CONFIG, N_TRAIN, WINDOW_SIZE, MAX_ORDER, fit_fixed_order, out_of_sample_lb_p

COLOR_INK = "#0b0b0b"
COLOR_MUTED = "#6b6a65"
COLOR_LIMIT = "#8a8a8a"
COLOR_AIC_METHOD = "#b0462f"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 9.5,
        "axes.labelsize": 9.5,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "axes.linewidth": 0.8,
    }
)


def _style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_MUTED)
    ax.spines["bottom"].set_color(COLOR_MUTED)
    ax.tick_params(colors=COLOR_MUTED)


def main() -> None:
    seizures = [1, 2, 3]
    fig, axes = plt.subplots(3, 3, figsize=(13, 9))

    for col, seizure_number in enumerate(seizures):
        values = pd.read_csv(f"data/mice/{CONFIG}/timeseries_seizure{seizure_number}.csv")["value"].to_numpy(float)
        train_values = values[:N_TRAIN]
        test_values = values[N_TRAIN:]

        baseline = fit_reference_model(train_values, 3, 3)

        orders, aics, in_sample_ps, oos_ps = [], [], [], []
        for p in range(1, MAX_ORDER + 1):
            model = fit_fixed_order(train_values, p)
            if model is None:
                continue
            oos_p = out_of_sample_lb_p(model, test_values, WINDOW_SIZE)
            orders.append(p)
            aics.append(model["aic"])
            in_sample_ps.append(max(model["in_sample_lb_p"], 1e-6))
            oos_ps.append(max(oos_p, 1e-6))

        # Row 1: AIC
        ax = axes[0, col]
        _style_axis(ax)
        ax.plot(orders, aics, color=COLOR_INK, marker="o", markersize=3, linewidth=1.0)
        ax.axhline(baseline["aic"], color=COLOR_AIC_METHOD, linewidth=1.2, linestyle="--",
                   label=f"AIC-method order={baseline['order']}\nAIC={baseline['aic']:.1f}")
        ax.set_title(f"Seizure {seizure_number}\nAIC vs AR(p) order", color=COLOR_INK)
        ax.legend(fontsize=7, frameon=False, loc="upper right")
        if col == 0:
            ax.set_ylabel("AIC")

        # Row 2: in-sample Ljung-Box p
        ax = axes[1, col]
        _style_axis(ax)
        ax.plot(orders, in_sample_ps, color=COLOR_INK, marker="o", markersize=3, linewidth=1.0)
        ax.axhline(0.05, color=COLOR_LIMIT, linewidth=1.0, linestyle="--")
        ax.set_yscale("log")
        ax.set_title("In-sample Ljung-Box p (log scale)", fontsize=9.0)
        if col == 0:
            ax.set_ylabel("p-value")

        # Row 3: out-of-sample Ljung-Box p
        ax = axes[2, col]
        _style_axis(ax)
        ax.plot(orders, oos_ps, color=COLOR_INK, marker="o", markersize=3, linewidth=1.0)
        ax.axhline(0.05, color=COLOR_LIMIT, linewidth=1.0, linestyle="--")
        ax.set_yscale("log")
        ax.set_ylim(1e-7, 1.5)
        ax.set_title("Out-of-sample Ljung-Box p (log scale)", fontsize=9.0)
        ax.set_xlabel("AR order p (q=0)")
        if col == 0:
            ax.set_ylabel("p-value")

    fig.suptitle("Does increasing model order fix residual autocorrelation? (seizures 1-3, channel 1, window=100)",
                 fontsize=12, color=COLOR_INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path = "results/mice/figs/order_escalation_vs_aic.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
