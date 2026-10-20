# SPDX-FileCopyrightText: 2022 German Aerospace Center (DLR e.V.), Ferdinand Rewicki
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.lib.stride_tricks as stride_tricks
import numpy.typing as npt


def z_normalize_timeseries(timeseries: npt.NDArray) -> npt.NDArray:
    """z-normalize timeseries

    :param timeseries:
    :return: z-normlizesd timeseries
    """
    mu = np.nanmean(timeseries)
    sigma = np.nanstd(timeseries)
    return (timeseries - mu) / sigma


def sanitize_flat_subsequences(timeseries: npt.NDArray, w: int) -> npt.NDArray:
    """Add linear trend if time series contains subsequences with near-constant values.

    MERLIN may report false positive anomalies in subsequences with near constant
    values. A sign that the problem exist is a spike near zero when plotting the
    histogram of movstd. Adding a large linear trend will solve this issue.

    :param timeseries: The z-normalized (multi/uni)-variate time series given as
    2D-numpy array of shape `(d, n)` with `d` being the number of dimensions of
    the timeseries and `n` it's length.

    :param w: The window-size. Typical the minimal subsequence length when running
    MERLIN.

    :return: The modified time series if near-constant subsequences are found, the
    original time series otherwise.
    """

    movwin = stride_tricks.sliding_window_view(timeseries, window_shape=w, axis=1)
    movsigma = np.nanstd(movwin, axis=2)

    numBins = 100
    edges = np.linspace(0, np.max(movsigma), numBins)
    h = np.histogram(movsigma, edges)
    if np.sum(h[0][0:5] > 0):
        return timeseries + range(0, timeseries.shape[1])

    return timeseries


def plot(
    ts: npt.NDArray,
    discords: List[int],
    lengths: List[int],
    anomalies: List[Tuple[int, int]] = None,
    tol: int = 0,
    name: str = "",
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot results.

    Create a two-fold plot with the upper one containging the original
    time series with the ground-truth label(s) shown. The second plot
    shows the detected discords plotted against their subsequence length.

    :param ts: The (univariate) timeseries given as numpy array.
    :param discords: The detected discord indices.
    :param lengths: The subsequence lengths.
    :param anomalies: The ground Truth anomalies given as `[(start, end)]`
    :param tol: Tolerance (in time-steps) to mark detected anomalies as true positives
    :param name: The name of the timeseries. Will be used as plot title.

    :return: Matplotlib's Figure and Axes objects.
    """
    fig, ax = plt.subplots(nrows=2, figsize=(15, 6), sharex=True)
    ax[0].plot(ts)
    discords = np.array(discords)
    lengths = np.array(lengths)
    hits = np.zeros_like(discords, dtype="bool")
    if anomalies is not None:
        for a in anomalies:
            ax[0].axvspan(a[0], a[1], color="green", alpha=0.5)
            ax[1].axvspan(a[0] - tol, a[1] + tol, color="green", alpha=0.3)
            ax[1].axvspan(a[0], a[1], color="green", alpha=0.5)
            hits = hits | ((discords >= a[0] - tol) & (discords <= a[1] + tol))
        ax[1].scatter(
            discords[hits],
            lengths[hits],
            color="red",
            marker="o",
            label="True Positives",
        )
        ax[1].scatter(
            discords[np.invert(hits)],
            lengths[np.invert(hits)],
            color="black",
            marker="x",
            alpha=0.7,
            label="False Positives",
        )
    else:
        ax[1].scatter(discords, lengths, color="red", marker="o")
    ax[1].set_ylabel("Discord Length")
    ax[1].set_xlabel("t")
    ax[1].legend()
    fig.suptitle(name)

    return fig, ax
