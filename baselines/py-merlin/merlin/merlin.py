# SPDX-FileCopyrightText: 2022 German Aerospace Center (DLR e.V.), Ferdinand Rewicki
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

import numpy as np
import numpy.typing as npt

import merlin.discord_discovery_gemm3 as dd
import merlin.utils as utils


def merlin(
    timeseries: npt.ArrayLike,
    lmin: int,
    lmax: int,
    sanitize: bool = False,
    timeit: bool = False
) -> Tuple[List[int], List[float], List[int]]:
    """Implementation of the MERLIN anomaly detection algorithm

    This implementation is based on the following paper:

    T. Nakamura, M. Imamura, R. Mercer and E. Keogh,
    "MERLIN: Parameter-Free Discovery of Arbitrary Length Anomalies
    in Massive Time Series Archives,"
    2020 IEEE International Conference on Data Mining (ICDM),
    2020, pp. 1190-1195, doi: 10.1109/ICDM50108.2020.00147.

    and on the Matlab implementation by Ryan Mercer (Version 3.1)

    As opposed to the algorithm presented in the aforementioned paper,
    this implementation uses the discord_discovery_gemm3 algorithm as done
    in the Matlab implementation by Ryan Mercer (Version 3.1)

    :param timeseries: The timeseries provided as a numpy array.
    :param lmin: The minimum subsequence length.
    :param lmax: The maximum subsequence length.
    :param sanitize: Whether to add a large linear trend to timeseries
        with near constant subsequences as MERLIN might report false positive
        anomalies in this case.
    :param timeit: Whether to time the execution or not.

    :return:
        discords_idx: The start indices of the found discords
        discords_dist: The distance to the nearest neighbor for each found discord.
        lengths: The length of each found discord.

    """
    _t = np.array(timeseries)
    if _t.ndim > 2:
        raise ValueError(
            "Time Series t should be given as numpy array with one or two " "dimensions"
        )
    _t = np.expand_dims(_t, axis=0) if _t.ndim == 1 else _t
    _t = utils.z_normalize_timeseries(_t)
    if sanitize:
        _t = utils.sanitize_flat_subsequences(_t, lmin)

    lengths = list(range(lmin, lmax + 1))
    num_lengths = len(lengths)
    distances = np.full((num_lengths, 1), -1 * np.inf, dtype=float)
    discords_idx = np.zeros_like(distances, dtype=int)
    nn_indices = np.zeros_like(distances, dtype=int)
    times = np.zeros(num_lengths, dtype=float)

    r_start = 2 * np.sqrt(lmin)
    i = 0
    while distances[0, 0] < 0:
        tmp_index, tmp_dist, tmp_nn, times[i] = dd.discord_discovery_gemm3(
            _t, lengths[i], r_start, timeit=timeit, validate=False
        )

        if len(tmp_index) > 0:
            distances[0, 0] = np.max(tmp_dist)
            max_index = np.argmax(tmp_dist)

            discords_idx[0, 0] = tmp_index[max_index]
            nn_indices[0, 0] = tmp_nn[max_index]

        r_start = 0.5 * r_start

    for i in range(1, 5):
        if i >= num_lengths:
            break
        r = distances[i - 1, 0] * 0.99
        while distances[i, 0] < 0:
            tmp_index, tmp_dist, tmp_nn, times[i] = dd.discord_discovery_gemm3(
                _t, lengths[i], r, timeit=timeit, validate=False
            )
            if len(tmp_index) > 0:
                distances[i, 0] = np.max(tmp_dist)
                max_index = np.argmax(tmp_dist)

                discords_idx[i, 0] = tmp_index[max_index]
                nn_indices[i, 0] = tmp_nn[max_index]

            r = 0.99 * r

    for i in range(5, len(lengths)):
        if i >= num_lengths:
            break
        m = np.mean(distances[i - 5 : i, 0])
        s = np.std(distances[i - 5 : i, 0], ddof=1)
        r = m - 2 * s
        while distances[i, 0] < 0:
            tmp_index, tmp_dist, tmp_nn, times[i] = dd.discord_discovery_gemm3(
                _t, lengths[i], r, timeit=timeit, validate=False
            )
            if len(tmp_index) > 0:
                distances[i, 0] = np.max(tmp_dist)
                max_index = np.argmax(tmp_dist)

                discords_idx[i, 0] = tmp_index[max_index]
                nn_indices[i, 0] = tmp_nn[max_index]

            r = r * 0.99
            r = r * 0.99

    if timeit:
        print(f"Total Exec Time: {times.sum():.2f} sec.")

    return discords_idx.flatten(), distances.flatten(), lengths
