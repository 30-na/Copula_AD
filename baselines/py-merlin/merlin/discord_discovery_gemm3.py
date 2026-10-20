# SPDX-FileCopyrightText: 2022 German Aerospace Center (DLR e.V.), Ferdinand Rewicki
#
# SPDX-License-Identifier: Apache-2.0

import math
import time
from typing import Tuple

import numpy as np
import numpy.typing as npt


def discord_discovery_gemm3(
    ts: npt.ArrayLike,
    subseqlen: int,
    r: float,
    validate: bool = True,
    timeit: bool = False,
) -> Tuple[npt.NDArray[int], npt.NDArray[float], npt.NDArray[int], float]:
    """Implementation of the discord_discovery_gemm algorithm

    Implementation of the discord_discovery_gemm algorithm, version 1.3.

    This implementation is based on the Malab implementation by Kaveh Kamgar
    which is based on the following paper:

    D. Yankov, E. Keogh and U. Rebbapragada,
    "Disk Aware Discord Discovery: Finding Unusual Time Series in Terabyte
    Sized Datasets,"
    Seventh IEEE International Conference on Data Mining (ICDM 2007),
    2007, pp. 381-390, doi: 10.1109/ICDM.2007.61.

    :param ts: time series given as numpy array
    :param subseqlen: subsequence length
    :param r: range parameter
    :param validate: Whether to validate and sanitize the input parameters or not.
    :param timeit: Whether to time the execution or not.
    and dimensions or not. This can lead to IndexErrors when turned off.
    Default: True

    :return:
        discords: starting indices of the found discords
        nn_dist: The distance between each discord an its nearest neighbor
        nn_pos: subsequence index of each discord's nearest neighbor.
        time: total execution time if timeit == True, -1 otherwise
    """
    times = {"init": -1.0, "select": -1.0, "refine": -1.0, "nn": -1.0}
    times["init"] = time.time()
    dd = DiscordDiscoveryGemm3(ts, subseqlen, r, validate)
    times["init"] = time.time() - times["init"]
    times["select"] = time.time()
    dd.select_candidates()
    times["select"] = time.time() - times["select"]
    times["refine"] = time.time()
    discords = dd.refine_discords()
    times["refine"] = time.time() - times["refine"]
    times["nn"] = time.time()
    nn_dist, nn_pos = dd.nn_dist_and_loc()
    times["nn"] = time.time() - times["nn"]

    dd.print_result()

    if timeit:
        totaltime = sum(times.values())
        print(
            f"length={subseqlen}, r={r:.3f} took: "
            f"{totaltime:.3f} seconds (init={times['init']:.3f}s, "
            f"select:{times['select']:.3f}s, refine: {times['refine']:.3f}s, "
            f"nn: {times['nn']:.3f}s)",
            end="\n",
            flush=True,
        )
    else:
        totaltime = -1.0

    return discords.flatten(), nn_dist, nn_pos.astype(int), totaltime


class DiscordDiscoveryGemm3:
    mxblocklen = 256

    def __init__(self, ts: npt.ArrayLike, subseqlen: int, r: float, validate=True):
        """

        :param ts: time series given as numpy array
        :param subseqlen: subsequence length
        :param r: range parameter
        :param validate: Whether to validate and sanitize the input parameters
        and dimensions or not. This can lead to IndexErrors when turned off.
        Default: True
        """
        self.ts: npt.NDArray = self._sanitize_input(ts) if validate else ts

        if validate:
            self._validate_subseqlen(subseqlen)
            self._validate_r(r, subseqlen)

        self.subseqlen: int = subseqlen
        self.r: float = 1 - (r**2) / (2 * subseqlen)

        self.min_sep: int = subseqlen
        self.subseqcount: int = self.ts.shape[1] - subseqlen + 1
        self.ss: npt.NDArray = self._init_subsequences()

        self.cand_count: int = 0
        self.cands: npt.NDArray = np.zeros((subseqlen, self.subseqcount))
        self.cands_idx: npt.NDArray = np.zeros((self.subseqcount, 1), dtype=int)

        self.disc_loc: npt.NDArray
        self.cands_nn_dist: npt.NDArray
        self.cands_nn_pos: npt.NDArray

    def _validate_subseqlen(self, subseqlen: int):
        if subseqlen <= 4 or math.floor(self.ts.shape[1] / 2) < subseqlen:
            raise ValueError(
                "Subsequence length parameter must be in the range of "
                f"4 < L <= floor(length(TS)/2) = {math.floor(self.ts.shape[1] / 2)}, "
                f"given: {subseqlen}"
            )

    def _validate_r(self, r: float, subseqlen: int):
        if not 0 < r <= 2 * math.sqrt(subseqlen):
            raise ValueError(
                "A valid choice of r must be greater than 0 and less than or "
                f"equal to 2 * sqrt(L) = {2 * math.sqrt(subseqlen)}, "
                f"given: {r}"
            )

    def _sanitize_input(self, ts: npt.ArrayLike) -> npt.NDArray:
        ts = np.array(ts)
        ts = np.expand_dims(ts, axis=0) if ts.ndim == 1 else ts

        return ts

    def _init_subsequences(self) -> npt.NDArray:
        ss = np.zeros((self.subseqlen, self.subseqcount))
        for i in range(0, self.subseqcount):
            ss[:, i] = self.ts[:, i : i + self.subseqlen]
        ss = ss - np.mean(ss, axis=0)
        ss_norm = np.linalg.norm(ss, ord=2, axis=0)
        ss = np.divide(ss, ss_norm, out=np.zeros_like(ss), where=ss_norm != 0)

        return ss

    def select_candidates(self):
        """Phase 1: Select discord candidates"""
        for i in range(0, self.subseqcount, self.mxblocklen):
            blocklen = min(self.mxblocklen, self.subseqcount - i)
            no_rej = np.ones((blocklen, 1), dtype=bool)

            if self.cand_count != 0:
                valid_cands = np.ones((self.cand_count, 1), dtype=bool)

                if int(self.cands_idx[0]) <= i + blocklen - self.min_sep - 1:
                    last_cmp = np.nonzero(
                        self.cands_idx[0 : self.cand_count]
                        <= i + blocklen - self.min_sep
                    )[0][-1]

                    if self.cands_idx[0] > i - self.min_sep:
                        last_no_confl = 0
                    else:
                        last_no_confl = np.nonzero(
                            self.cands_idx[0 : self.cand_count] <= i - self.min_sep
                        )[0][-1]
                        cr = (
                            self.cands[:, 0 : last_no_confl + 1].transpose()
                            @ self.ss[:, i : i + blocklen]
                        )
                        valid_cands[np.any(cr > self.r, axis=1).nonzero()[0]] = False
                        no_rej[np.any(cr > self.r, axis=0)] = False

                    for pos in range(last_no_confl, last_cmp + 1):
                        first_cmp_seq = int(max(i, self.cands_idx[pos] + self.min_sep))
                        cr = (
                            self.cands[:, pos]
                            @ self.ss[:, first_cmp_seq : i + blocklen]
                        )

                        invalid = (cr > self.r).nonzero()[0] + first_cmp_seq - i
                        if len(invalid) > 0:
                            valid_cands[pos] = False
                            no_rej[invalid] = False

                    surviving_count = sum(valid_cands)[0]
                    if surviving_count < self.cand_count:
                        self.cands_idx[0:surviving_count] = self.cands_idx[
                            valid_cands.nonzero()[0]
                        ]
                        self.cands[:, 0:surviving_count] = self.cands[
                            :, valid_cands.nonzero()[0]
                        ]
                        self.cand_count = surviving_count

            if blocklen >= self.min_sep:
                for pos in range(0 + self.min_sep, blocklen):
                    cr = (
                        self.ss[:, i + pos] @ self.ss[:, i : i + pos - self.min_sep + 1]
                    )
                    others = (cr > self.r).nonzero()[0]
                    if len(others) > 0:
                        no_rej[pos] = False
                        no_rej[others] = False

            can_add = no_rej.nonzero()[0] + i
            if len(can_add) > 0:
                prev_count = self.cand_count
                self.cand_count += len(can_add)
                self.cands[:, prev_count : prev_count + len(can_add)] = self.ss[
                    :, can_add
                ]
                self.cands_idx[prev_count : prev_count + len(can_add)] = can_add[
                    ..., None
                ]

    def refine_discords(self) -> npt.NDArray:
        """Phase 2: Refine discord candidates"""
        for i in range(0, self.subseqcount, self.mxblocklen):
            if self.cand_count == 0:
                break

            valid_cands = np.ones((self.cand_count, 1), dtype=bool)
            blocklen = min(self.mxblocklen, self.subseqcount - i)

            min_cand_idx = max(i + blocklen, i + self.min_sep)

            if self.cands_idx[self.cand_count - 1] < min_cand_idx:
                break

            first_cmp_pos = np.nonzero(
                self.cands_idx[0 : self.cand_count] >= min_cand_idx
            )[0][0]
            no_conflict_begin = i + blocklen + self.min_sep - 1

            if self.cands_idx[self.cand_count - 1] < no_conflict_begin:
                no_confl_cmp_pos = self.cand_count + 1
            else:
                idx_gt_no_conflict_begin = (
                    np.nonzero(
                        self.cands_idx[first_cmp_pos : self.cand_count]
                        >= no_conflict_begin
                    )[0][0]
                    + 1
                )

                if idx_gt_no_conflict_begin == 0 and first_cmp_pos == 0:
                    no_confl_cmp_pos = 1
                else:
                    no_confl_cmp_pos = idx_gt_no_conflict_begin + first_cmp_pos

            for pos in range(first_cmp_pos, no_confl_cmp_pos - 1):
                if np.any(
                    self.cands[:, pos].transpose()
                    @ self.ss[:, i : int(self.cands_idx[pos] - self.min_sep)]
                    > self.r
                ):
                    valid_cands[pos] = False

            if no_confl_cmp_pos < self.cand_count:
                cr = (
                    self.cands[:, no_confl_cmp_pos - 1 : self.cand_count].transpose()
                    @ self.ss[:, i : i + blocklen]
                )
                valid_cands[
                    np.nonzero(np.any(cr > self.r, axis=1)) + no_confl_cmp_pos - 1
                ] = False

            surviving_count = np.sum(valid_cands)
            if surviving_count < self.cand_count:
                self.cands_idx[0:surviving_count] = self.cands_idx[
                    valid_cands.nonzero()[0]
                ]
                self.cands[:, 0:surviving_count] = self.cands[
                    :, valid_cands.nonzero()[0]
                ]
                self.cand_count = surviving_count

        self.disc_loc = self.cands_idx[0 : self.cand_count]

        return self.disc_loc

    def nn_dist_and_loc(self) -> Tuple[npt.NDArray, npt.NDArray]:
        """Calculate nearest neighbor information for each found discord"""
        cands = self.cands[:, 0 : self.cand_count]
        cands_idx = self.cands_idx[0 : self.cand_count]

        self.cands_nn_dist = np.ones(self.cand_count) * -1
        self.cands_nn_pos = np.ones(self.cand_count) * -1

        if self.cand_count > 0:
            for i in range(0, self.subseqcount, self.mxblocklen):
                blocklen = min(self.mxblocklen, self.subseqcount - i)
                first_confl = np.nonzero(
                    (cands_idx > i - self.min_sep)
                    & (cands_idx <= i + blocklen + self.min_sep - 1)
                )[0]

                if len(first_confl) == 0:
                    first_confl = self.cand_count + 1
                    last_confl = []
                else:
                    first_confl = first_confl[0]
                    last_confl = np.nonzero(
                        (cands_idx > i - self.min_sep)
                        & (cands_idx <= i + blocklen + self.min_sep - 1)
                    )[0][-1]

                if first_confl > 0:
                    _mxtmp = (
                        cands[:, 0:first_confl].transpose()
                        @ self.ss[:, i : i + blocklen]
                    )
                    mxcr = np.max(_mxtmp, axis=1, initial=-1)
                    mxcor_pos = (
                        np.argmax(_mxtmp, axis=1) if (mxcr != -np.inf).all() else 0
                    )
                    update = np.nonzero(mxcr > self.cands_nn_dist[0:first_confl])
                    self.cands_nn_pos[update] = mxcor_pos[update] + i
                    self.cands_nn_dist[update] = mxcr[update]

                if first_confl <= self.cand_count:
                    for pos in range(first_confl, last_confl + 1):
                        if abs(cands_idx[pos] - i) < self.min_sep:
                            if abs(cands_idx[pos] - (i + blocklen - 1)) >= self.min_sep:
                                cmp_begin = int(cands_idx[pos] + self.min_sep)
                                _mxtmp = (
                                    cands[:, pos].transpose()
                                    @ self.ss[:, cmp_begin : i + blocklen]
                                )
                                mxcr = np.max(_mxtmp, axis=0, initial=-np.inf)
                                mxcr_idx = (
                                    np.argmax(_mxtmp, axis=0) if mxcr != -np.inf else 0
                                )
                                if mxcr != -np.inf and mxcr > self.cands_nn_dist[pos]:
                                    self.cands_nn_dist[pos] = mxcr
                                    self.cands_nn_pos[pos] = mxcr_idx
                        elif abs(cands_idx[pos] - (i + blocklen - 1)) < self.min_sep:
                            _mxtmp = (
                                cands[:, pos].transpose()
                                @ self.ss[
                                    :, i : int((cands_idx[pos] - self.min_sep)[0])
                                ]
                            )
                            mxcr = np.max(_mxtmp, axis=0, initial=-np.inf)
                            mxcr_idx = (
                                np.argmax(_mxtmp, axis=0) if mxcr != -np.inf else 0
                            )
                        else:
                            _mxtmp_lo = (
                                cands[:, pos].transpose()
                                @ self.ss[:, i : int(cands_idx[pos] - self.min_sep + 1)]
                            )
                            _mxtmp_hi = (
                                cands[:, pos].transpose()
                                @ self.ss[
                                    :, int(cands_idx[pos] + self.min_sep) : i + blocklen
                                ]
                            )
                            mxcr_lo = np.max(_mxtmp_lo, axis=0, initial=-np.inf)
                            mxcr_idx_lo = (
                                np.argmax(_mxtmp_lo, axis=0)
                                if mxcr_lo != -np.inf
                                else 0
                            )
                            mxcr_hi = np.max(_mxtmp_hi, axis=0, initial=-np.inf)
                            mxcr_idx_hi = (
                                np.argmax(_mxtmp_hi, axis=0)
                                if mxcr_hi != -np.inf
                                else 0
                            )
                            if mxcr_lo > mxcr_hi and mxcr_idx_hi > -1:
                                mxcr = mxcr_lo
                                mxcr_idx = mxcr_idx_lo + i
                            else:
                                mxcr = mxcr_hi
                                mxcr_idx = (
                                    mxcr_idx_hi
                                    + int(cands_idx[pos] + self.min_sep)
                                    + self.min_sep
                                )

                            if mxcr > float(self.cands_nn_dist[pos]):
                                self.cands_nn_dist[pos] = mxcr
                                self.cands_nn_pos[pos] = mxcr_idx

                    if last_confl + 1 < self.cand_count:
                        _mxtmp = (
                            cands[:, last_confl + 1 :].transpose()
                            @ self.ss[:, i : i + blocklen - 1]
                        )
                        mxcr = np.max(_mxtmp, axis=1)
                        mxcor_pos = np.argmax(_mxtmp, axis=1)
                        update = np.nonzero(mxcr > self.cands_nn_dist[last_confl + 1 :])
                        self.cands_nn_pos[update + last_confl + 1] = (
                            mxcor_pos[update] + i
                        )
                        self.cands_nn_dist[update + last_confl + 1] = mxcr[update]

        self.cands_nn_dist = np.sqrt(
            np.maximum(0, 2 * self.subseqlen * (1 - self.cands_nn_dist))
        )

        return self.cands_nn_dist, self.cands_nn_pos

    def print_result(self):
        if self.cands_nn_dist.shape[0] > 0:
            max_index = np.argmax(self.cands_nn_dist)
            print(
                f"The top discord of length {self.subseqlen} "
                f"is at {int(self.disc_loc[max_index])} "
                f"with a discord distance of {self.cands_nn_dist[max_index]:.3f}. "
                f"Its nearest neighbor is at {int(self.cands_nn_pos[max_index])}"
            )
