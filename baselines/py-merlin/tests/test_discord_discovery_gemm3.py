# SPDX-FileCopyrightText: 2022 German Aerospace Center (DLR e.V.), Ferdinand Rewicki
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import numpy.testing
import numpy.typing as npt
import pytest

from merlin.discord_discovery_gemm3 import (
    DiscordDiscoveryGemm3,
    discord_discovery_gemm3,
)


def get_univariate_time_series() -> npt.NDArray:
    time_series_t = np.linspace(1, 1000, 1000)
    time_series_t = np.sin(time_series_t)
    for i in range(10):
        time_series_t[600 + i] = i
        time_series_t[609 + i] = 10 - i

    return time_series_t


class TestDiscordDiscoveryGemm3:
    def test_it_exposes_a_function_endpoint(self):
        # given
        time_series_t = get_univariate_time_series()
        subseqlen = 15
        r = 3.8

        expected_discords = list(range(591, 615))
        expected_distances = [
            3.8121,
            4.4826,
            5.0789,
            5.4300,
            5.2033,
            5.0381,
            4.9255,
            4.8308,
            4.7058,
            4.5918,
            4.5890,
            4.6749,
            4.6749,
            4.5890,
            4.5766,
            4.6638,
            4.7847,
            4.8276,
            4.8271,
            4.9100,
            5.1464,
            5.3740,
            4.8819,
            4.040,
        ]
        expected_nn_idx = [
            547,
            403,
            127,
            302,
            295,
            836,
            931,
            664,
            857,
            493,
            659,
            933,
            116,
            405,
            380,
            653,
            92,
            231,
            282,
            113,
            76,
            28,
            105,
            935,
        ]

        # when
        discords, nn_dist, nn_idx, _ = discord_discovery_gemm3(
            time_series_t, subseqlen, r
        )

        # then
        np.testing.assert_array_equal(expected_discords, discords)
        np.testing.assert_array_almost_equal(expected_distances, nn_dist, decimal=4)
        np.testing.assert_array_equal(expected_nn_idx, nn_idx)

    def test_it_calculates_a_list_of_discord_candidates_univariate(self):
        # given
        time_series_t = get_univariate_time_series()
        subseqlen = 15
        r = 3.8
        gemm3 = DiscordDiscoveryGemm3(time_series_t, subseqlen, r)

        # the expectations have been generated using the original matlab implementation
        # of the discord_discovery_gemm3 by Kaveh Kamgar
        expected_candidates = list(range(591, 615))

        # when
        gemm3.select_candidates()
        cands_idx = gemm3.cands_idx

        # then
        assert len(cands_idx[cands_idx > 0]) == 24
        np.testing.assert_array_equal(cands_idx[cands_idx > 0], expected_candidates)

    def test_it_refines_the_discord_candidates(self):
        # given
        time_series_t = get_univariate_time_series()
        subseqlen = 15
        r = 3.8

        # the expectations have been generated using the original matlab implementation
        # of the discord_discovery_gemm3 by Kaveh Kamgar
        expected_candidates = list(range(591, 615))
        gemm3 = DiscordDiscoveryGemm3(time_series_t, subseqlen, r)
        gemm3.select_candidates()

        # when
        disc_loc = gemm3.refine_discords()

        # then
        assert len(disc_loc) == 24
        np.testing.assert_array_equal(disc_loc.flatten(), expected_candidates)

    def test_it_calculates_nearest_neighbour_information(self):
        # given
        time_series_t = get_univariate_time_series()
        subseqlen = 15
        r = 3.8

        # the expectations have been generated using the original matlab implementation
        # of the discord_discovery_gemm3 by Kaveh Kamgar
        expected_distances = [
            3.8121,
            4.4826,
            5.0789,
            5.4300,
            5.2033,
            5.0381,
            4.9255,
            4.8308,
            4.7058,
            4.5918,
            4.5890,
            4.6749,
            4.6749,
            4.5890,
            4.5766,
            4.6638,
            4.7847,
            4.8276,
            4.8271,
            4.9100,
            5.1464,
            5.3740,
            4.8819,
            4.040,
        ]
        gemm3 = DiscordDiscoveryGemm3(time_series_t, subseqlen, r)
        gemm3.select_candidates()
        _ = gemm3.refine_discords()

        # when
        cands_nn_dist, cands_nn_pos = gemm3.nn_dist_and_loc()

        # then
        np.testing.assert_array_almost_equal(
            cands_nn_dist, expected_distances, decimal=4
        )

    def test_it_validates_subsequence_length(self):
        time_series_t = get_univariate_time_series()
        r = 3.8
        with pytest.raises(ValueError, match="^Subsequence length parameter must .*$"):
            _ = DiscordDiscoveryGemm3(time_series_t, 4, r)

        gemm3 = DiscordDiscoveryGemm3(time_series_t, 5, r)
        assert isinstance(gemm3, DiscordDiscoveryGemm3)

        gemm3 = DiscordDiscoveryGemm3(time_series_t, 500, r)
        assert isinstance(gemm3, DiscordDiscoveryGemm3)

        with pytest.raises(ValueError, match="^Subsequence length parameter must .*$"):
            _ = DiscordDiscoveryGemm3(time_series_t, 501, r)

    def test_it_validates_r(self):
        time_series_t = get_univariate_time_series()
        subseqlen = 15
        with pytest.raises(ValueError, match="^A valid choice of r must be .*$"):
            _ = DiscordDiscoveryGemm3(time_series_t, subseqlen, 0)

        gemm3 = DiscordDiscoveryGemm3(time_series_t, subseqlen, 1)
        assert isinstance(gemm3, DiscordDiscoveryGemm3)

        gemm3 = DiscordDiscoveryGemm3(time_series_t, subseqlen, 7.74)
        assert isinstance(gemm3, DiscordDiscoveryGemm3)

        with pytest.raises(ValueError, match="^A valid choice of r must be .*$"):
            _ = DiscordDiscoveryGemm3(time_series_t, subseqlen, 7.75)

    def test_it_sanitizes_the_input(self):
        # given
        time_series_t = get_univariate_time_series()

        # when
        gemm3 = DiscordDiscoveryGemm3(time_series_t, 15, 1)

        # then
        assert gemm3.ts.shape == (1, 1000)

    def test_input_verification_can_be_turned_off(self):
        # given
        time_series_t = get_univariate_time_series()

        # when
        with pytest.raises(IndexError, match="^tuple index out of range$"):
            _ = DiscordDiscoveryGemm3(time_series_t, 4, 3.8, validate=False)

        with pytest.raises(IndexError, match="^tuple index out of range$"):
            _ = DiscordDiscoveryGemm3(time_series_t, 15, 0, validate=False)

    def test_dimension_expansion_can_be_turned_off(self):
        # given
        time_series_t = get_univariate_time_series()

        # when
        with pytest.raises(IndexError, match="^tuple index out of range$"):
            _ = DiscordDiscoveryGemm3(time_series_t, 15, 1, validate=False)
