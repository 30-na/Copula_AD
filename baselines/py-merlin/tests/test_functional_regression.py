# SPDX-FileCopyrightText: 2022 German Aerospace Center (DLR e.V.), Ferdinand Rewicki
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import numpy.testing as npt

import merlin


class TestDiscordDiscoveryGemm3Regression:
    def test_nn_dist_and_loc_update(self):
        # given
        t = np.genfromtxt(
            "tests/fixtures/"
            "041_UCR_Anomaly_DISTORTEDLab2Cmac011215EPG5_7000_17390_17520.txt"
        )

        # when
        discords, distances, lengths = merlin.merlin(
            t, 130, 131, sanitize=False, timeit=False
        )

        # then
        npt.assert_array_equal(discords, [29640, 29639])
        npt.assert_array_almost_equal(distances, [10.7538686, 10.795978])
        npt.assert_array_equal(lengths, [130, 131])

    def test_first_confl_gt_zero_float_not_indexable(self):
        # given
        t = np.genfromtxt(
            "tests/fixtures/"
            "113_UCR_Anomaly_CIMIS44AirTemperature1_4000_5391_5392.txt"
        )

        # when
        discords, distances, lengths = merlin.merlin(
            t, 4, 5, sanitize=False, timeit=False
        )

        # then
        npt.assert_array_equal(discords, [4294, 7224])
        npt.assert_array_almost_equal(distances, [2.828427, 3.162278])
        npt.assert_array_equal(lengths, [4, 5])

    def test_ucr_004(self):
        # given
        t = np.genfromtxt(
            "tests/fixtures/" "004_UCR_Anomaly_DISTORTEDBIDMC1_2500_5400_5600.txt"
        )
        expected_discords = [
            5409,
            5408,
            5406,
            5405,
            5404,
            5403,
            5402,
            5401,
            5400,
            5399,
            5398,
        ]
        expected_distances = [
            10.86365958,
            10.8735411,
            10.88648238,
            10.90131496,
            10.91819758,
            10.93411313,
            10.94952048,
            10.96588535,
            10.98350432,
            11.00164631,
            11.02089862,
        ]
        expected_lengths = [170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180]

        # when
        discords, distances, lengths = merlin.merlin(
            t, 170, 180, sanitize=False, timeit=False
        )

        # then
        npt.assert_array_equal(discords, expected_discords)
        npt.assert_array_almost_equal(distances, expected_distances)
        npt.assert_array_equal(lengths, expected_lengths)

    def test_ucr_005(self):
        # given
        t = np.genfromtxt(
            "tests/fixtures/"
            "005_UCR_Anomaly_DISTORTEDCIMIS44AirTemperature1_4000_5391_5392.txt"
        )
        expected_discords_1 = [8112, 3885, 2826, 5388, 5387, 5386, 6452]
        expected_distances_1 = [
            0.11130339,
            1.0372883,
            1.40102328,
            1.6834964,
            1.95933054,
            2.19787436,
            2.37526861,
        ]
        expected_lengths_1 = [4, 5, 6, 7, 8, 9, 10]
        expected_discords_2 = [7019, 2826, 5388, 5387, 5386, 6452, 3741]
        expected_distances_2 = [
            0.51102285,
            1.40102328,
            1.6834964,
            1.95933054,
            2.19787436,
            2.37526861,
            2.93321243,
        ]
        expected_lengths_2 = [5, 6, 7, 8, 9, 10, 11]

        # when
        discords_1, distances_1, lengths_1 = merlin.merlin(
            t, 4, 10, sanitize=False, timeit=False
        )
        discords_2, distances_2, lengths_2 = merlin.merlin(
            t, 5, 11, sanitize=False, timeit=False
        )

        # then
        npt.assert_array_equal(discords_1, expected_discords_1)
        npt.assert_array_almost_equal(distances_1, expected_distances_1)
        npt.assert_array_equal(lengths_1, expected_lengths_1)

        npt.assert_array_equal(discords_2, expected_discords_2)
        npt.assert_array_almost_equal(distances_2, expected_distances_2)
        npt.assert_array_equal(lengths_2, expected_lengths_2)

    def test_ucr_060(self):
        # given
        t = np.genfromtxt(
            "tests/fixtures/" "060_UCR_Anomaly_DISTORTEDgait2_22000_46500_46800.txt"
        )
        expected_discords = [120]
        expected_distances = [16.54705816]
        expected_lengths = [285]

        # when
        discords, distances, lengths = merlin.merlin(
            t, 285, 285, sanitize=False, timeit=False
        )

        # then
        npt.assert_array_equal(discords, expected_discords)
        npt.assert_array_almost_equal(distances, expected_distances)
        npt.assert_array_equal(lengths, expected_lengths)
