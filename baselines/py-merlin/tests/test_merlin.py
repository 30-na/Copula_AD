# SPDX-FileCopyrightText: 2022 German Aerospace Center (DLR e.V.), Ferdinand Rewicki
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import numpy.testing
import numpy.typing as npt
import pytest
from pytest_mock import MockerFixture

import merlin


def get_univariate_time_series() -> npt.NDArray:
    time_series_t = np.linspace(1, 1000, 1000)
    time_series_t = np.sin(time_series_t)
    for i in range(10):
        time_series_t[600 + i] = i
        time_series_t[609 + i] = 10 - i

    return time_series_t


class TestMerlin:
    def test_it_returns_a_set_of_discords(self):
        # given
        time_series_t = get_univariate_time_series()
        minl, maxl = 15, 25
        expected_idxs = [594, 612, 607, 601, 596, 612, 588, 603, 602, 595, 590]
        expected_distances = [
            5.43,
            5.4433,
            5.6304,
            5.9793,
            6.0839,
            6.2652,
            6.4118,
            6.4835,
            6.7384,
            6.8088,
            6.899,
        ]
        expected_lengths = list(range(15, 26))

        # when
        discords, distances, lengths = merlin.merlin(time_series_t, minl, maxl)

        # then
        assert len(discords) == 11
        assert len(distances) == 11
        assert len(lengths) == 11

        np.testing.assert_array_equal(expected_idxs, discords)
        np.testing.assert_array_almost_equal(expected_distances, distances, decimal=4)
        np.testing.assert_array_equal(expected_lengths, lengths)

    def test_it_handles_multivariate_data(self, mocker: MockerFixture):
        pass

    def test_it_verifies_input_dimensions(self):
        time_series_t = np.zeros((2, 5, 100))
        minl, maxl = 15, 25

        with pytest.raises(
            ValueError,
            match="Time Series t should be given as numpy array with one or two "
            "dimensions",
        ):
            # when
            _ = merlin.merlin(time_series_t, minl, maxl)

    def test_it_can_add_linear_trend(self, mocker: MockerFixture):
        # given
        time_series_t = get_univariate_time_series()
        time_series_t = np.expand_dims(time_series_t, axis=0)
        minl, maxl = 15, 25
        sanitize = mocker.patch("merlin.utils.sanitize_flat_subsequences")
        sanitize.return_value = time_series_t

        # when
        _, _, _ = merlin.merlin(time_series_t, minl, maxl, sanitize=True)

        # then
        sanitize.assert_called_once()
