# SPDX-FileCopyrightText: 2022 German Aerospace Center (DLR e.V.), Ferdinand Rewicki
#
# SPDX-License-Identifier: Apache-2.0

import matplotlib
import numpy as np
import numpy.testing as npts
import numpy.typing as npt

import merlin.utils as utils


def get_univariate_time_series() -> npt.NDArray:
    time_series_t = np.linspace(1, 1000, 1000)
    time_series_t = np.sin(time_series_t)
    for i in range(10):
        time_series_t[600 + i] = i
        time_series_t[609 + i] = 10 - i

    return time_series_t


class TestNormalization:
    def test_it_z_normalizes_a_timeseries(self):
        # given
        ts = get_univariate_time_series()
        expected = (ts - np.nanmean(ts)) / np.nanstd(ts)

        # when
        ts_normalized = utils.z_normalize_timeseries(ts)

        # then
        npts.assert_array_almost_equal(ts_normalized, expected)


class TestFlatSubsequenceSanitation:
    def test_it_adds_linear_trend_if_time_series_contains_flat_subsequences(self):
        # given
        ts = get_univariate_time_series()
        ts_flat = get_univariate_time_series()
        ts_flat[100:120] = 1

        ts = utils.z_normalize_timeseries(np.expand_dims(ts, axis=0))
        ts_flat = utils.z_normalize_timeseries(np.expand_dims(ts_flat, axis=0))

        # when
        ts_sanitized = utils.sanitize_flat_subsequences(ts, 15)
        ts_flat_sanitized = utils.sanitize_flat_subsequences(ts_flat, 15)

        # then
        np.testing.assert_array_equal(
            ts_sanitized, ts, err_msg="Non-flat TS schanges after sanitation"
        )

        expected_ts = (ts_flat - np.nanmean(ts_flat)) / np.nanstd(ts_flat) + range(
            0, 1000
        )
        np.testing.assert_array_almost_equal(
            ts_flat_sanitized, expected_ts, err_msg="Flat TS not meeting expectation"
        )


class TestPlot:
    def test_it_plots_the_results(self):
        # given
        ts = get_univariate_time_series()
        discords = list(range(605, 615))
        lengths = list(range(10, 20))
        anomalies = [(600, 620)]
        name = "Test"

        # when
        fig, ax = utils.plot(ts, discords, lengths, anomalies, tol=0, name=name)

        # then
        assert isinstance(fig, matplotlib.pyplot.Figure)
        assert isinstance(ax, np.ndarray)
        assert ax.shape == (2,)
        assert isinstance(ax[0], matplotlib.pyplot.Axes)
        assert isinstance(ax[1], matplotlib.pyplot.Axes)
        npts.assert_array_equal(ax[0].lines[0].get_data()[1], ts)
