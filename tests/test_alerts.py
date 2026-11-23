"""AQI scale and alert helper tests."""

import numpy as np
import pandas as pd
import pytest

from airwatch.alerts import ADVICE, GROUPS, LEVELS, aqi_level, best_window, nowcast, pm25_to_aqi


@pytest.mark.parametrize("pm,aqi", [
    (0.0, 0), (9.0, 50), (9.1, 51), (35.4, 100), (35.5, 101), (55.4, 150),
    (55.5, 151), (125.4, 200), (125.5, 201), (225.4, 300), (225.5, 301), (325.4, 500),
    (12.0, 56), (9.09, 50),
])
def test_pm25_to_aqi_breakpoints(pm, aqi):
    assert pm25_to_aqi(pm) == aqi


def test_aqi_invalid():
    assert pm25_to_aqi(float("nan")) is None
    assert pm25_to_aqi(-1) is None
    assert pm25_to_aqi(1000) == 500


def test_levels_and_colors():
    assert aqi_level(40).key == "good"
    assert aqi_level(100).key == "moderate"
    assert aqi_level(160).key == "unhealthy"
    assert aqi_level(450).key == "hazardous"
    assert all(lv.color.startswith("#") for lv in LEVELS)


def test_advice_complete():
    for lv in LEVELS:
        assert len(ADVICE[lv.key]) == len(GROUPS)


def test_nowcast_constant_and_weighting():
    assert nowcast(np.full(12, 20.0)) == pytest.approx(20.0)
    rising = np.linspace(10, 60, 12)
    assert nowcast(rising) > rising.mean()  # recent hours weigh more


def test_best_window_picks_lowest_daytime():
    t = pd.date_range("2025-01-01 00:00", periods=24, freq="h")
    pm = np.full(24, 50.0)
    pm[3:5] = 1.0  # night, must be ignored
    pm[13:15] = 10.0
    start, val = best_window(t, pm)
    assert start.hour == 13 and val == pytest.approx(10.0)
