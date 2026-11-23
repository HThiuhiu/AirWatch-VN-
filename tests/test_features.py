"""Leakage tests: features at origin t must not change when the future changes."""

import numpy as np
import pandas as pd
import pytest

from airwatch import config as C
from airwatch.features import clean, future_mean24, origin_features, supervised
from airwatch.fetch import load_sample


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return load_sample("hanoi")


def test_origin_features_ignore_future(raw):
    t = raw.index[len(raw) // 2]
    base = origin_features(clean(raw), "hanoi")
    noisy = raw.copy()
    rng = np.random.default_rng(0)
    fut = noisy.index > t
    noisy.loc[fut] = noisy.loc[fut] * rng.uniform(0.2, 5.0, size=noisy.loc[fut].shape)
    pert = origin_features(clean(noisy), "hanoi")
    pd.testing.assert_series_equal(base.loc[t], pert.loc[t])
    # All rows up to t identical too
    pd.testing.assert_frame_equal(base.loc[:t], pert.loc[:t])


def test_truncated_history_gives_same_features(raw):
    """Features computed on data cut at t equal features computed on full data."""
    t = raw.index[700]
    full = origin_features(clean(raw), "hanoi").loc[t]
    cut = origin_features(clean(raw[raw.index <= t]), "hanoi").loc[t]
    pd.testing.assert_series_equal(full, cut)


def test_supervised_target_alignment(raw):
    cdf = clean(raw)
    f = origin_features(cdf, "hanoi")
    s = supervised(f, cdf[C.TARGET], horizons=[1, 6, 24])
    t = f.index[100]
    for h in [1, 6, 24]:
        row = s[(s.index == t) & (s["horizon"] == h)].iloc[0]
        assert row["y"] == cdf[C.TARGET].loc[t + pd.Timedelta(hours=h)]
        assert row["base_persist"] == cdf[C.TARGET].loc[t]
        assert row["base_snaive"] == cdf[C.TARGET].loc[t + pd.Timedelta(hours=h - 24)]
        # seasonal naive must be known at origin
        assert t + pd.Timedelta(hours=h - 24) <= t


def test_future_mean24(raw):
    p = clean(raw)[C.TARGET]
    fm = future_mean24(p)
    t = p.index[50]
    expected = p.loc[t + pd.Timedelta(hours=1): t + pd.Timedelta(hours=24)].mean()
    assert fm.loc[t] == pytest.approx(expected)


def test_clean_drops_future_and_negative(raw):
    d = raw.copy()
    d.iloc[10, d.columns.get_loc("pm2_5")] = -5
    now = d.index[500]
    c = clean(d, now=now)
    assert c.index.max() == now
    assert (c["pm2_5"].dropna() >= 0).all()
