"""Fetch tests using the bundled sample and a mocked API (no network)."""

import pandas as pd

from airwatch import config as C
from airwatch import fetch as F


def test_sample_has_expected_columns():
    for key in C.CITIES:
        df = F.load_sample(key)
        assert set(C.AQ_VARS + C.WEATHER_VARS) <= set(df.columns)
        assert df.index.is_monotonic_increasing
        assert len(df) > 24 * 30


def test_to_frame_parses_open_meteo_json():
    js = {"hourly": {"time": ["2025-01-01T00:00", "2025-01-01T01:00"],
                     "pm2_5": [10.5, None]}}
    df = F._to_frame(js)
    assert list(df.columns) == ["pm2_5"]
    assert df.index[0] == pd.Timestamp("2025-01-01 00:00")
    assert pd.isna(df["pm2_5"].iloc[1])


def test_fetch_all_falls_back_to_sample(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "RAW_DIR", tmp_path)

    def boom(*a, **k):
        raise RuntimeError("no network")

    monkeypatch.setattr(F, "_get", boom)
    out = F.fetch_all(refresh=True)
    assert set(out) == set(C.CITIES)
    assert all(len(v) > 0 for v in out.values())


def test_cache_is_used_without_refresh(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "RAW_DIR", tmp_path)
    F.load_sample("hanoi").to_parquet(tmp_path / "hanoi.parquet")
    monkeypatch.setattr(F, "_get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("API")))
    df = F.fetch_city(C.CITIES["hanoi"], refresh=False)
    assert len(df) > 0


def test_chunks_cover_range():
    from datetime import date
    ch = F._chunks(date(2023, 1, 1), date(2024, 1, 10), days=180)
    assert ch[0][0] == date(2023, 1, 1) and ch[-1][1] == date(2024, 1, 10)
    for (_, e), (s, _) in zip(ch, ch[1:], strict=False):
        assert (s - e).days == 1
