"""Download and cache Open-Meteo air-quality and weather data."""

from __future__ import annotations

import logging
import os
import time
from datetime import date, timedelta

import pandas as pd
import requests

from airwatch import config as C

log = logging.getLogger(__name__)


def _get(url: str, params: dict, retries: int = 3) -> dict:
    """GET JSON with simple retries."""
    last: Exception | None = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            js = r.json()
            if js.get("error"):
                raise RuntimeError(js.get("reason"))
            return js
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"API failed: {url} ({last})")


def _to_frame(js: dict) -> pd.DataFrame:
    """Convert Open-Meteo hourly JSON to a DataFrame indexed by local time."""
    df = pd.DataFrame(js["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time").astype(float)


def _chunks(start: date, end: date, days: int = 180) -> list[tuple[date, date]]:
    out, s = [], start
    while s <= end:
        e = min(s + timedelta(days=days - 1), end)
        out.append((s, e))
        s = e + timedelta(days=1)
    return out


def fetch_air_quality(city: C.City, start: date, end: date) -> pd.DataFrame:
    """Hourly air quality (CAMS model data) for [start, end]; future days give CAMS forecast."""
    parts = []
    for s, e in _chunks(start, end):
        js = _get(C.AQ_URL, {
            "latitude": city.lat, "longitude": city.lon, "hourly": ",".join(C.AQ_VARS),
            "start_date": s.isoformat(), "end_date": e.isoformat(), "timezone": C.TIMEZONE,
        })
        parts.append(_to_frame(js))
    return pd.concat(parts)


def fetch_weather(city: C.City, start: date, end: date) -> pd.DataFrame:
    """Hourly weather: ERA5 archive for history + forecast API for the last ~2 weeks and future."""
    today = date.today()
    archive_end = min(end, today - timedelta(days=6))
    parts = []
    if start <= archive_end:
        for s, e in _chunks(start, archive_end, days=365):
            js = _get(C.ARCHIVE_URL, {
                "latitude": city.lat, "longitude": city.lon, "hourly": ",".join(C.WEATHER_VARS),
                "start_date": s.isoformat(), "end_date": e.isoformat(), "timezone": C.TIMEZONE,
            })
            parts.append(_to_frame(js))
    if end > archive_end:
        js = _get(C.FORECAST_URL, {
            "latitude": city.lat, "longitude": city.lon, "hourly": ",".join(C.WEATHER_VARS),
            "past_days": 14, "forecast_days": 3, "timezone": C.TIMEZONE,
        })
        parts.append(_to_frame(js))
    df = pd.concat(parts)
    # Prefer archive (first) over forecast for overlapping hours.
    return df[~df.index.duplicated(keep="first")].sort_index()


def raw_path(city_key: str) -> os.PathLike[str]:
    return C.RAW_DIR / f"{city_key}.parquet"


def fetch_city(city: C.City, refresh: bool = False) -> pd.DataFrame:
    """Return merged AQ+weather data for a city, using the parquet cache.

    Without ``refresh`` an existing cache is returned as-is (no API call).
    With ``refresh`` only the last 10 days onward are re-downloaded and merged.
    """
    path = raw_path(city.key)
    cached = pd.read_parquet(path) if path.exists() else None
    if cached is not None and not refresh:
        return cached
    end = date.today() + timedelta(days=2)
    if cached is not None:
        start = (cached.index.max() - pd.Timedelta(days=10)).date()
    else:
        start = date.fromisoformat(C.HISTORY_START)
    aq = fetch_air_quality(city, start, end)
    wx = fetch_weather(city, start, end)
    new = aq.join(wx, how="left")
    if cached is not None:
        new = pd.concat([cached[cached.index < new.index.min()], new])
    new = new[~new.index.duplicated(keep="last")].sort_index()
    new.index.name = "time"
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    new.to_parquet(path)
    log.info("%s: %d rows %s -> %s", city.key, len(new), new.index.min(), new.index.max())
    return new


def load_sample(city_key: str) -> pd.DataFrame:
    """Load bundled sample data (works offline)."""
    df = pd.read_csv(C.SAMPLE_DIR / f"{city_key}.csv", parse_dates=["time"]).set_index("time")
    return df.astype(float)


def load_city(city_key: str) -> pd.DataFrame:
    """Load cached raw data; fall back to bundled sample if no cache."""
    path = raw_path(city_key)
    if path.exists():
        return pd.read_parquet(path)
    log.warning("No cache for %s, using data/sample", city_key)
    return load_sample(city_key)


def fetch_all(refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Fetch every city; on API error fall back to cache or sample so work is not blocked."""
    out = {}
    for key, city in C.CITIES.items():
        try:
            out[key] = fetch_city(city, refresh=refresh)
        except Exception as e:  # noqa: BLE001
            log.error("Fetch failed for %s: %s -> using cache/sample", key, e)
            out[key] = load_city(key)
    return out


def fetch_openaq(city: C.City, radius_m: int = 25000) -> pd.DataFrame | None:
    """Optional: latest PM2.5 station readings from OpenAQ v3 (needs OPENAQ_API_KEY).

    Returns None when no key is set. Used only for display/validation, not training.
    """
    key = os.getenv("OPENAQ_API_KEY")
    if not key:
        return None
    r = requests.get(
        "https://api.openaq.org/v3/locations",
        params={"coordinates": f"{city.lat},{city.lon}", "radius": radius_m,
                "parameters_id": 2, "limit": 50},
        headers={"X-API-Key": key}, timeout=30,
    )
    r.raise_for_status()
    rows = [{"name": loc.get("name"), "lat": loc["coordinates"]["latitude"],
             "lon": loc["coordinates"]["longitude"],
             "last_update": (loc.get("datetimeLast") or {}).get("local")}
            for loc in r.json().get("results", [])]
    return pd.DataFrame(rows)
