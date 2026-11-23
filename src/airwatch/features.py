"""Cleaning and leak-free feature engineering.

Rule: every feature at origin time ``t`` uses only data observed at or before ``t``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from airwatch import config as C

CITY_CODE = {k: i for i, k in enumerate(C.CITIES)}


def clean(df: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Regular hourly index, drop future rows, remove outliers, fill short gaps.

    Only backward-looking operations are used (trailing median, forward fill),
    so cleaning itself cannot leak future values.
    """
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    if now is not None:
        df = df[df.index <= now]
    idx = pd.date_range(df.index.min(), df.index.max(), freq="h", name="time")
    df = df.reindex(idx)
    pollutants = [c for c in C.AQ_VARS if c in df]
    df[pollutants] = df[pollutants].where(df[pollutants] >= 0)
    # Spike filter: > 4x trailing 72h median and > 150 µg/m³ => treat as missing.
    med = df[C.TARGET].rolling(72, min_periods=12).median().shift(1)
    spike = (df[C.TARGET] > 4 * med) & (df[C.TARGET] > 150)
    df.loc[spike, C.TARGET] = np.nan
    df["precipitation"] = df["precipitation"].fillna(0.0) if "precipitation" in df else 0.0
    return df.ffill(limit=3)


def origin_features(df: pd.DataFrame, city_key: str) -> pd.DataFrame:
    """Features known at each origin hour t (index = t)."""
    p = df[C.TARGET]
    f = pd.DataFrame(index=df.index)
    f["pm25"] = p
    for lag in C.LAGS:
        f[f"pm25_lag{lag}"] = p.shift(lag)
    for w in C.ROLL_WINDOWS:
        r = p.rolling(w, min_periods=max(2, w // 2))
        f[f"pm25_mean{w}"] = r.mean()
        f[f"pm25_max{w}"] = r.max()
        f[f"pm25_std{w}"] = r.std()
    f["pm25_mean72"] = p.rolling(72, min_periods=36).mean()
    f["pm25_diff1"] = p - p.shift(1)
    f["pm25_diff24"] = p - p.shift(24)
    for c in ["pm10", "nitrogen_dioxide", "ozone", "carbon_monoxide"]:
        if c in df:
            f[c] = df[c]
            f[f"{c}_mean24"] = df[c].rolling(24, min_periods=12).mean()
    f["temp"] = df["temperature_2m"]
    f["rh"] = df["relative_humidity_2m"]
    f["wind_speed"] = df["wind_speed_10m"]
    f["wind_speed_mean6"] = df["wind_speed_10m"].rolling(6, min_periods=3).mean()
    rad = np.deg2rad(df["wind_direction_10m"])
    f["wind_sin"] = np.sin(rad)
    f["wind_cos"] = np.cos(rad)
    # Wind vector components (u: east-west, v: north-south), "from" convention.
    f["wind_u"] = -df["wind_speed_10m"] * np.sin(rad)
    f["wind_v"] = -df["wind_speed_10m"] * np.cos(rad)
    f["precip"] = df["precipitation"]
    f["precip_6h"] = df["precipitation"].rolling(6, min_periods=1).sum()
    f["precip_24h"] = df["precipitation"].rolling(24, min_periods=1).sum()
    f["blh"] = df["boundary_layer_height"]
    f["blh_mean24"] = df["boundary_layer_height"].rolling(24, min_periods=12).mean()
    f["temp_diff24"] = df["temperature_2m"] - df["temperature_2m"].shift(24)
    t = f.index
    f["hour"] = t.hour
    f["dow"] = t.day_of_week
    f["month"] = t.month
    f["season"] = (t.month % 12) // 3  # 0=DJF, 1=MAM, 2=JJA, 3=SON
    f["city"] = CITY_CODE[city_key]
    return f


def supervised(feats: pd.DataFrame, series: pd.Series, horizons: list[int] | None = None,
               stride: int = 1) -> pd.DataFrame:
    """Stack origins x horizons into one long table with target ``y`` = PM2.5 at t+h.

    Also adds baseline predictions: persistence (value at t) and seasonal naive
    (value at t+h-24, which is <= t for h <= 24, so it is known at t).
    """
    horizons = horizons or C.HORIZONS
    base = feats.iloc[::stride]
    out = []
    for h in horizons:
        x = base.copy()
        x["horizon"] = h
        tgt_time = x.index + pd.Timedelta(hours=h)
        x["target_hour"] = tgt_time.hour
        x["target_hour_sin"] = np.sin(2 * np.pi * tgt_time.hour / 24)
        x["target_hour_cos"] = np.cos(2 * np.pi * tgt_time.hour / 24)
        x["y"] = series.reindex(tgt_time).to_numpy()
        x["base_persist"] = series.reindex(x.index).to_numpy()
        x["base_snaive"] = series.reindex(tgt_time - pd.Timedelta(hours=24)).to_numpy()
        x["target_time"] = tgt_time
        out.append(x)
    df = pd.concat(out)
    df.index.name = "origin"
    return df


def future_mean24(series: pd.Series) -> pd.Series:
    """Mean PM2.5 over (t, t+24] for each origin t (classification target)."""
    return series[::-1].rolling(24, min_periods=24).mean()[::-1].shift(-1)


FEATURE_EXCLUDE = {"y", "base_persist", "base_snaive", "target_time", "y_mean24", "city_key"}


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in FEATURE_EXCLUDE]


def build_all(raw: dict[str, pd.DataFrame], now: pd.Timestamp | None = None
              ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Clean + build origin features for all cities.

    Returns (origin table with city_key and y_mean24 columns, cleaned per-city frames).
    """
    rows, cleaned = [], {}
    for key, df in raw.items():
        cdf = clean(df, now=now)
        cleaned[key] = cdf
        f = origin_features(cdf, key)
        f["y_mean24"] = future_mean24(cdf[C.TARGET])
        f["city_key"] = key
        rows.append(f)
    return pd.concat(rows), cleaned


def save_processed(origin: pd.DataFrame) -> None:
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    origin.to_parquet(C.PROCESSED_DIR / "features.parquet")
