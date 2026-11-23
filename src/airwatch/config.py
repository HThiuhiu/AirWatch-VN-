"""Central configuration for AirWatch VN."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SEED: int = 42

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"
FORECAST_DIR = DATA_DIR / "forecast"
MODEL_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
METRICS_PATH = REPORTS_DIR / "metrics.json"

TIMEZONE = "Asia/Bangkok"  # UTC+7, same offset as Vietnam (Asia/Ho_Chi_Minh)
HISTORY_START = "2023-01-01"


@dataclass(frozen=True)
class City:
    """A city to monitor."""

    key: str
    name_vi: str
    lat: float
    lon: float


CITIES: dict[str, City] = {
    "hanoi": City("hanoi", "Hà Nội", 21.0285, 105.8542),
    "hcmc": City("hcmc", "TP. Hồ Chí Minh", 10.8231, 106.6297),
}

AQ_VARS = ["pm2_5", "pm10", "nitrogen_dioxide", "ozone", "carbon_monoxide", "us_aqi"]
WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "boundary_layer_height",
]

AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Thresholds for 24-hour mean PM2.5 (µg/m³).
# QCVN 05:2023/BTNMT (Vietnam national ambient air quality standard): 24h PM2.5 = 50.
# WHO Air Quality Guidelines 2021: 24h PM2.5 = 15.
THRESHOLDS: dict[str, float] = {"qcvn": 50.0, "who": 15.0}

TARGET = "pm2_5"
HORIZONS: list[int] = list(range(1, 25))
LAGS: list[int] = [1, 2, 3, 6, 12, 24]
ROLL_WINDOWS: list[int] = [6, 24]
QUANTILES: list[float] = [0.1, 0.5, 0.9]

# Time split (local dates). Train < VAL_START <= val < TEST_START <= test.
VAL_START = "2025-07-01"
TEST_START = "2025-11-01"
# Rolling-origin backtest: folds start every BACKTEST_MONTHS from BACKTEST_FIRST until data end.
# Each fold trains on all data before its start (expanding window) and tests the next block.
BACKTEST_FIRST = "2025-01-01"
BACKTEST_MONTHS = 3
# Train on every k-th origin hour to keep training fast (test always uses every hour).
TRAIN_ORIGIN_STRIDE = 2

LGB_PARAMS: dict = {
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": SEED,
    "deterministic": True,
    "num_threads": 4,
}
NUM_BOOST_ROUND = 2000
EARLY_STOPPING = 100
