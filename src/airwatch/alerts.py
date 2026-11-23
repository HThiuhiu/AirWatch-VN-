"""AQI scale, health advice and best-time-to-go-out logic."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

# US EPA PM2.5 AQI breakpoints (revision effective May 2024).
PM25_BREAKPOINTS: list[tuple[float, float, int, int]] = [
    (0.0, 9.0, 0, 50),
    (9.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 125.4, 151, 200),
    (125.5, 225.4, 201, 300),
    (225.5, 325.4, 301, 500),
]


@dataclass(frozen=True)
class AqiLevel:
    """One AQI category."""

    lo: int
    hi: int
    key: str
    name_vi: str
    color: str


LEVELS: list[AqiLevel] = [
    AqiLevel(0, 50, "good", "Tốt", "#00e400"),
    AqiLevel(51, 100, "moderate", "Trung bình", "#ffff00"),
    AqiLevel(101, 150, "usg", "Kém (nhóm nhạy cảm)", "#ff7e00"),
    AqiLevel(151, 200, "unhealthy", "Xấu", "#ff0000"),
    AqiLevel(201, 300, "very_unhealthy", "Rất xấu", "#8f3f97"),
    AqiLevel(301, 500, "hazardous", "Nguy hại", "#7e0023"),
]


def pm25_to_aqi(pm: float) -> int | None:
    """Convert a PM2.5 concentration (µg/m³) to US AQI (EPA 2024 breakpoints)."""
    if pm is None or (isinstance(pm, float) and math.isnan(pm)) or pm < 0:
        return None
    c = math.floor(round(pm * 10, 6)) / 10  # EPA truncates to 0.1 µg/m³, so no band gaps
    for c_lo, c_hi, i_lo, i_hi in PM25_BREAKPOINTS:
        if c <= c_hi + 1e-9:
            return round((i_hi - i_lo) / (c_hi - c_lo) * (c - c_lo) + i_lo)
    return 500


def aqi_level(aqi: int | None) -> AqiLevel | None:
    if aqi is None:
        return None
    for lv in LEVELS:
        if aqi <= lv.hi:
            return lv
    return LEVELS[-1]


def nowcast(pm_last12: pd.Series | np.ndarray) -> float:
    """EPA NowCast for PM2.5 from the last 12 hourly values (most recent last)."""
    x = np.asarray(pm_last12, dtype=float)[-12:][::-1]  # most recent first
    valid = ~np.isnan(x)
    if valid[:3].sum() < 2:
        return float("nan")
    cmin, cmax = np.nanmin(x), np.nanmax(x)
    w = max(cmin / cmax, 0.5) if cmax > 0 else 1.0
    weights = w ** np.arange(len(x))
    return float(np.nansum(weights * np.nan_to_num(x)) / np.sum(weights[valid]))


GROUPS = ["Người khỏe mạnh", "Trẻ em", "Người cao tuổi", "Người có bệnh hô hấp/tim mạch"]

# advice[level_key][group_index] = dict of 4 actions
_YES, _NO, _LIMIT = "Có", "Không", "Hạn chế"


def _row(outside: str, exercise: str, mask: str, window: str) -> dict[str, str]:
    return {"Ra ngoài": outside, "Tập thể dục ngoài trời": exercise,
            "Đeo khẩu trang (N95/KF94)": mask, "Đóng cửa sổ": window}


ADVICE: dict[str, list[dict[str, str]]] = {
    "good": [_row(_YES, _YES, _NO, _NO)] * 4,
    "moderate": [
        _row(_YES, _YES, _NO, _NO),
        _row(_YES, _YES, _NO, _NO),
        _row(_YES, "Có, giảm cường độ", _NO, _NO),
        _row(_YES, "Có, theo dõi triệu chứng", "Nên đeo", _NO),
    ],
    "usg": [
        _row(_YES, "Có, giảm thời gian", "Nên đeo khi ở lâu", "Nên đóng"),
        _row(_LIMIT, "Hạn chế", "Nên đeo", "Nên đóng"),
        _row(_LIMIT, "Hạn chế", "Nên đeo", "Nên đóng"),
        _row(_LIMIT, "Không nên", "Đeo", "Đóng"),
    ],
    "unhealthy": [
        _row(_LIMIT, "Không nên (tập trong nhà)", "Đeo", "Đóng"),
        _row("Tránh ra ngoài", "Không", "Đeo", "Đóng"),
        _row("Tránh ra ngoài", "Không", "Đeo", "Đóng"),
        _row("Ở trong nhà", "Không", "Đeo", "Đóng, dùng lọc khí nếu có"),
    ],
    "very_unhealthy": [
        _row("Tránh ra ngoài", "Không", "Đeo", "Đóng, dùng lọc khí nếu có"),
    ] + [_row("Ở trong nhà", "Không", "Đeo khi buộc phải ra", "Đóng, dùng lọc khí nếu có")] * 3,
    "hazardous": [_row("Ở trong nhà", "Không", "Đeo khi buộc phải ra",
                       "Đóng, dùng lọc khí nếu có")] * 4,
}


def advice_table(level_key: str) -> pd.DataFrame:
    """Health advice for all groups at one AQI level (reference only, not medical advice)."""
    return pd.DataFrame(ADVICE[level_key], index=GROUPS)


def best_window(times: pd.Series, pm: pd.Series, hours: int = 2,
                day_start: int = 6, day_end: int = 21) -> tuple[pd.Timestamp, float] | None:
    """Lowest-mean ``hours``-long window of forecast PM2.5 within daytime hours.

    Returns (window start time, mean PM2.5) or None if no daytime window.
    """
    s = pd.Series(np.asarray(pm, dtype=float), index=pd.DatetimeIndex(times)).sort_index()
    roll = s.rolling(hours).mean().shift(-(hours - 1))
    ok = (s.index.hour >= day_start) & (s.index.hour + hours - 1 <= day_end)
    cand = roll[ok].dropna()
    if cand.empty:
        return None
    t = cand.idxmin()
    return t, float(cand.loc[t])
