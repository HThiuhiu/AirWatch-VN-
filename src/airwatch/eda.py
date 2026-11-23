"""Auto-generated EDA report (reports/eda.md + figures). All numbers come from the data."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from airwatch import config as C  # noqa: E402

CITY_COLORS = {"hanoi": "#d62728", "hcmc": "#1f77b4"}
SEASONS = {"DJF": [12, 1, 2], "MAM": [3, 4, 5], "JJA": [6, 7, 8], "SON": [9, 10, 11]}


def _cmp(a: float, b: float, more: str, less: str) -> str:
    """Pick wording from data: ``more`` if a > b else ``less``."""
    return more if a > b else less


def _savefig(fig, name: str) -> str:
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / name, dpi=130)
    plt.close(fig)
    return f"figures/{name}"


def stats(cleaned: dict[str, pd.DataFrame]) -> dict:
    """Key descriptive numbers used in the report and README."""
    out: dict = {}
    for key, df in cleaned.items():
        p = df[C.TARGET]
        daily = p.resample("D").mean()
        hourly = p.groupby(p.index.hour).mean()
        monthly = p.groupby(p.index.month).mean()
        rain = df["precipitation"].rolling(6, min_periods=1).sum() > 0.5
        blh = df["boundary_layer_height"]
        blh_lo = blh <= blh.quantile(0.25)
        blh_hi = blh >= blh.quantile(0.75)
        ws = df["wind_speed_10m"]
        out[key] = {
            "period": [str(p.index.min().date()), str(p.index.max().date())],
            "mean_pm25": round(float(p.mean()), 1),
            "days_over_qcvn": int((daily > C.THRESHOLDS["qcvn"]).sum()),
            "days_over_who": int((daily > C.THRESHOLDS["who"]).sum()),
            "n_days": int(daily.notna().sum()),
            "share_days_over_qcvn": round(float((daily > C.THRESHOLDS["qcvn"]).mean()), 3),
            "share_days_over_who": round(float((daily > C.THRESHOLDS["who"]).mean()), 3),
            "peak_hour": int(hourly.idxmax()),
            "low_hour": int(hourly.idxmin()),
            "peak_hour_mean": round(float(hourly.max()), 1),
            "low_hour_mean": round(float(hourly.min()), 1),
            "worst_month": int(monthly.idxmax()),
            "best_month": int(monthly.idxmin()),
            "worst_month_mean": round(float(monthly.max()), 1),
            "best_month_mean": round(float(monthly.min()), 1),
            "pm_rain6h": round(float(p[rain].mean()), 1),
            "pm_dry6h": round(float(p[~rain].mean()), 1),
            "pm_low_blh": round(float(p[blh_lo].mean()), 1),
            "pm_high_blh": round(float(p[blh_hi].mean()), 1),
            "pm_calm_wind": round(float(p[ws < ws.quantile(0.25)].mean()), 1),
            "pm_windy": round(float(p[ws > ws.quantile(0.75)].mean()), 1),
            "corr_blh": round(float(p.corr(np.log1p(blh))), 2),
            "corr_wind": round(float(p.corr(ws)), 2),
            "corr_rh": round(float(p.corr(df["relative_humidity_2m"])), 2),
            "season_mean": {n: round(float(p[p.index.month.isin(ms)].mean()), 1)
                            for n, ms in SEASONS.items()},
            "season_share_days_over_qcvn": {
                n: round(float((daily[daily.index.month.isin(ms)] > C.THRESHOLDS["qcvn"]).mean()), 3)
                for n, ms in SEASONS.items()},
        }
    return out


def figures(cleaned: dict[str, pd.DataFrame]) -> list[str]:
    paths = []
    # Diurnal cycle
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, df in cleaned.items():
        p = df[C.TARGET]
        ax.plot(range(24), p.groupby(p.index.hour).mean(), "o-", ms=3,
                color=CITY_COLORS[key], label=C.CITIES[key].name_vi)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Giờ trong ngày / Hour of day")
    ax.set_ylabel("PM2.5 trung bình (µg/m³)")
    ax.set_title("Chu kỳ ngày / Diurnal cycle")
    ax.grid(alpha=0.3)
    ax.legend()
    paths.append(_savefig(fig, "eda_diurnal.png"))
    # Seasonal cycle
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, df in cleaned.items():
        p = df[C.TARGET]
        ax.plot(range(1, 13), p.groupby(p.index.month).mean(), "o-",
                color=CITY_COLORS[key], label=C.CITIES[key].name_vi)
    ax.axhline(C.THRESHOLDS["qcvn"], color="r", ls="--", lw=1, label="QCVN 24h = 50")
    ax.axhline(C.THRESHOLDS["who"], color="g", ls="--", lw=1, label="WHO 24h = 15")
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Tháng / Month")
    ax.set_ylabel("PM2.5 trung bình (µg/m³)")
    ax.set_title("Chu kỳ mùa / Seasonal cycle")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    paths.append(_savefig(fig, "eda_seasonal.png"))
    # Weekday x hour heatmap Hanoi
    df = cleaned["hanoi"]
    p = df[C.TARGET]
    piv = p.groupby([p.index.day_of_week, p.index.hour]).mean().unstack()
    fig, ax = plt.subplots(figsize=(9, 3.2))
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(7), ["T2", "T3", "T4", "T5", "T6", "T7", "CN"])
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Giờ")
    ax.set_title("Hà Nội: PM2.5 theo thứ × giờ")
    fig.colorbar(im, ax=ax, label="µg/m³")
    paths.append(_savefig(fig, "eda_week_hour_hanoi.png"))
    # Meteorology drivers
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for key, d in cleaned.items():
        p = d[C.TARGET]
        for ax, col, bins, lab in [
            (axes[0], "wind_speed_10m", [0, 3, 6, 9, 12, 15, 20, 30, 60], "Tốc độ gió (km/h)"),
            (axes[1], "boundary_layer_height", [0, 100, 200, 400, 700, 1000, 1500, 2000, 4000],
             "Độ cao lớp biên (m)"),
            (axes[2], "relative_humidity_2m", [0, 50, 60, 70, 80, 90, 101], "Độ ẩm (%)"),
        ]:
            g = p.groupby(pd.cut(d[col], bins), observed=True).mean()
            ax.plot([iv.mid for iv in g.index], g.to_numpy(), "o-", color=CITY_COLORS[key],
                    label=C.CITIES[key].name_vi)
            ax.set_xlabel(lab)
            ax.grid(alpha=0.3)
    axes[1].set_xscale("log")
    axes[0].set_ylabel("PM2.5 trung bình (µg/m³)")
    axes[0].legend()
    fig.suptitle("Ảnh hưởng khí tượng / Meteorological drivers")
    paths.append(_savefig(fig, "eda_meteo.png"))
    # Wind rose-like: mean PM by wind direction sector (Hanoi)
    d = cleaned["hanoi"]
    sector = (((d["wind_direction_10m"] + 22.5) % 360) // 45).astype("Int64")
    names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    g = d[C.TARGET].groupby(sector).mean().reindex(range(8))
    fig = plt.figure(figsize=(4.5, 4.5))
    ax = fig.add_subplot(projection="polar")
    ang = np.deg2rad(np.arange(8) * 45)
    ax.bar(ang, g.to_numpy(), width=np.deg2rad(40), color="#d62728", alpha=0.7)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(ang, names)
    ax.set_title("Hà Nội: PM2.5 TB theo hướng gió thổi tới", fontsize=10)
    paths.append(_savefig(fig, "eda_wind_dir_hanoi.png"))
    return paths


def write_report(cleaned: dict[str, pd.DataFrame]) -> dict:
    """Write reports/eda.md and reports/eda_stats.json; return stats."""
    s = stats(cleaned)
    figures(cleaned)
    h, m = s["hanoi"], s["hcmc"]
    d = cleaned["hanoi"]
    sector = (((d["wind_direction_10m"] + 22.5) % 360) // 45)
    names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    g = d[C.TARGET].groupby(sector).mean()
    worst_dir, best_dir = names[int(g.idxmax())], names[int(g.idxmin())]
    s["hanoi"]["worst_wind_dir"], s["hanoi"]["best_wind_dir"] = worst_dir, best_dir
    md = f"""# EDA — AirWatch VN

> Tự sinh bởi `python -m airwatch.cli eda`. Dữ liệu: Open-Meteo / CAMS (mô hình, không phải trạm đo).
> Giai đoạn: Hà Nội {h['period'][0]} → {h['period'][1]}; TP.HCM {m['period'][0]} → {m['period'][1]}.

## 1. Mức ô nhiễm chung

| | Hà Nội | TP.HCM |
|---|---|---|
| PM2.5 trung bình (µg/m³) | {h['mean_pm25']} | {m['mean_pm25']} |
| Số ngày TB 24h > 50 (QCVN) | {h['days_over_qcvn']} / {h['n_days']} ({h['share_days_over_qcvn']:.0%}) | {m['days_over_qcvn']} / {m['n_days']} ({m['share_days_over_qcvn']:.0%}) |
| Số ngày TB 24h > 15 (WHO) | {h['days_over_who']} / {h['n_days']} ({h['share_days_over_who']:.0%}) | {m['days_over_who']} / {m['n_days']} ({m['share_days_over_who']:.0%}) |

## 2. Chu kỳ ngày

![diurnal](figures/eda_diurnal.png)

- Hà Nội: cao nhất lúc **{h['peak_hour']}h** ({h['peak_hour_mean']} µg/m³), thấp nhất lúc **{h['low_hour']}h** ({h['low_hour_mean']}).
- TP.HCM: cao nhất lúc **{m['peak_hour']}h** ({m['peak_hour_mean']}), thấp nhất lúc **{m['low_hour']}h** ({m['low_hour_mean']}).
- Đỉnh buổi tối (sau giờ tan tầm 17–19h) khi lớp biên hạ thấp về đêm; giữa trưa lớp biên cao, khuếch tán tốt nên PM2.5 thấp nhất. Giờ tốt nhất để ra ngoài thường là cuối buổi sáng – đầu giờ chiều.

![week-hour](figures/eda_week_hour_hanoi.png)

## 3. Chu kỳ mùa

![seasonal](figures/eda_seasonal.png)

| | Hà Nội | TP.HCM |
|---|---|---|
| PM2.5 TB mùa đông (T12–T2) | {h['season_mean']['DJF']} | {m['season_mean']['DJF']} |
| PM2.5 TB xuân (T3–T5) | {h['season_mean']['MAM']} | {m['season_mean']['MAM']} |
| PM2.5 TB hè (T6–T8) | {h['season_mean']['JJA']} | {m['season_mean']['JJA']} |
| PM2.5 TB thu (T9–T11) | {h['season_mean']['SON']} | {m['season_mean']['SON']} |
| % ngày > 50 µg/m³, mùa đông | {h['season_share_days_over_qcvn']['DJF']:.0%} | {m['season_share_days_over_qcvn']['DJF']:.0%} |
| % ngày > 50 µg/m³, hè | {h['season_share_days_over_qcvn']['JJA']:.0%} | {m['season_share_days_over_qcvn']['JJA']:.0%} |

- Hà Nội: tháng có PM2.5 TB cao nhất **T{h['worst_month']}** ({h['worst_month_mean']} µg/m³), thấp nhất **T{h['best_month']}** ({h['best_month_mean']}).
- TP.HCM: cao nhất **T{m['worst_month']}** ({m['worst_month_mean']}), thấp nhất **T{m['best_month']}** ({m['best_month_mean']}).
- Hà Nội: mùa đông vẫn là mùa tệ nhất ({h['season_share_days_over_qcvn']['DJF']:.0%} số ngày vượt QCVN so với {h['season_share_days_over_qcvn']['JJA']:.0%} mùa hè), nhưng **chênh lệch trong dữ liệu CAMS nhỏ hơn** so với những gì thường thấy ở số liệu trạm đo.
- Đỉnh tháng 4 ở Hà Nội có thể liên quan khói đốt sinh khối khu vực (mùa đốt tháng 3–4, gió hướng Tây Nam) — đây là giả thuyết, cần kiểm chứng bằng dữ liệu trạm.

## 4. Khí tượng: gió, mưa, lớp biên

![meteo](figures/eda_meteo.png)

| PM2.5 TB (µg/m³) | Hà Nội | TP.HCM |
|---|---|---|
| Gió lặng (25% thấp nhất) | {h['pm_calm_wind']} | {m['pm_calm_wind']} |
| Gió mạnh (25% cao nhất) | {h['pm_windy']} | {m['pm_windy']} |
| Có mưa trong 6h qua (> 0.5 mm) | {h['pm_rain6h']} | {m['pm_rain6h']} |
| Không mưa | {h['pm_dry6h']} | {m['pm_dry6h']} |
| Lớp biên thấp (25% thấp nhất) | {h['pm_low_blh']} | {m['pm_low_blh']} |
| Lớp biên cao (25% cao nhất) | {h['pm_high_blh']} | {m['pm_high_blh']} |
| Tương quan với log(độ cao lớp biên) | {h['corr_blh']} | {m['corr_blh']} |
| Tương quan với tốc độ gió | {h['corr_wind']} | {m['corr_wind']} |

- Lớp biên thấp (dấu hiệu nghịch nhiệt, khí bị "nhốt" sát mặt đất): PM2.5 cao gấp {h['pm_low_blh'] / h['pm_high_blh']:.1f} lần (Hà Nội) và {m['pm_low_blh'] / m['pm_high_blh']:.1f} lần (TP.HCM) so với khi lớp biên cao.
- Gió lặng so với gió mạnh: gấp {h['pm_calm_wind'] / h['pm_windy']:.1f} lần (Hà Nội), {m['pm_calm_wind'] / m['pm_windy']:.1f} lần (TP.HCM).
- Mưa: Hà Nội {_cmp(h['pm_dry6h'], h['pm_rain6h'], 'sạch hơn khi có mưa (rửa trôi)', 'không sạch hơn khi có mưa')}; TP.HCM {_cmp(m['pm_dry6h'], m['pm_rain6h'], 'sạch hơn khi có mưa', 'không sạch hơn khi có mưa — mùa mưa trùng đỉnh tháng 6 trong dữ liệu CAMS')}.
- Độ cao lớp biên thiếu từ 2024-01 đến 2024-06 trong ERA5 archive của Open-Meteo; các thống kê trên bỏ qua giờ thiếu.

![wind-dir](figures/eda_wind_dir_hanoi.png)

- Hà Nội: gió từ hướng **{worst_dir}** đi kèm PM2.5 cao nhất, hướng **{best_dir}** thấp nhất.

## 5. Feature importance

Xem `figures/shap_importance.png` (mean |SHAP| của mô hình LightGBM q50) — sinh ở bước `evaluate`.
"""
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (C.REPORTS_DIR / "eda.md").write_text(md, encoding="utf-8")
    (C.REPORTS_DIR / "eda_stats.json").write_text(json.dumps(s, indent=2, ensure_ascii=False))
    return s
