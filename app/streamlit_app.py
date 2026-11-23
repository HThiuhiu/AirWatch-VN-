"""AirWatch VN — Streamlit dashboard (Vietnamese UI).

Reads only small committed files (data/forecast, reports), so it runs on Streamlit Cloud
without training models or calling the API.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from airwatch import config as C  # noqa: E402
from airwatch.alerts import LEVELS, advice_table, aqi_level, best_window, pm25_to_aqi  # noqa: E402

st.set_page_config(page_title="AirWatch VN", page_icon="🌫️", layout="wide")


# ---------- data ----------

def _forecast_dir() -> Path:
    d = C.FORECAST_DIR
    return d if (d / "summary.json").exists() else C.SAMPLE_DIR / "forecast"


@st.cache_data(ttl=600)
def load(city: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    d = _forecast_dir()
    summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    fc = pd.read_csv(d / f"{city}_forecast.csv", parse_dates=["time"])
    hist = pd.read_csv(d / f"{city}_history.csv", parse_dates=["time"])
    return summary, fc, hist


@st.cache_data
def load_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def text_color(bg: str) -> str:
    return "#111" if bg in ("#ffff00", "#00e400", "#ff7e00") else "#fff"


def card(title: str, value: str, sub: str, bg: str) -> str:
    return (f"<div style='background:{bg};color:{text_color(bg)};padding:16px 18px;"
            f"border-radius:12px;height:100%'><div style='font-size:14px;opacity:.85'>{title}</div>"
            f"<div style='font-size:34px;font-weight:700;line-height:1.2'>{value}</div>"
            f"<div style='font-size:14px'>{sub}</div></div>")


def prob_color(p: float) -> str:
    return "#2e7d32" if p < 0.3 else ("#f9a825" if p < 0.6 else "#c62828")


# ---------- sidebar ----------

st.sidebar.title("🌫️ AirWatch VN")
city = st.sidebar.radio("Chọn thành phố", list(C.CITIES), format_func=lambda k: C.CITIES[k].name_vi)
summary, fc, hist = load(city)
s = summary["cities"][city]
st.sidebar.caption(f"Cập nhật: {summary['generated_at']}")
st.sidebar.caption(
    "Nguồn: Open-Meteo Air Quality API — dữ liệu **mô hình** CAMS (Copernicus), "
    "không phải trạm quan trắc. Dự báo: LightGBM (AirWatch VN)."
)
st.sidebar.caption(
    f"Ngưỡng PM2.5 TB 24h: QCVN 05:2023/BTNMT = {C.THRESHOLDS['qcvn']:g} µg/m³; "
    f"WHO 2021 = {C.THRESHOLDS['who']:g} µg/m³."
)

st.title(f"Chất lượng không khí — {C.CITIES[city].name_vi}")
tab_now, tab_model, tab_insight = st.tabs(["📍 Hiện tại & dự báo 24h", "📊 Mô hình", "🔎 Insight"])

# ---------- tab 1 ----------
with tab_now:
    lv = aqi_level(s["aqi_now"])
    peak_aqi = int(fc["aqi_q50"].max())
    peak_lv = aqi_level(peak_aqi)
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(card("AQI hiện tại (US EPA, NowCast)", str(s["aqi_now"]), lv.name_vi, lv.color),
                unsafe_allow_html=True)
    c2.markdown(card("PM2.5 giờ gần nhất", f"{s['pm25_now']:.0f}",
                     f"µg/m³ · TB 24h qua: {s['pm25_mean24_past']:.0f}", "#37474f"),
                unsafe_allow_html=True)
    c3.markdown(card("Dự báo TB 24h tới", f"{s['forecast_mean24_q50']:.0f}",
                     f"µg/m³ · đỉnh AQI dự báo: {peak_aqi} ({peak_lv.name_vi})", peak_lv.color),
                unsafe_allow_html=True)
    p50 = s["p_exceed"]["qcvn"]
    c4.markdown(card("⚠️ Cảnh báo sớm", f"{p50:.0%}",
                     f"P(TB 24h tới > {C.THRESHOLDS['qcvn']:g} µg/m³ — QCVN)<br>"
                     f"P(> {C.THRESHOLDS['who']:g} — WHO): {s['p_exceed']['who']:.0%}",
                     prob_color(p50)), unsafe_allow_html=True)
    st.write("")

    # Chart: 7 days history + 24h forecast band
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fc["time"], y=fc["q90"], line=dict(width=0), showlegend=False,
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fc["time"], y=fc["q10"], fill="tonexty", line=dict(width=0),
                             fillcolor="rgba(31,119,180,0.25)", name="Khoảng 80% (q10–q90)"))
    fig.add_trace(go.Scatter(x=hist["time"], y=hist["pm2_5"], name="7 ngày qua",
                             line=dict(color="#455a64", width=1.8)))
    fig.add_trace(go.Scatter(x=fc["time"], y=fc["q50"], name="Dự báo (trung vị)",
                             line=dict(color="#1f77b4", width=2.5)))
    for thr, name, col in [(C.THRESHOLDS["qcvn"], "QCVN 50", "#c62828"),
                           (C.THRESHOLDS["who"], "WHO 15", "#2e7d32")]:
        fig.add_hline(y=thr, line=dict(color=col, dash="dash", width=1),
                      annotation_text=name, annotation_position="top left")
    fig.add_vline(x=pd.Timestamp(s["origin"]), line=dict(color="#999", dash="dot"))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10),
                      yaxis_title="PM2.5 (µg/m³)", hovermode="x unified",
                      legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, width="stretch")
    st.caption("Ngưỡng QCVN/WHO áp dụng cho trung bình 24 giờ; đường giờ chỉ để tham chiếu.")

    left, right = st.columns([1, 1.4])
    with left:
        st.subheader("🕒 Khung giờ tốt nhất để ra ngoài")
        bw = best_window(fc["time"], fc["q50"], hours=2)
        if bw:
            t, v = bw
            a = pm25_to_aqi(v)
            st.success(f"**{t:%H:%M}–{t + pd.Timedelta(hours=2):%H:%M}, ngày {t:%d/%m}** — "
                       f"PM2.5 dự báo ~{v:.0f} µg/m³ (AQI ~{a}, {aqi_level(a).name_vi}).")
        worst = fc.loc[fc["q50"].idxmax()]
        st.warning(f"Tránh: khoảng **{worst['time']:%H:%M} ngày {worst['time']:%d/%m}** — "
                   f"PM2.5 dự báo cao nhất ~{worst['q50']:.0f} µg/m³.")
        st.caption("Chỉ xét 6:00–21:00, dựa trên dự báo trung vị 24h tới.")
    with right:
        st.subheader("🩺 Khuyến nghị sức khỏe")
        options = {lv_.key: lv_.name_vi for lv_ in LEVELS}
        default = peak_lv.key
        pick = st.selectbox("Mức AQI", list(options), index=list(options).index(default),
                            format_func=lambda k: f"{options[k]}"
                            + (" — mức cao nhất 24h tới" if k == default else ""))
        st.dataframe(advice_table(pick), width="stretch")
        st.caption("⚠️ Thông tin **tham khảo**, không thay thế tư vấn y tế. Người có bệnh nền "
                   "nên làm theo hướng dẫn của bác sĩ.")

# ---------- tab 2 ----------
with tab_model:
    m = load_json(C.METRICS_PATH)
    if not m:
        st.info("Chưa có reports/metrics.json — chạy `python -m airwatch.cli evaluate`.")
    else:
        st.markdown(f"**Tập test:** {m['split']['test'][0]} → {m['split']['test'][1][:10]} "
                    f"(mọi giờ, horizon 1–24h; huấn luyện chỉ dùng dữ liệu trước "
                    f"{m['split']['test'][0]}).")
        rows = []
        for key in C.CITIES:
            for name, lab in [("persistence", "Persistence"),
                              ("seasonal_naive", "Seasonal naive (t−24h)"),
                              ("lightgbm_q50", "LightGBM q50")]:
                o = m["test"][key]["overall"][name]
                rows.append({"Thành phố": C.CITIES[key].name_vi, "Mô hình": lab,
                             "MAE": round(o["mae"], 2), "RMSE": round(o["rmse"], 2)})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        c1, c2 = st.columns(2)
        c1.metric("Độ phủ khoảng q10–q90 (mục tiêu 80%)",
                  f"{m['interval']['coverage_q10_q90']:.1%}",
                  f"trước hiệu chỉnh: {m['interval_uncalibrated']['coverage_q10_q90']:.1%}",
                  delta_color="off")
        c2.metric("Độ rộng TB khoảng 80%", f"{m['interval']['mean_width']:.1f} µg/m³")
        arows = []
        for name, a in m["alerts"].items():
            for key in C.CITIES:
                x = a[key]
                arows.append({
                    "Ngưỡng": f"{name.upper()} > {a['threshold']:g}",
                    "Thành phố": C.CITIES[key].name_vi, "Tỷ lệ sự kiện": f"{x['event_rate']:.0%}",
                    "Brier (mô hình)": round(x["brier_model"], 3),
                    "Brier (persistence)": round(x["brier_persistence"], 3),
                    "Brier (khí hậu)": round(x["brier_climatology"], 3),
                    "Precision": round(x["precision_model"], 2),
                    "Recall": round(x["recall_model"], 2)})
        st.subheader("Cảnh báo vượt ngưỡng (TB 24h tới, p ≥ 0.5)")
        st.dataframe(pd.DataFrame(arows), width="stretch", hide_index=True)
        for f, cap in [("mae_by_horizon.png", "MAE theo horizon"),
                       ("backtest.png", "Rolling-origin backtest"),
                       ("forecast_example.png", "Ví dụ dự báo 24h trước"),
                       ("reliability.png", "Độ tin cậy xác suất"),
                       ("shap_importance.png", "Feature importance (SHAP)")]:
            if (C.FIG_DIR / f).exists():
                st.image(str(C.FIG_DIR / f), caption=cap)

# ---------- tab 3 ----------
with tab_insight:
    e = load_json(C.REPORTS_DIR / "eda_stats.json")
    if e:
        x = e[city]
        c1, c2, c3 = st.columns(3)
        c1.metric("PM2.5 trung bình", f"{x['mean_pm25']} µg/m³")
        c2.metric("Ngày vượt QCVN (TB 24h > 50)", f"{x['share_days_over_qcvn']:.0%}")
        c3.metric("Ngày vượt WHO (TB 24h > 15)", f"{x['share_days_over_who']:.0%}")
        st.markdown(
            f"- Giờ ô nhiễm nhất: **{x['peak_hour']}h**; sạch nhất: **{x['low_hour']}h**.\n"
            f"- Lớp biên thấp: PM2.5 TB **{x['pm_low_blh']}** vs "
            f"lớp biên cao **{x['pm_high_blh']}**.\n"
            f"- Gió lặng: **{x['pm_calm_wind']}** vs gió mạnh **{x['pm_windy']}** µg/m³.\n"
            f"- Có mưa 6h qua: **{x['pm_rain6h']}** vs không mưa **{x['pm_dry6h']}** µg/m³."
        )
    for f in ["eda_diurnal.png", "eda_seasonal.png", "eda_meteo.png", "eda_week_hour_hanoi.png"]:
        if (C.FIG_DIR / f).exists():
            st.image(str(C.FIG_DIR / f))
    st.caption("Chi tiết: reports/eda.md")
