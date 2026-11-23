"""Time-based split, rolling-origin backtest, metrics and figures."""

from __future__ import annotations

import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import brier_score_loss, precision_score, recall_score  # noqa: E402

from airwatch import config as C  # noqa: E402
from airwatch.features import CITY_CODE, supervised  # noqa: E402
from airwatch.models import CLS_DROP, Forecaster, qname, train_classifier, train_quantile  # noqa: E402

log = logging.getLogger(__name__)


# ---------- data assembly ----------

def long_table(origin: pd.DataFrame, cleaned: dict[str, pd.DataFrame], stride: int = 1
               ) -> pd.DataFrame:
    """Supervised long table (origins x horizons) for all cities, rows with known target only."""
    parts = []
    for key, cdf in cleaned.items():
        f = origin[origin["city_key"] == key]
        parts.append(supervised(f, cdf[C.TARGET], stride=stride))
    df = pd.concat(parts)
    return df[df["y"].notna() & df["pm25"].notna()]


def purge_before(long_df: pd.DataFrame, boundary: pd.Timestamp) -> pd.DataFrame:
    """Rows whose *target* lies strictly before ``boundary`` (no overlap with later test)."""
    return long_df[long_df["target_time"] < boundary]


def origins_between(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp | None = None
                    ) -> pd.DataFrame:
    idx = df.index
    m = idx >= start
    if end is not None:
        m &= idx < end
    return df[m]


def origin_purge_before(origin: pd.DataFrame, boundary: pd.Timestamp) -> pd.DataFrame:
    """Origins whose 24h-ahead label window ends before ``boundary``."""
    o = origin[origin["y_mean24"].notna() & origin["pm25"].notna()]
    return o[o.index + pd.Timedelta(hours=24) < boundary]


# ---------- metrics ----------

def _mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def _rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def point_metrics(df: pd.DataFrame, pred_cols: dict[str, str]) -> dict:
    """Overall and per-horizon MAE/RMSE for each method."""
    res: dict = {"overall": {}, "by_horizon": {}}
    for name, col in pred_cols.items():
        d = df[df[col].notna()]
        res["overall"][name] = {"mae": _mae(d["y"], d[col]), "rmse": _rmse(d["y"], d[col]),
                                "n": int(len(d))}
        g = d.groupby("horizon")
        res["by_horizon"][name] = {
            "mae": [round(_mae(x["y"], x[col]), 3) for _, x in g],
            "rmse": [round(_rmse(x["y"], x[col]), 3) for _, x in g],
        }
    return res


def interval_metrics(df: pd.DataFrame) -> dict:
    inside = (df["y"] >= df["q10"]) & (df["y"] <= df["q90"])
    return {
        "coverage_q10_q90": float(inside.mean()),
        "nominal": 0.8,
        "mean_width": float((df["q90"] - df["q10"]).mean()),
        "coverage_by_horizon": [round(float(x), 3) for x in inside.groupby(df["horizon"]).mean()],
    }


def alert_metrics(y_true: np.ndarray, prob: np.ndarray, persist: np.ndarray, base_rate: float
                  ) -> dict:
    """Brier + precision/recall at p>=0.5, vs persistence and climatology."""
    y = y_true.astype(int)
    pred = (prob >= 0.5).astype(int)
    return {
        "event_rate": float(y.mean()),
        "brier_model": float(brier_score_loss(y, prob)),
        "brier_climatology": float(brier_score_loss(y, np.full_like(prob, base_rate))),
        "brier_persistence": float(brier_score_loss(y, persist.astype(float))),
        "precision_model": float(precision_score(y, pred, zero_division=0)),
        "recall_model": float(recall_score(y, pred, zero_division=0)),
        "precision_persistence": float(precision_score(y, persist, zero_division=0)),
        "recall_persistence": float(recall_score(y, persist, zero_division=0)),
        "n": int(len(y)),
    }


# ---------- main evaluation ----------

def run(origin: pd.DataFrame, cleaned: dict[str, pd.DataFrame]) -> tuple[dict, dict[str, int]]:
    """Train/val/test evaluation + rolling-origin backtest. Returns (metrics, best rounds)."""
    val_start, test_start = pd.Timestamp(C.VAL_START), pd.Timestamp(C.TEST_START)
    full = long_table(origin, cleaned)
    train_long = long_table(origin, cleaned, stride=C.TRAIN_ORIGIN_STRIDE)

    # 1) Early stopping on val. Train targets < val_start; val origins in [val, test) with
    #    targets < test_start.
    tr = purge_before(train_long, val_start)
    va = purge_before(origins_between(train_long, val_start), test_start)
    rounds: dict[str, int] = {}
    val_fc = Forecaster()
    for q in C.QUANTILES:
        m = train_quantile(tr, q, val=va)
        val_fc.quantile[qname(q)] = m
        rounds[qname(q)] = int(m.best_iteration or C.NUM_BOOST_ROUND)
        log.info("%s best_iter=%d", qname(q), rounds[qname(q)])
    # Conformalized quantile regression: widen q10/q90 so val coverage hits 80%.
    cqr = conformal_offsets(va, val_fc.predict_quantiles(va))
    o_tr = origin_purge_before(origin, val_start)
    o_va = origin_purge_before(origins_between(origin, val_start), test_start)
    for name, thr in C.THRESHOLDS.items():
        m = train_classifier(o_tr, thr, val=o_va)
        rounds[f"cls_{name}"] = int(m.best_iteration or 300)

    # 2) Refit on train+val (targets < test_start), evaluate on test (every origin hour).
    trv = purge_before(train_long, test_start)
    o_trv = origin_purge_before(origin, test_start)
    fc = Forecaster(cqr=cqr)
    for q in C.QUANTILES:
        fc.quantile[qname(q)] = train_quantile(trv, q, num_round=rounds[qname(q)])
    for name, thr in C.THRESHOLDS.items():
        fc.classifier[name] = train_classifier(o_trv, thr, num_round=rounds[f"cls_{name}"])

    # Same rows for every method: drop the few rows where t+h-24 is missing (seasonal naive).
    test = origins_between(full, test_start)
    test = test[test["base_snaive"].notna()].copy()
    raw_iv = fc.predict_quantiles(test, calibrate=False)
    test[["q10", "q50", "q90"]] = fc.predict_quantiles(test)
    preds = {"persistence": "base_persist", "seasonal_naive": "base_snaive", "lightgbm_q50": "q50"}
    metrics: dict = {
        "split": {"train_end": C.VAL_START, "val": [C.VAL_START, C.TEST_START],
                  "test": [C.TEST_START, str(test["target_time"].max())],
                  "n_test_rows": int(len(test))},
        "rounds": rounds,
        "cqr_offsets": cqr,
        "test": {"all": point_metrics(test, preds)},
        "interval": interval_metrics(test),
        "interval_uncalibrated": interval_metrics(
            test.assign(q10=raw_iv["q10"], q90=raw_iv["q90"])),
        "alerts": {},
    }
    for key in cleaned:
        metrics["test"][key] = point_metrics(test[test["city_key"] == key], preds)
        metrics["interval_" + key] = interval_metrics(test[test["city_key"] == key])

    o_te = origins_between(origin, test_start)
    o_te = o_te[o_te["y_mean24"].notna() & o_te["pm25_mean24"].notna()]
    proba = fc.predict_proba(o_te)
    for name, thr in C.THRESHOLDS.items():
        metrics["alerts"][name] = {"threshold": thr}
        base = float((o_trv["y_mean24"] > thr).mean())
        for key in ["all", *cleaned]:
            sel = np.ones(len(o_te), bool) if key == "all" else (o_te["city_key"] == key).to_numpy()
            metrics["alerts"][name][key] = alert_metrics(
                (o_te["y_mean24"] > thr).to_numpy()[sel], proba[f"p_{name}"].to_numpy()[sel],
                (o_te["pm25_mean24"] > thr).to_numpy().astype(int)[sel], base)

    # 3) Rolling-origin backtest (expanding window, q50 model, fixed rounds).
    metrics["backtest"] = backtest(full, train_long, rounds[qname(0.5)])

    # 4) Figures.
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_horizon(metrics)
    fig_backtest(metrics["backtest"])
    fig_example(test)
    fig_reliability(o_te, proba)
    metrics["feature_importance"] = fig_importance(fc, test.sample(min(20000, len(test)),
                                                                   random_state=C.SEED))
    return metrics, {**rounds, "cqr": cqr}


def conformal_offsets(va: pd.DataFrame, pred: pd.DataFrame, level: float = 0.8
                      ) -> dict[str, list[float]]:
    """Split-conformal offset per (city, horizon): q10 -= Q, q90 += Q (Romano et al. 2019)."""
    score = np.maximum(pred["q10"] - va["y"], va["y"] - pred["q90"])
    out: dict[str, list[float]] = {}
    for key, code in CITY_CODE.items():
        offs = []
        for h in C.HORIZONS:
            s = score[(va["city"] == code) & (va["horizon"] == h)].to_numpy()
            n = len(s)
            q = min(1.0, level * (n + 1) / n) if n else 0.0
            offs.append(round(float(np.quantile(s, q)), 3) if n else 0.0)
        out[key] = offs
    return out


def backtest(full: pd.DataFrame, train_long: pd.DataFrame, rounds: int) -> list[dict]:
    """Expanding-window rolling-origin backtest over blocks of BACKTEST_MONTHS."""
    folds = []
    start = pd.Timestamp(C.BACKTEST_FIRST)
    last = full.index.max()
    while start < last - pd.Timedelta(days=14):
        end = start + pd.DateOffset(months=C.BACKTEST_MONTHS)
        tr = purge_before(train_long, start)
        te = origins_between(full, start, end)
        te = te[(te["target_time"] < end) & te["base_snaive"].notna()]
        m = train_quantile(tr, 0.5, num_round=rounds)
        p = m.predict(te[m.feature_name()])
        folds.append({
            "start": str(start.date()), "end": str(min(end, last).date()), "n": int(len(te)),
            "mae_persistence": _mae(te["y"], te["base_persist"]),
            "mae_seasonal_naive": _mae(te["y"], te["base_snaive"]),
            "mae_lightgbm": _mae(te["y"], p),
            "rmse_persistence": _rmse(te["y"], te["base_persist"]),
            "rmse_seasonal_naive": _rmse(te["y"], te["base_snaive"]),
            "rmse_lightgbm": _rmse(te["y"], p),
        })
        log.info("fold %s: %s", start.date(), folds[-1]["mae_lightgbm"])
        start = end
    return folds


# ---------- figures ----------

COLORS = {"persistence": "#9e9e9e", "seasonal_naive": "#d4a017", "lightgbm_q50": "#1f77b4"}
LABELS = {"persistence": "Persistence", "seasonal_naive": "Seasonal naive (t-24h)",
          "lightgbm_q50": "LightGBM (q50)"}


def fig_horizon(metrics: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, key in zip(axes, ["hanoi", "hcmc"], strict=False):
        bh = metrics["test"][key]["by_horizon"]
        for name in COLORS:
            ax.plot(C.HORIZONS, bh[name]["mae"], label=LABELS[name], color=COLORS[name], lw=2)
        ax.set_title(C.CITIES[key].name_vi)
        ax.set_xlabel("Horizon (h)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("MAE PM2.5 (µg/m³)")
    axes[0].legend()
    fig.suptitle("Test MAE by forecast horizon")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "mae_by_horizon.png", dpi=130)
    plt.close(fig)


def fig_backtest(folds: list[dict]) -> None:
    df = pd.DataFrame(folds)
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(10, 4))
    w = 0.27
    for i, (name, col) in enumerate([("persistence", "mae_persistence"),
                                     ("seasonal_naive", "mae_seasonal_naive"),
                                     ("lightgbm_q50", "mae_lightgbm")]):
        ax.bar(x + (i - 1) * w, df[col], w, label=LABELS[name], color=COLORS[name])
    ax.set_xticks(x, df["start"], rotation=30)
    ax.set_ylabel("MAE (µg/m³), mean over 1–24h")
    ax.set_title("Rolling-origin backtest (expanding window)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "backtest.png", dpi=130)
    plt.close(fig)


def fig_example(test: pd.DataFrame) -> None:
    """Sliding 24h-ahead forecast (h=24) vs actual for Hanoi in the highest-PM test month."""
    d = test[(test["city_key"] == "hanoi") & (test["horizon"] == 24)].set_index("target_time")
    if d.empty:
        return
    month = d["y"].resample("MS").mean().idxmax()
    d = d[(d.index >= month) & (d.index < month + pd.DateOffset(days=21))].sort_index()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(d.index, d["q10"], d["q90"], color="#1f77b4", alpha=0.2, label="q10–q90")
    ax.plot(d.index, d["y"], color="k", lw=1.2, label="Actual (CAMS)")
    ax.plot(d.index, d["q50"], color="#1f77b4", lw=1.5, label="Forecast q50 (24h ahead)")
    ax.axhline(C.THRESHOLDS["qcvn"], color="r", ls="--", lw=1, label="QCVN 50 µg/m³")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_title("Hà Nội — 24h-ahead forecast vs actual (test period)")
    ax.legend(loc="upper left", ncols=4, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "forecast_example.png", dpi=130)
    plt.close(fig)


def fig_reliability(o_te: pd.DataFrame, proba: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    bins = np.linspace(0, 1, 11)
    for name, thr in C.THRESHOLDS.items():
        p = proba[f"p_{name}"].to_numpy()
        y = (o_te["y_mean24"] > thr).to_numpy()
        b = np.clip(np.digitize(p, bins) - 1, 0, 9)
        xs = [p[b == i].mean() for i in range(10) if (b == i).sum() >= 20]
        ys = [y[b == i].mean() for i in range(10) if (b == i).sum() >= 20]
        ax.plot(xs, ys, "o-", label=f"{name.upper()} > {thr:g}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability: 24h-mean exceedance")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "reliability.png", dpi=130)
    plt.close(fig)


def fig_importance(fc: Forecaster, sample: pd.DataFrame, top: int = 15) -> dict[str, float]:
    """Mean |SHAP| (LightGBM TreeSHAP via pred_contrib) of the q50 model."""
    m = fc.quantile["q50"]
    X = sample[m.feature_name()]
    contrib = m.predict(X, pred_contrib=True)[:, :-1]
    imp = pd.Series(np.abs(contrib).mean(0), index=m.feature_name()).sort_values()
    fig, ax = plt.subplots(figsize=(6, 5))
    imp.tail(top).plot.barh(ax=ax, color="#1f77b4")
    ax.set_xlabel("mean |SHAP| (µg/m³)")
    ax.set_title("Feature importance — LightGBM q50")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "shap_importance.png", dpi=130)
    plt.close(fig)
    return {k: round(float(v), 3) for k, v in imp.sort_values(ascending=False).head(top).items()}


def save_metrics(metrics: dict) -> None:
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    C.METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))


__all__ = ["run", "save_metrics", "CLS_DROP"]
