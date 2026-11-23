"""Command line: python -m airwatch.cli {fetch,eda,evaluate,train,forecast,all}."""

from __future__ import annotations

import argparse
import json
import logging
import random

import numpy as np
import pandas as pd

from airwatch import config as C
from airwatch import fetch as F
from airwatch.alerts import aqi_level, nowcast, pm25_to_aqi
from airwatch.features import build_all, save_processed, supervised

log = logging.getLogger("airwatch")


def now_local() -> pd.Timestamp:
    """Current local hour (naive, UTC+7)."""
    return pd.Timestamp.now(tz=C.TIMEZONE).tz_localize(None).floor("h")


def _data() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    raw = {k: F.load_city(k) for k in C.CITIES}
    origin, cleaned = build_all(raw, now=now_local())
    save_processed(origin)
    return origin, cleaned


def cmd_fetch(refresh: bool) -> None:
    for k, df in F.fetch_all(refresh=refresh).items():
        log.info("%s: %d rows, %s -> %s", k, len(df), df.index.min(), df.index.max())


def cmd_eda() -> None:
    from airwatch.eda import write_report

    _, cleaned = _data()
    write_report(cleaned)
    log.info("wrote reports/eda.md")


def cmd_evaluate() -> None:
    from airwatch.evaluate import run, save_metrics

    origin, cleaned = _data()
    metrics, rounds = run(origin, cleaned)
    save_metrics(metrics)
    C.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (C.MODEL_DIR / "rounds.json").write_text(json.dumps(rounds))
    o = metrics["test"]["all"]["overall"]
    log.info("test MAE: persistence %.2f | seasonal naive %.2f | LightGBM %.2f",
             o["persistence"]["mae"], o["seasonal_naive"]["mae"], o["lightgbm_q50"]["mae"])


def _rounds() -> dict[str, int]:
    for p in [C.MODEL_DIR / "rounds.json", C.METRICS_PATH]:
        if p.exists():
            js = json.loads(p.read_text(encoding="utf-8"))
            return js.get("rounds", js)
    return {"q10": 400, "q50": 400, "q90": 400, **{f"cls_{k}": 200 for k in C.THRESHOLDS}}


def cmd_train() -> None:
    from airwatch.evaluate import long_table, origin_purge_before
    from airwatch.models import fit_forecaster

    origin, cleaned = _data()
    train = long_table(origin, cleaned, stride=C.TRAIN_ORIGIN_STRIDE)
    o_train = origin_purge_before(origin, now_local() + pd.Timedelta(hours=1))
    fc = fit_forecaster(train, o_train, _rounds())
    fc.save()
    log.info("saved models to %s (%d train rows)", C.MODEL_DIR, len(train))


def cmd_forecast() -> None:
    from airwatch.models import Forecaster

    origin, cleaned = _data()
    fc = Forecaster.load()
    C.FORECAST_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict = {"generated_at": str(pd.Timestamp.now(tz=C.TIMEZONE).floor("min")),
                     "data_source": "Open-Meteo Air Quality (CAMS model data)", "cities": {}}
    for key, cdf in cleaned.items():
        f = origin[origin["city_key"] == key]
        f = f[f["pm25"].notna()]
        t0 = f.index.max()
        X = supervised(f.loc[[t0]], cdf[C.TARGET])
        q = fc.predict_quantiles(X)
        out = pd.DataFrame({"time": X["target_time"].to_numpy(), "horizon": X["horizon"].to_numpy(),
                            **{c: q[c].round(1).to_numpy() for c in q}})
        out["aqi_q50"] = [pm25_to_aqi(v) for v in out["q50"]]
        out.to_csv(C.FORECAST_DIR / f"{key}_forecast.csv", index=False)
        hist = cdf.loc[cdf.index > t0 - pd.Timedelta(days=7), [C.TARGET, "us_aqi"]].round(1)
        hist.reset_index().to_csv(C.FORECAST_DIR / f"{key}_history.csv", index=False)
        p = fc.predict_proba(f.loc[[t0]]).iloc[0]
        nc = nowcast(cdf.loc[:t0, C.TARGET].tail(12).to_numpy())
        aqi = pm25_to_aqi(nc)
        lv = aqi_level(aqi)
        summary["cities"][key] = {
            "origin": str(t0), "pm25_now": round(float(f.loc[t0, "pm25"]), 1),
            "pm25_nowcast": round(nc, 1), "aqi_now": aqi, "level": lv.key if lv else None,
            "pm25_mean24_past": round(float(f.loc[t0, "pm25_mean24"]), 1),
            "forecast_mean24_q50": round(float(out["q50"].mean()), 1),
            "p_exceed": {k: round(float(p[f"p_{k}"]), 3) for k in C.THRESHOLDS},
        }
        log.info("%s: origin %s, P(>50)=%.2f", key, t0, p["p_qcvn"])
    (C.FORECAST_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                                 encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    random.seed(C.SEED)
    np.random.seed(C.SEED)
    ap = argparse.ArgumentParser(prog="airwatch")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pf = sub.add_parser("fetch", help="download data (uses cache unless --refresh)")
    pf.add_argument("--refresh", action="store_true", help="update the last days from the API")
    for name in ["eda", "evaluate", "train", "forecast", "all"]:
        sub.add_parser(name)
    a = ap.parse_args(argv)
    if a.cmd == "fetch":
        cmd_fetch(a.refresh)
    elif a.cmd == "eda":
        cmd_eda()
    elif a.cmd == "evaluate":
        cmd_evaluate()
    elif a.cmd == "train":
        cmd_train()
    elif a.cmd == "forecast":
        cmd_forecast()
    elif a.cmd == "all":
        cmd_fetch(False)
        cmd_eda()
        cmd_evaluate()
        cmd_train()
        cmd_forecast()


if __name__ == "__main__":
    main()
