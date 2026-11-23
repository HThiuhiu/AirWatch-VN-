"""LightGBM quantile forecaster (horizon as a feature) and exceedance classifiers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from airwatch import config as C
from airwatch.features import CITY_CODE, feature_cols

CLS_DROP = {"horizon", "target_hour", "target_hour_sin", "target_hour_cos"}


def _qname(q: float) -> str:
    return f"q{int(round(q * 100)):02d}"


def _dataset(df: pd.DataFrame, cols: list[str], label: str) -> lgb.Dataset:
    return lgb.Dataset(df[cols], label=df[label], categorical_feature=["city"],
                       free_raw_data=False)


def train_quantile(train: pd.DataFrame, q: float, val: pd.DataFrame | None = None,
                   num_round: int | None = None) -> lgb.Booster:
    """Train one quantile model. Early-stops on ``val`` if given, else uses ``num_round``."""
    cols = feature_cols(train)
    params = {**C.LGB_PARAMS, "objective": "quantile", "alpha": q, "metric": "quantile"}
    dtr = _dataset(train, cols, "y")
    if val is not None:
        dva = _dataset(val, cols, "y")
        return lgb.train(params, dtr, C.NUM_BOOST_ROUND, valid_sets=[dva],
                         callbacks=[lgb.early_stopping(C.EARLY_STOPPING, verbose=False)])
    return lgb.train(params, dtr, num_round or 500)


def train_classifier(train: pd.DataFrame, threshold: float, val: pd.DataFrame | None = None,
                     num_round: int | None = None) -> lgb.Booster:
    """Binary model for P(mean PM2.5 over next 24h > threshold)."""
    cols = [c for c in feature_cols(train) if c not in CLS_DROP]
    tr = train.assign(lbl=(train["y_mean24"] > threshold).astype(int))
    params = {**C.LGB_PARAMS, "objective": "binary", "metric": "binary_logloss"}
    dtr = _dataset(tr, cols, "lbl")
    if val is not None:
        va = val.assign(lbl=(val["y_mean24"] > threshold).astype(int))
        return lgb.train(params, dtr, C.NUM_BOOST_ROUND, valid_sets=[_dataset(va, cols, "lbl")],
                         callbacks=[lgb.early_stopping(C.EARLY_STOPPING, verbose=False)])
    return lgb.train(params, dtr, num_round or 300)


@dataclass
class Forecaster:
    """Bundle of quantile and exceedance models."""

    quantile: dict[str, lgb.Booster] = field(default_factory=dict)
    classifier: dict[str, lgb.Booster] = field(default_factory=dict)
    cqr: dict[str, list[float]] = field(default_factory=dict)  # conformal offsets per city/h

    def predict_quantiles(self, X: pd.DataFrame, calibrate: bool = True) -> pd.DataFrame:
        """Predict q10/q50/q90 (sorted so they never cross), with optional conformal widening."""
        out = pd.DataFrame(index=X.index)
        for name, m in self.quantile.items():
            out[name] = m.predict(X[m.feature_name()], num_iteration=m.best_iteration or None)
        names = sorted(self.quantile)
        out[names] = np.sort(out[names].to_numpy(), axis=1)
        if calibrate and self.cqr:
            off = np.zeros(len(X))
            for key, code in CITY_CODE.items():
                if key in self.cqr:
                    sel = (X["city"] == code).to_numpy()
                    off[sel] = np.asarray(self.cqr[key])[X["horizon"].to_numpy()[sel] - 1]
            out["q10"] -= off
            out["q90"] += off
        out[names] = out[names].clip(lower=0)
        return out

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for name, m in self.classifier.items():
            out[f"p_{name}"] = m.predict(X[m.feature_name()],
                                         num_iteration=m.best_iteration or None)
        return out

    def save(self, path=C.MODEL_DIR) -> None:
        path.mkdir(parents=True, exist_ok=True)
        meta = {"quantile": list(self.quantile), "classifier": list(self.classifier),
                "cqr": self.cqr}
        models = {**self.quantile, **{f"cls_{k}": v for k, v in self.classifier.items()}}
        for n, m in models.items():
            m.save_model(str(path / f"{n}.txt"))
        (path / "meta.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, path=C.MODEL_DIR) -> Forecaster:
        meta = json.loads((path / "meta.json").read_text())
        return cls(
            quantile={n: lgb.Booster(model_file=str(path / f"{n}.txt")) for n in meta["quantile"]},
            classifier={n: lgb.Booster(model_file=str(path / f"cls_{n}.txt"))
                        for n in meta["classifier"]},
            cqr=meta.get("cqr", {}),
        )


def fit_forecaster(train: pd.DataFrame, origin_train: pd.DataFrame,
                   rounds: dict) -> Forecaster:
    """Fit all models with fixed boosting rounds (found earlier by early stopping)."""
    fc = Forecaster(cqr=rounds.get("cqr", {}))
    for q in C.QUANTILES:
        fc.quantile[_qname(q)] = train_quantile(train, q, num_round=rounds[_qname(q)])
    for name, thr in C.THRESHOLDS.items():
        fc.classifier[name] = train_classifier(origin_train, thr, num_round=rounds[f"cls_{name}"])
    return fc


def qname(q: float) -> str:
    """Public name for a quantile, e.g. 0.1 -> 'q10'."""
    return _qname(q)
