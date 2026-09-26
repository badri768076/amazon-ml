"""Pair classifiers: Logistic Regression baseline, LightGBM, XGBoost, Random Forest.

All four are trained on the same training pairs and evaluated with the same
entity-level validation protocol. LightGBM / XGBoost early-stop on the inner
hold-out (never on validation). Models are checkpointed with joblib.
"""
from __future__ import annotations

import json
import logging
import time

import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import Config, detect_device
from feature_engineering import feature_columns

log = logging.getLogger("ber.models")

MODEL_NAMES = ["lr", "lgbm", "xgb", "rf"]
ENSEMBLE_MODELS = ["lgbm", "xgb", "rf"]


def to_xy(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
    X = df.select(feature_columns()).to_numpy().astype(np.float32)
    y = df["label"].to_numpy().astype(np.int8) if "label" in df.columns else None
    return X, y


def _fill(X: np.ndarray) -> np.ndarray:
    """NaN -> -1 for models without native missing-value handling."""
    return np.where(np.isnan(X), -1.0, X).astype(np.float32)


class PairModels:
    """Holds the four fitted models + their best iteration counts."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.models: dict = {}
        self.best_iter: dict = {}

    # ------------------------------------------------------------------ fit
    def fit(self, X: np.ndarray, y: np.ndarray, X_es: np.ndarray | None = None, y_es: np.ndarray | None = None,
            fixed_iters: dict | None = None, which: list[str] | None = None) -> "PairModels":
        m, seed, nw = self.cfg.model, self.cfg.seed, self.cfg.n_workers
        which = which or MODEL_NAMES
        log.info("Training on %d pairs (%d positives, %.2f%%)", len(y), int(y.sum()), 100 * y.mean())

        if "lr" in which:
            t = time.time()
            self.models["lr"] = make_pipeline(
                StandardScaler(), LogisticRegression(C=m.lr_C, max_iter=m.lr_max_iter, random_state=seed))
            self.models["lr"].fit(_fill(X), y)
            log.info("  Logistic Regression baseline trained (%.1fs)", time.time() - t)

        if "lgbm" in which:
            t = time.time()
            params = dict(m.lgbm_params, seed=seed, num_threads=nw, deterministic=True, force_row_wise=True)
            dtr = lgb.Dataset(X, y, free_raw_data=False)
            if fixed_iters:
                self.models["lgbm"] = lgb.train(params, dtr, num_boost_round=fixed_iters["lgbm"])
                self.best_iter["lgbm"] = fixed_iters["lgbm"]
            else:
                des = lgb.Dataset(X_es, y_es, reference=dtr)
                bst = lgb.train(params, dtr, num_boost_round=m.lgbm_rounds, valid_sets=[des],
                                callbacks=[lgb.early_stopping(m.early_stopping_rounds, verbose=False)])
                self.models["lgbm"], self.best_iter["lgbm"] = bst, bst.best_iteration or m.lgbm_rounds
            log.info("  LightGBM trained: %d rounds (%.1fs)", self.best_iter["lgbm"], time.time() - t)

        if "xgb" in which:
            t = time.time()
            dev = getattr(m, "device", "auto")
            if dev == "auto":
                dev = detect_device()
            params = dict(m.xgb_params, seed=seed)
            if dev == "cuda":
                params["device"] = "cuda"
                params["tree_method"] = "hist"
                log.info("  Training XGBoost on GPU (CUDA - RTX 4060)...")
            else:
                params["device"] = "cpu"
                params["nthread"] = nw
                log.info("  Training XGBoost on CPU (%d workers)...", nw)

            dtr = xgb.DMatrix(X, label=y, missing=np.nan)
            try:
                if fixed_iters:
                    self.models["xgb"] = xgb.train(params, dtr, num_boost_round=fixed_iters["xgb"])
                    self.best_iter["xgb"] = fixed_iters["xgb"]
                else:
                    des = xgb.DMatrix(X_es, label=y_es, missing=np.nan)
                    bst = xgb.train(params, dtr, num_boost_round=m.xgb_rounds, evals=[(des, "es")],
                                    early_stopping_rounds=m.early_stopping_rounds, verbose_eval=False)
                    self.models["xgb"], self.best_iter["xgb"] = bst, bst.best_iteration + 1
                log.info("  XGBoost trained: %d rounds (%.1fs)", self.best_iter["xgb"], time.time() - t)
            except Exception as exc:
                if params.get("device") == "cuda":
                    log.warning("XGBoost GPU failed (%s). Retrying on CPU fallback...", exc)
                    params["device"] = "cpu"
                    params["nthread"] = nw
                    if fixed_iters:
                        self.models["xgb"] = xgb.train(params, dtr, num_boost_round=fixed_iters["xgb"])
                        self.best_iter["xgb"] = fixed_iters["xgb"]
                    else:
                        des = xgb.DMatrix(X_es, label=y_es, missing=np.nan)
                        bst = xgb.train(params, dtr, num_boost_round=m.xgb_rounds, evals=[(des, "es")],
                                        early_stopping_rounds=m.early_stopping_rounds, verbose_eval=False)
                        self.models["xgb"], self.best_iter["xgb"] = bst, bst.best_iteration + 1
                    log.info("  XGBoost trained on CPU fallback: %d rounds (%.1fs)", self.best_iter["xgb"], time.time() - t)
                else:
                    raise

        if "rf" in which:
            t = time.time()
            Xr, yr = X, y
            if len(y) > m.rf_max_train_rows:
                idx = np.random.default_rng(seed).choice(len(y), m.rf_max_train_rows, replace=False)
                Xr, yr = X[idx], y[idx]
            self.models["rf"] = RandomForestClassifier(**m.rf_params, n_jobs=nw, random_state=seed)
            self.models["rf"].fit(_fill(Xr), yr)
            log.info("  Random Forest trained on %d rows (%.1fs)", len(yr), time.time() - t)
        return self

    # -------------------------------------------------------------- predict
    def predict(self, X: np.ndarray, which: list[str] | None = None) -> dict[str, np.ndarray]:
        out = {}
        for name in which or list(self.models):
            mdl = self.models[name]
            if name == "lgbm":
                p = mdl.predict(X, num_iteration=self.best_iter.get("lgbm"))
            elif name == "xgb":
                try:
                    p = mdl.predict(xgb.DMatrix(X, missing=np.nan),
                                    iteration_range=(0, self.best_iter.get("xgb", 0)))
                except Exception:
                    p = mdl.predict(xgb.DMatrix(X, missing=np.nan))
            else:
                p = mdl.predict_proba(_fill(X))[:, 1]
            out[name] = np.asarray(p, dtype=np.float32)
        return out

    # ----------------------------------------------------------- persist
    def save(self, tag: str) -> None:
        joblib.dump({"models": self.models, "best_iter": self.best_iter}, self.cfg.path("models", f"{tag}.joblib"))

    @classmethod
    def load(cls, cfg: Config, tag: str) -> "PairModels":
        obj = cls(cfg)
        d = joblib.load(cfg.art / "models" / f"{tag}.joblib")
        obj.models, obj.best_iter = d["models"], d["best_iter"]
        return obj

    def feature_importance(self) -> dict:
        cols = feature_columns()
        imp = {}
        if "lgbm" in self.models:
            g = self.models["lgbm"].feature_importance("gain")
            imp["lgbm_gain"] = dict(sorted(zip(cols, map(float, g)), key=lambda x: -x[1])[:30])
        if "rf" in self.models:
            imp["rf"] = dict(sorted(zip(cols, map(float, self.models["rf"].feature_importances_)),
                                    key=lambda x: -x[1])[:30])
        if "lr" in self.models:
            coef = self.models["lr"][-1].coef_[0]
            imp["lr_coef"] = dict(sorted(zip(cols, map(float, coef)), key=lambda x: -abs(x[1]))[:30])
        return imp


def save_importance(cfg: Config, models: PairModels) -> None:
    (cfg.report_dir / "feature_importance.json").write_text(json.dumps(models.feature_importance(), indent=2))
