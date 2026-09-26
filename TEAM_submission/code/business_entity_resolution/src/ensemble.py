"""Weighted probability ensemble + optional calibration.

    P_final = w1 * P_LGBM + w2 * P_XGB + w3 * P_RF,   w >= 0, sum(w) = 1

Equal weights are the starting point; every weight vector on a simplex grid
(step 0.1 -> 66 combinations) is evaluated and the one with the best validation
F0.5 (at its own best threshold) is kept.

Calibration (Platt / isotonic, fitted per model on the *inner* hold-out) is
only retained if it improves validation F0.5 by more than ``MIN_GAIN``.
"""
from __future__ import annotations

import itertools
import json
import logging

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from config import Config
from evaluation import GroupEvaluator
from models import ENSEMBLE_MODELS
from threshold_optimization import best_threshold, search_rule

log = logging.getLogger("ber.ensemble")
MIN_GAIN = 0.001


def weight_grid(step: float, n: int = 3) -> list[tuple[float, ...]]:
    k = int(round(1 / step))
    out = [tuple(round(c / k, 4) for c in combo) for combo in itertools.product(range(k + 1), repeat=n)
           if sum(combo) == k]
    eq = tuple([round(1 / n, 4)] * n)
    return [eq] + out     # equal weights evaluated first (baseline)


def combine(probs: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    return sum(weights[m] * probs[m] for m in ENSEMBLE_MODELS).astype(np.float32)


class Calibrator:
    def __init__(self, method: str):
        self.method, self.models = method, {}

    def fit(self, probs: dict[str, np.ndarray], y: np.ndarray) -> "Calibrator":
        for m in ENSEMBLE_MODELS:
            p = np.clip(probs[m], 1e-6, 1 - 1e-6)
            if self.method == "platt":
                self.models[m] = LogisticRegression().fit(np.log(p / (1 - p)).reshape(-1, 1), y)
            elif self.method == "isotonic":
                self.models[m] = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y)
        return self

    def transform(self, probs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self.method == "none":
            return probs
        out = dict(probs)
        for m in ENSEMBLE_MODELS:
            p = np.clip(probs[m], 1e-6, 1 - 1e-6)
            if self.method == "platt":
                out[m] = self.models[m].predict_proba(np.log(p / (1 - p)).reshape(-1, 1))[:, 1].astype(np.float32)
            else:
                out[m] = self.models[m].predict(p).astype(np.float32)
        return out


def search_weights(probs: dict[str, np.ndarray], ev: GroupEvaluator, cfg: Config) -> tuple[dict, list[dict]]:
    rows, best = [], None
    for w in weight_grid(cfg.model.ensemble_weight_step):
        wd = dict(zip(ENSEMBLE_MODELS, w))
        t, m = best_threshold(combine(probs, wd), ev, cfg)
        rows.append({**{f"w_{k}": v for k, v in wd.items()}, "threshold": t, **m})
        if best is None or m["f0_5"] > best[1]["f0_5"] + 1e-9:
            best = (wd, {"threshold": t, **m})
    eq = rows[0]
    log.info("Equal weights F0.5=%.4f | best weights %s F0.5=%.4f", eq["f0_5"], best[0], best[1]["f0_5"])
    return best[0], rows


def select_ensemble(val_probs: dict, ev: GroupEvaluator, inner_probs: dict, inner_y: np.ndarray,
                    cfg: Config) -> dict:
    """Choose calibration, weights and decision rule on validation F0.5."""
    methods = ["none", "platt", "isotonic"] if cfg.model.calibration == "auto" else [cfg.model.calibration]
    if "none" not in methods:
        methods = ["none"] + methods
    results = {}
    for meth in methods:
        cal = Calibrator(meth).fit(inner_probs, inner_y) if meth != "none" else Calibrator("none")
        vp = cal.transform(val_probs)
        weights, rows = search_weights(vp, ev, cfg)
        rule, rule_rows = search_rule(combine(vp, weights), ev, cfg)
        results[meth] = {"calibrator": cal, "weights": weights, "rule": rule, "weight_rows": rows,
                         "rule_rows": rule_rows}
        log.info("Calibration=%-8s weights=%s F0.5=%.4f (P=%.4f R=%.4f)", meth, weights, rule["f0_5"],
                 rule["precision"], rule["recall"])
    chosen = "none"
    for meth in methods:
        if results[meth]["rule"]["f0_5"] > results[chosen]["rule"]["f0_5"] + MIN_GAIN:
            chosen = meth
    res = results[chosen]
    log.info("Selected calibration: %s", chosen)
    joblib.dump({"calibrator": res["calibrator"], "weights": res["weights"], "calibration": chosen},
                cfg.path("models", "ensemble.joblib"))
    summary = {m: {"weights": r["weights"], "f0_5": r["rule"]["f0_5"], "precision": r["rule"]["precision"],
                   "recall": r["rule"]["recall"]} for m, r in results.items()}
    (cfg.report_dir / "ensemble_selection.json").write_text(json.dumps(
        {"chosen_calibration": chosen, "weights": res["weights"], "by_calibration": summary}, indent=2))
    import polars as pl
    pl.DataFrame(res["weight_rows"]).write_csv(cfg.report_dir / "ensemble_weight_search.tsv", separator="\t")
    return {"calibration": chosen, **res}


def load_ensemble(cfg: Config) -> dict:
    return joblib.load(cfg.art / "models" / "ensemble.joblib")
