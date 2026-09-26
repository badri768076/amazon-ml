"""Entity-level decision rule selected by validation F0.5.

Decision rule for a Source-1 entity with candidate probabilities p_1..p_n:

    keep candidate j  iff  p_j >= t                       (absolute threshold)
                     and  p_j >= r * max_k p_k            (relative-to-best ratio)
                     and  rank_j <= K   (K = 0: unlimited) (optional cap)

* Every candidate satisfying the rule is kept -> multiple matches are natural.
* If none satisfies it, the entity gets an empty match list -> singletons are natural.
* t = 0.5 is never assumed; t, r and K are grid-searched on validation F0.5.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import polars as pl

from config import Config
from evaluation import GroupEvaluator

log = logging.getLogger("ber.threshold")


def decision_mask(p: np.ndarray, t: float, rel: float = 0.0, max_k: int = 0,
                  grp_max: np.ndarray | None = None, ranks: np.ndarray | None = None) -> np.ndarray:
    m = p >= t
    if rel > 0 and grp_max is not None:
        m &= p >= rel * grp_max
    if max_k and ranks is not None:
        m &= ranks <= max_k
    return m


def threshold_table(p: np.ndarray, ev: GroupEvaluator, cfg: Config, rel: float = 0.0, max_k: int = 0) -> list[dict]:
    gm = ev.group_max(p) if rel > 0 else None
    rk = ev.group_rank(p) if max_k else None
    rows = []
    for t in cfg.thresholds:
        m = ev.evaluate(decision_mask(p, t, rel, max_k, gm, rk))
        rows.append({"threshold": t, **m})
    return rows


def best_threshold(p: np.ndarray, ev: GroupEvaluator, cfg: Config) -> tuple[float, dict]:
    """Fast search over the absolute threshold only (used inside weight search)."""
    best_t, best = None, {"f0_5": -1}
    for t in cfg.thresholds:
        m = ev.evaluate(p >= t)
        if m["f0_5"] > best["f0_5"]:
            best_t, best = t, m
    return best_t, best


def search_rule(p: np.ndarray, ev: GroupEvaluator, cfg: Config) -> tuple[dict, list[dict]]:
    """Full grid over (t, r, K); returns the best rule and all evaluated rows."""
    gm, rk = ev.group_max(p), ev.group_rank(p)
    rows, best = [], None
    for rel in cfg.relative_ratios:
        for k in cfg.max_matches_options:
            for t in cfg.thresholds:
                m = ev.evaluate(decision_mask(p, t, rel, k, gm, rk))
                row = {"threshold": t, "relative_ratio": rel, "max_matches": k, **m}
                rows.append(row)
                # tie-break: higher F0.5, then higher precision, then simpler rule
                key = (round(m["f0_5"], 6), round(m["precision"], 6), -rel, -(k or 0))
                if best is None or key > best[0]:
                    best = (key, row)
    return best[1], rows


def apply_rule(df: pl.DataFrame, prob_col: str, rule: dict) -> pl.DataFrame:
    """Apply the selected rule to a scored pair frame (grouped by ``s1_idx``)."""
    p = pl.col(prob_col)
    cond = p >= rule["threshold"]
    if rule.get("relative_ratio", 0) > 0:
        cond &= p >= rule["relative_ratio"] * p.max().over("s1_idx")
    if rule.get("max_matches", 0):
        cond &= p.rank("ordinal", descending=True).over("s1_idx") <= rule["max_matches"]
    return df.with_columns(pred=cond)


def save_rule(cfg: Config, rule: dict, table: list[dict]) -> None:
    (cfg.art / "models").mkdir(parents=True, exist_ok=True)
    (cfg.art / "models" / "decision_rule.json").write_text(json.dumps(rule, indent=2))
    pl.DataFrame(table).write_csv(cfg.report_dir / "threshold_search.tsv", separator="\t")
    log.info("Selected rule: t=%.2f r=%.2f K=%s -> P=%.4f R=%.4f F0.5=%.4f singleton=%.4f",
             rule["threshold"], rule.get("relative_ratio", 0), rule.get("max_matches", 0), rule["precision"],
             rule["recall"], rule["f0_5"], rule["singleton_correctness"] or float("nan"))


def load_rule(cfg: Config) -> dict:
    return json.loads((cfg.art / "models" / "decision_rule.json").read_text())
