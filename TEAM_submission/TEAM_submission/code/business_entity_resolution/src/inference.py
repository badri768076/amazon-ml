"""Final model refit and test inference.

1. (optional) refit LightGBM / XGBoost / RF on *all* labelled train entities with
   the iteration counts found by early stopping (scaled for the larger set),
2. build features for every test candidate pair (the exact pair set in
   candidate_pairs.tsv),
3. P_LGBM, P_XGB, P_RF -> calibration (if selected) -> weighted P_final,
4. apply the selected entity-level decision rule (threshold / relative / cap).
"""
from __future__ import annotations

import logging

import numpy as np
import polars as pl

from candidate_generation import entity_tables
from config import Config
from ensemble import combine, load_ensemble
from feature_engineering import build_feature_matrix
from models import ENSEMBLE_MODELS, PairModels, to_xy
from threshold_optimization import apply_rule, load_rule
from training_pairs import sample_negatives

log = logging.getLogger("ber.inference")


def score(models: PairModels, ens: dict, df: pl.DataFrame) -> tuple[np.ndarray, dict]:
    X, _ = to_xy(df)
    raw = models.predict(X, which=ENSEMBLE_MODELS)
    probs = ens["calibrator"].transform(raw)
    return combine(probs, ens["weights"]), raw


def final_models(cfg: Config) -> PairModels:
    """Refit on train+inner+val when configured (and when no calibration is used)."""
    tag = "final"
    path = cfg.art / "models" / f"{tag}.joblib"
    if path.exists() and not cfg.force:
        log.info("[checkpoint] final models exist")
        return PairModels.load(cfg, tag)
    selected = PairModels.load(cfg, "selection")
    ens = load_ensemble(cfg)
    if not cfg.model.refit_on_full or ens["calibration"] != "none":
        log.info("Using selection-stage models for test (refit_on_full=%s, calibration=%s)",
                 cfg.model.refit_on_full, ens["calibration"])
        selected.save(tag)
        return selected
    parts = [pl.read_parquet(cfg.art / "training" / f"{r}_pairs.parquet") for r in ("train", "inner", "val")]
    full = pl.concat([parts[0], sample_negatives(pl.concat([parts[1], parts[2]], how="diagonal"), cfg)],
                     how="diagonal")
    X, y = to_xy(full)
    frac = len(y) / max(1, len(to_xy(parts[0])[1]))
    iters = {m: max(10, int(round(selected.best_iter[m] * min(1.3, frac ** 0.5)))) for m in ("lgbm", "xgb")}
    log.info("Refitting on all labelled entities: %d pairs, iterations %s", len(y), iters)
    models = PairModels(cfg).fit(X, y, fixed_iters=iters, which=ENSEMBLE_MODELS)
    models.save(tag)
    return models


def predict_test(cfg: Config) -> pl.DataFrame:
    out = cfg.art / "predictions" / "test_scored.parquet"
    if out.exists() and not cfg.force:
        log.info("[checkpoint] test predictions exist")
        return pl.read_parquet(out)
    s1, cand = entity_tables(cfg, "test")
    pairs = pl.read_parquet(cfg.art / "candidates" / "test_pairs.parquet")
    feat_dir = build_feature_matrix(cfg, "test", pairs, s1, cand)
    models, ens, rule = final_models(cfg), load_ensemble(cfg), load_rule(cfg)
    import shutil
    scored_parts_dir = cfg.art / "predictions" / "_parts_test"
    if cfg.force:
        shutil.rmtree(scored_parts_dir, ignore_errors=True)
    scored_parts_dir.mkdir(parents=True, exist_ok=True)
    for pi, part in enumerate(sorted(feat_dir.glob("part-*.parquet"))):
        part_path = scored_parts_dir / f"scored_{pi:05d}.parquet"
        if part_path.exists() and not cfg.force:
            continue
        df = pl.read_parquet(part)
        p, raw = score(models, ens, df)
        sub = df.select("s1_idx", "cand_idx", "source").with_columns(
            p_final=pl.Series(p), **{f"p_{m}": pl.Series(v) for m, v in raw.items()})
        sub = apply_rule(sub, "p_final", rule)
        scored_parts_dir.mkdir(parents=True, exist_ok=True)
        sub.write_parquet(part_path)
        del df, sub, p, raw
        import gc; gc.collect()

    scored_files = sorted(scored_parts_dir.glob("scored_*.parquet"))
    res = pl.read_parquet(scored_files) if scored_files else pl.DataFrame(
        schema={"s1_idx": pl.UInt32, "cand_idx": pl.UInt32, "source": pl.Int8, "p_final": pl.Float32, "pred": pl.Boolean})
    res = (res.join(s1.select(pl.col("idx").alias("s1_idx"), pl.col("entity_id").alias("s1_id")), on="s1_idx")
              .join(cand.select(pl.col("idx").alias("cand_idx"), pl.col("entity_id").alias("cand_id")), on="cand_idx"))
    out.parent.mkdir(parents=True, exist_ok=True)
    res.write_parquet(out)
    shutil.rmtree(scored_parts_dir, ignore_errors=True)
    n_s1 = s1.height
    matched_s1 = res.filter(pl.col("pred"))["s1_id"].n_unique()
    log.info("Test inference: %d pairs scored, %d matches, %d/%d Source-1 with >=1 match, %d singletons "
             "(no match), avg %.3f matches/S1", res.height, int(res["pred"].sum()), matched_s1, n_s1,
             n_s1 - matched_s1, res["pred"].sum() / max(1, n_s1))
    return res
