"""Labelled pair construction with Source-1 entity-level splitting.

* Positives  : candidate pairs that appear in ``train_ground_truth`` (S1<->S2, S1<->S3).
* Negatives  : *hard* negatives only - non-matching pairs that blocking itself
               retrieved (same country / similar name / similar address / shared
               postal code / shared tokens). The hardest ``neg_hard_per_s1`` (by
               blocking rank) are kept plus ``neg_random_per_s1`` random in-block
               negatives per S1 for the training split. No random cross-product negatives.
* Splitting  : by Source-1 entity (80/20 by default) *before* any pair is built, so
               no Source-1 entity of the validation split contributes to training.
               A further inner hold-out (10% of training entities) is used for early
               stopping and calibration so validation stays untouched until selection.
* Validation : keeps the *complete* candidate set, so validation metrics reflect the
               real decision problem, and counts ground-truth matches missed by
               blocking as false negatives.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import polars as pl

from config import Config
from data_loader import gt_pairs, load, parse_gt, resolve_gt_policy

log = logging.getLogger("ber.pairs")


def select_train_entities(cfg: Config) -> tuple[list[str], pl.DataFrame]:
    """Labelled Source-1 entities in scope + exploded GT pairs (s1_id, cand_id)."""
    s1_ids = load(cfg, "train_source1")["entity_id"]
    gt = load(cfg, "train_ground_truth")
    policy = resolve_gt_policy(cfg, gt)
    gt_ids = parse_gt(gt)["source1_entity_id"]
    if policy == "exclude":
        labelled = s1_ids.filter(s1_ids.is_in(gt_ids))
    else:
        labelled = s1_ids
    log.info("GT policy '%s': %d labelled train Source-1 entities (of %d)", policy, labelled.len(), s1_ids.len())
    labelled = labelled.sort()
    if cfg.max_train_s1 and labelled.len() > cfg.max_train_s1:
        labelled = labelled.sample(cfg.max_train_s1, seed=cfg.seed).sort()
        log.info("Sub-sampled to max_train_s1=%d entities", cfg.max_train_s1)
    pairs = gt_pairs(gt).filter(pl.col("s1_id").is_in(labelled))
    return labelled.to_list(), pairs


def split_entities(ids: list[str], cfg: Config) -> dict[str, str]:
    """Deterministic S1-level split -> {entity_id: 'train'|'inner'|'val'}."""
    rng = np.random.default_rng(cfg.seed)
    ids = np.array(sorted(ids))
    perm = rng.permutation(len(ids))
    n_val = int(round(len(ids) * cfg.val_ratio))
    n_inner = int(round((len(ids) - n_val) * cfg.inner_holdout_ratio))
    role = np.empty(len(ids), dtype=object)
    role[perm[:n_val]] = "val"
    role[perm[n_val:n_val + n_inner]] = "inner"
    role[perm[n_val + n_inner:]] = "train"
    return dict(zip(ids.tolist(), role.tolist()))


def label_features(feats: pl.DataFrame, s1: pl.DataFrame, cand: pl.DataFrame,
                   gtp: pl.DataFrame) -> pl.DataFrame:
    """Attach entity ids + binary label to a feature frame."""
    return (feats.join(s1.select(pl.col("idx").alias("s1_idx"), pl.col("entity_id").alias("s1_id")),
                       on="s1_idx", how="left")
                 .join(cand.select(pl.col("idx").alias("cand_idx"), pl.col("entity_id").alias("cand_id")),
                       on="cand_idx", how="left")
                 .join(gtp.with_columns(label=pl.lit(1, pl.Int8)), on=["s1_id", "cand_id"], how="left")
                 .with_columns(pl.col("label").fill_null(0)))


def sample_negatives(df: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    """Keep all positives, the hardest negatives and a few random in-block negatives per S1."""
    df = df.sort("s1_idx", "cand_idx")
    neg = (df.filter(pl.col("label") == 0)
             .sort(["s1_idx", "blk_quick", "cand_idx"], descending=[False, True, False])
             .with_columns(hr=pl.int_range(pl.len()).over("s1_idx")))
    hard = neg.filter(pl.col("hr") < cfg.neg_hard_per_s1)
    rest = (neg.filter(pl.col("hr") >= cfg.neg_hard_per_s1).sort("s1_idx", "cand_idx")
               .with_columns(rr=pl.int_range(pl.len()).shuffle(seed=cfg.seed).over("s1_idx"))
               .filter(pl.col("rr") < cfg.neg_random_per_s1).drop("rr"))
    out = pl.concat([df.filter(pl.col("label") == 1), hard.drop("hr"), rest.drop("hr")], how="diagonal")
    return out.sort("s1_idx", "cand_idx")


def build_training_sets(cfg: Config, feat_dir, s1: pl.DataFrame, cand: pl.DataFrame,
                        gtp: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Returns {'train','inner','val'} labelled frames + writes checkpoint & report.

    Feature parts are processed one at a time (each holds complete Source-1 groups),
    so only the down-sampled training pairs and the validation pairs are kept in memory.
    """
    from pathlib import Path
    roles = split_entities(s1["entity_id"].to_list(), cfg)
    role_df = pl.DataFrame({"s1_id": list(roles), "role": list(roles.values())})
    acc: dict[str, list] = {"train": [], "inner": [], "val": []}
    for part in sorted(Path(feat_dir).glob("part-*.parquet")):
        lab = label_features(pl.read_parquet(part), s1, cand, gtp).join(role_df, on="s1_id", how="left")
        for r in acc:
            sub = lab.filter(pl.col("role") == r).drop("role")
            acc[r].append(sample_negatives(sub, cfg) if r == "train" else sub)
    out = {}
    for r, parts in acc.items():
        out[r] = pl.concat(parts).sort("s1_idx", "cand_idx")
        out[r].write_parquet(cfg.path("training", f"{r}_pairs.parquet"))
    # ground truth of val / inner entities (incl. pairs missed by blocking) for honest metrics
    for r in ("inner", "val"):
        g = gtp.join(role_df.filter(pl.col("role") == r), on="s1_id", how="inner").select("s1_id", "cand_id")
        g.write_parquet(cfg.path("training", f"{r}_gt.parquet"))
        role_df.filter(pl.col("role") == r).select("s1_id").write_parquet(cfg.path("training", f"{r}_entities.parquet"))
    rep = {r: {"pairs": out[r].height, "positives": int(out[r]["label"].sum()),
               "negatives": int((out[r]["label"] == 0).sum()),
               "s1_entities": int(sum(1 for v in roles.values() if v == r))} for r in out}
    (cfg.report_dir / "training_pairs.json").write_text(json.dumps(rep, indent=2))
    for r, v in rep.items():
        log.info("Pairs %-5s: entities=%d pairs=%d pos=%d neg=%d", r, v["s1_entities"], v["pairs"],
                 v["positives"], v["negatives"])
    # leakage guard
    tr, va = set(out["train"]["s1_id"].unique()), set(out["val"]["s1_id"].unique())
    assert not tr & va, "Source-1 leakage between train and validation"
    return out
