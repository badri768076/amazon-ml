"""Candidate generation: union of all blocking strategies + fuzzy fallback + capping.

Output (checkpointed) per split in ``artifacts/candidates/``:
  <split>_s1.parquet     normalised Source-1 records with integer ``idx``
  <split>_cand.parquet   normalised Source-2 + Source-3 records with ``idx`` and ``source``
  <split>_pairs.parquet  final candidate pairs: s1_idx, cand_idx, source, blk_score,
                         blk_bits, blk_n_strat, blk_rank  -- exactly what the model scores
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from blocking import STRATEGY_BIT, STRATEGY_SPECS, build_index, query_index
from config import Config
from preprocessing import load_normalized

log = logging.getLogger("ber.candidates")


# --------------------------------------------------------------------------- #
# Entity tables
# --------------------------------------------------------------------------- #
def entity_tables(cfg: Config, split: str, s1_ids: list[str] | None = None,
                  write: bool = True) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Source-1 frame and combined S2+S3 candidate frame, each with an int ``idx``."""
    d = cfg.art / "candidates"
    p1, pc = d / f"{split}_s1.parquet", d / f"{split}_cand.parquet"
    if p1.exists() and pc.exists() and not cfg.force:
        return pl.read_parquet(p1), pl.read_parquet(pc)
    d.mkdir(parents=True, exist_ok=True)
    s1 = load_normalized(cfg, f"{split}_source1")
    if s1_ids is not None:
        s1 = s1.filter(pl.col("entity_id").is_in(s1_ids))
    s1 = s1.sort("entity_id").with_row_index("idx")
    cand = pl.concat([
        load_normalized(cfg, f"{split}_source2").with_columns(source=pl.lit(2, pl.Int8)),
        load_normalized(cfg, f"{split}_source3").with_columns(source=pl.lit(3, pl.Int8)),
    ]).with_row_index("idx")
    if write:
        s1.write_parquet(p1)
        cand.write_parquet(pc)
    return s1, cand


# --------------------------------------------------------------------------- #
# Fuzzy retrieval (TF-IDF char n-gram sparse index + RapidFuzz re-rank)
# --------------------------------------------------------------------------- #
class FuzzyIndex:
    """Country-partitioned sparse TF-IDF index over one candidate source."""

    def __init__(self, cand: pl.DataFrame, source: int, cfg: Config, cache_dir: Path):
        b = cfg.blocking
        sub = cand.filter(pl.col("source") == source).select("idx", "name_core", "normalized_country")
        self.cand_idx = sub["idx"].to_numpy()
        self.names = sub["name_core"].fill_null("").to_list()
        countries = sub["normalized_country"].fill_null("").to_numpy()
        mpath, vpath = cache_dir / f"fuzzy_s{source}.npz", cache_dir / f"fuzzy_s{source}_vec.joblib"
        if mpath.exists() and vpath.exists() and not cfg.force:
            self.vec, self.X = joblib.load(vpath), sparse.load_npz(mpath).tocsr()
        else:
            # absolute document-frequency cap keeps posting lists (and SpMM cost) bounded
            max_df = b.fuzzy_max_postings if len(self.names) > b.fuzzy_max_postings else 1.0
            self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2, max_df=max_df,
                                       max_features=150_000, dtype=np.float32, sublinear_tf=True)
            self.X = self.vec.fit_transform(self.names).tocsr() if self.names else sparse.csr_matrix((0, 1))
            sparse.save_npz(mpath, self.X)
            joblib.dump(self.vec, vpath)
        import gc; gc.collect()
        self.partitions: dict[str, np.ndarray] = {}
        null_rows = np.flatnonzero(countries == "")
        for c in np.unique(countries):
            if c:
                self.partitions[c] = np.union1d(np.flatnonzero(countries == c), null_rows)
        self.all_rows = np.arange(len(self.names))
        self._sub_cache: dict[str, sparse.csr_matrix] = {}

    def _submatrix(self, country: str):
        if country not in self._sub_cache:
            rows = self.partitions.get(country, self.all_rows) if country else self.all_rows
            self._sub_cache[country] = (rows, self.X[rows].T.tocsr())
        return self._sub_cache[country]

    def query(self, s1_idx: np.ndarray, names: list[str], countries: list[str | None], cfg: Config):
        b = cfg.blocking
        out = []
        if self.X.shape[0] == 0:
            return out
        by_country: dict[str, list[int]] = {}
        for i, c in enumerate(countries):
            by_country.setdefault(c or "", []).append(i)
        for c, pos in by_country.items():
            rows, XT = self._submatrix(c)
            for start in range(0, len(pos), b.fuzzy_query_batch):
                sel = pos[start:start + b.fuzzy_query_batch]
                Q = self.vec.transform([names[i] for i in sel]).astype(np.float32)
                S = (Q @ XT).tocsr()
                for r, i in enumerate(sel):
                    lo, hi = S.indptr[r], S.indptr[r + 1]
                    if hi == lo:
                        continue
                    data, cols = S.data[lo:hi], S.indices[lo:hi]
                    keep = data >= b.fuzzy_min_score
                    data, cols = data[keep], cols[keep]
                    if data.size == 0:
                        continue
                    top = np.argsort(-data, kind="stable")[: b.fuzzy_top_k_retrieve]
                    cand_rows = rows[cols[top]]
                    rf = process.cpdist([names[i]] * len(cand_rows), [self.names[j] for j in cand_rows],
                                        scorer=fuzz.token_set_ratio, workers=1) / 100.0
                    comb = 0.5 * data[top] + 0.5 * rf
                    best = np.argsort(-comb, kind="stable")[: b.fuzzy_top_k_keep]
                    for j in best:
                        out.append((int(s1_idx[i]), int(self.cand_idx[cand_rows[j]]), float(comb[j])))
        return out


# --------------------------------------------------------------------------- #
# Main entry
# --------------------------------------------------------------------------- #
def generate_candidates(cfg: Config, split: str, s1_ids: list[str] | None = None,
                        gt: pl.DataFrame | None = None) -> Path:
    """Run every blocking strategy, union, fuzzy fallback, cap. Checkpointed."""
    out_path = cfg.art / "candidates" / f"{split}_pairs.parquet"
    if out_path.exists() and not cfg.force:
        log.info("[checkpoint] %s candidates exist -> %s", split, out_path.name)
        return out_path
    b = cfg.blocking
    s1, cand = entity_tables(cfg, split, s1_ids)
    log.info("Blocking %s: |S1|=%d |S2+S3|=%d", split, s1.height, cand.height)
    parts_dir = cfg.art / "candidates" / f"_parts_{split}"
    if cfg.force:
        shutil.rmtree(parts_dir, ignore_errors=True)
    parts_dir.mkdir(parents=True, exist_ok=True)
    n_chunks = max(1, int(np.ceil(s1.height / b.s1_chunk_size)))
    chunks = [s1.slice(i * b.s1_chunk_size, b.s1_chunk_size) for i in range(n_chunks)]

    # 1) key-based strategies: build one index at a time, query every chunk
    for spec in STRATEGY_SPECS:
        if hasattr(b, "enabled_strategies") and spec.name not in b.enabled_strategies:
            log.info("  skipping strategy %s (disabled in config)", spec.name)
            continue
        all_chunks_exist = all((parts_dir / f"c{ci:05d}_{spec.name}.parquet").exists() for ci in range(n_chunks))
        if all_chunks_exist and not cfg.force:
            log.info("  [checkpoint] index %s chunks already computed -> skipping", spec.name)
            continue
        index = build_index(cand, spec, cfg)
        for ci, ch in enumerate(chunks):
            part_path = parts_dir / f"c{ci:05d}_{spec.name}.parquet"
            if part_path.exists() and not cfg.force:
                continue
            p = query_index(ch, index, cfg)
            if p.height:
                p.write_parquet(part_path)
        del index
        import gc; gc.collect()

    # 2) per chunk: union, fuzzy fallback, cap
    fuzzy = {}
    if b.fuzzy_enabled:
        cache = cfg.art / "candidates" / f"_index_{split}"
        cache.mkdir(parents=True, exist_ok=True)
        for src in (2, 3):
            fuzzy[src] = FuzzyIndex(cand, src, cfg, cache)
    cand_src = cand.select(pl.col("idx").alias("cand_idx"), "source")
    cand_text = cand.select(pl.col("idx").alias("cand_idx"), pl.col("name_core").alias("c_name"),
                            pl.col("normalized_address").alias("c_addr"), pl.col("primary_postal").alias("c_pin"),
                            pl.col("building_number").alias("c_bno"))
    gt_idx = _gt_to_idx(gt, s1, cand) if gt is not None else None
    stats = {"pre_cap_pairs": 0, "pre_cap_gt_hits": 0, "fuzzy_pairs": 0, "fuzzy_queries": 0}
    capped_parts_dir = cfg.art / "candidates" / f"_capped_parts_{split}"
    shutil.rmtree(capped_parts_dir, ignore_errors=True)
    capped_parts_dir.mkdir(parents=True, exist_ok=True)
    for ci, ch in enumerate(tqdm(chunks, desc=f"union {split}", disable=n_chunks < 3)):
        files = sorted(parts_dir.glob(f"c{ci:05d}_*.parquet"))
        pairs = (pl.concat([pl.read_parquet(f) for f in files]) if files else
                 pl.DataFrame(schema={"s1_idx": pl.UInt32, "cand_idx": pl.UInt32,
                                      "contrib": pl.Float32, "bits": pl.Int32}))
        pairs = (pairs.group_by("s1_idx", "cand_idx")
                      .agg(blk_score=pl.col("contrib").sum(), blk_bits=pl.col("bits").bitwise_or())
                      .join(cand_src, on="cand_idx", how="left"))
        if fuzzy:
            counts = pairs.group_by("s1_idx", "source").len()
            for src, fidx in fuzzy.items():
                have = counts.filter((pl.col("source") == src) & (pl.col("len") >= b.fuzzy_min_candidates))
                need = ch.filter(~pl.col("idx").is_in(have["s1_idx"]))
                if need.height == 0:
                    continue
                stats["fuzzy_queries"] += need.height
                res = fidx.query(need["idx"].to_numpy(), need["name_core"].fill_null("").to_list(),
                                 need["normalized_country"].to_list(), cfg)
                if res:
                    fp = pl.DataFrame(res, schema={"s1_idx": pl.UInt32, "cand_idx": pl.UInt32,
                                                   "fz": pl.Float32}, orient="row")
                    fp = fp.with_columns(blk_score=(b.weights["fuzzy"] * 10 * pl.col("fz")).cast(pl.Float32),
                                         blk_bits=pl.lit(STRATEGY_BIT["fuzzy"], pl.Int32),
                                         source=pl.lit(src, pl.Int8)).drop("fz")
                    stats["fuzzy_pairs"] += fp.height
                    pairs = (pl.concat([pairs.select(fp.columns), fp])
                               .group_by("s1_idx", "cand_idx", "source")
                               .agg(pl.col("blk_score").sum(), pl.col("blk_bits").bitwise_or()))
        stats["pre_cap_pairs"] += pairs.height
        if gt_idx is not None:
            stats["pre_cap_gt_hits"] += pairs.join(gt_idx, on=["s1_idx", "cand_idx"], how="semi").height
        pairs = _rerank_and_cap(pairs, ch, cand_text, cfg)
        pairs.write_parquet(capped_parts_dir / f"cap_{ci:05d}.parquet")
        del pairs
        import gc; gc.collect()

    capped_files = sorted(capped_parts_dir.glob("cap_*.parquet"))
    final = (pl.read_parquet(capped_files) if capped_files else
             pl.DataFrame(schema={"s1_idx": pl.UInt32, "cand_idx": pl.UInt32, "source": pl.Int8,
                                  "blk_score": pl.Float32, "blk_bits": pl.Int32, "blk_quick": pl.Float32,
                                  "blk_rank": pl.UInt16}))
    final = final.with_columns(
        blk_n_strat=pl.col("blk_bits").map_batches(lambda s: _popcount(s), return_dtype=pl.Int8),
        s1_idx=pl.col("s1_idx").cast(pl.UInt32), cand_idx=pl.col("cand_idx").cast(pl.UInt32),
    ).sort("s1_idx", "source", "blk_rank")
    final.write_parquet(out_path)
    shutil.rmtree(parts_dir, ignore_errors=True)
    shutil.rmtree(capped_parts_dir, ignore_errors=True)
    stats.update({"split": split, "final_pairs": final.height})
    (cfg.report_dir / f"blocking_stats_{split}.json").write_text(json.dumps(stats, indent=2))
    log.info("Candidates %s: pre-cap pairs=%d, final pairs=%d, fuzzy queries=%d (+%d pairs)",
             split, stats["pre_cap_pairs"], final.height, stats["fuzzy_queries"], stats["fuzzy_pairs"])
    return out_path


def _rerank_and_cap(pairs: pl.DataFrame, s1_chunk: pl.DataFrame, cand_text: pl.DataFrame,
                    cfg: Config) -> pl.DataFrame:
    """Stage 1: keep top ``pre_rank_keep`` per (S1, source) by blocking score.
    Stage 2: re-rank with a cheap name/address similarity and keep the final cap.

    ``blk_rank`` is the final rank (1 = best) inside the (S1, source) group.
    """
    b = cfg.blocking
    # sort first so every ordinal rank below has a deterministic tie-break (cand_idx)
    pairs = (pairs.sort(["s1_idx", "source", "blk_score", "cand_idx"], descending=[False, False, True, False])
                  .with_columns(r0=pl.int_range(pl.len()).over("s1_idx", "source"))
                  .filter(pl.col("r0") < b.pre_rank_keep).drop("r0"))
    if pairs.height == 0:
        return pairs.with_columns(blk_quick=pl.lit(0.0, pl.Float32), blk_rank=pl.lit(0, pl.UInt16))
    t = (pairs.join(s1_chunk.select(pl.col("idx").alias("s1_idx"), pl.col("name_core").alias("s_name"),
                                    pl.col("normalized_address").alias("s_addr"),
                                    pl.col("primary_postal").alias("s_pin"), pl.col("building_number").alias("s_bno")),
                    on="s1_idx", how="left")
              .join(cand_text, on="cand_idx", how="left"))
    sn, cn = t["s_name"].fill_null("").to_list(), t["c_name"].fill_null("").to_list()
    sa, ca = t["s_addr"].fill_null("").to_list(), t["c_addr"].fill_null("").to_list()
    w = cfg.n_workers
    name_set = process.cpdist(sn, cn, scorer=fuzz.token_set_ratio, workers=w) / 100.0
    name_r = process.cpdist(sn, cn, scorer=fuzz.ratio, workers=w) / 100.0
    addr_set = process.cpdist(sa, ca, scorer=fuzz.token_set_ratio, workers=w) / 100.0
    addr_r = process.cpdist(sa, ca, scorer=fuzz.ratio, workers=w) / 100.0
    pin = (t["s_pin"] == t["c_pin"]).fill_null(False).to_numpy().astype(np.float32)
    bno = (t["s_bno"] == t["c_bno"]).fill_null(False).to_numpy().astype(np.float32)
    blk_norm = np.minimum(t["blk_score"].to_numpy() / 60.0, 1.0)
    # address evidence carries real weight so that chains (same name, many branches) keep the right branch
    quick = (0.35 * name_set + 0.15 * name_r + 0.20 * addr_set + 0.10 * addr_r + 0.10 * pin + 0.05 * bno
             + 0.05 * blk_norm).astype(np.float32)
    return (pairs.with_columns(blk_quick=pl.Series(quick))
                 .sort(["s1_idx", "source", "blk_quick", "cand_idx"], descending=[False, False, True, False])
                 .with_columns(blk_rank=(pl.int_range(pl.len()).over("s1_idx", "source") + 1).cast(pl.UInt16))
                 .filter(pl.col("blk_rank") <= b.max_candidates_per_source))


def _popcount(s: pl.Series) -> pl.Series:
    a = s.to_numpy().astype(np.uint32)
    cnt = np.zeros_like(a, dtype=np.int8)
    for i in range(16):
        cnt += ((a >> i) & 1).astype(np.int8)
    return pl.Series(cnt)


def _gt_to_idx(gt_pairs: pl.DataFrame, s1: pl.DataFrame, cand: pl.DataFrame) -> pl.DataFrame:
    """(s1_id, cand_id) -> (s1_idx, cand_idx) restricted to entities in scope."""
    return (gt_pairs.join(s1.select(pl.col("entity_id").alias("s1_id"), pl.col("idx").alias("s1_idx")),
                          on="s1_id", how="inner")
                    .join(cand.select(pl.col("entity_id").alias("cand_id"), pl.col("idx").alias("cand_idx")),
                          on="cand_id", how="inner")
                    .select("s1_idx", "cand_idx"))


# --------------------------------------------------------------------------- #
# Candidate recall
# --------------------------------------------------------------------------- #
def candidate_recall(cfg: Config, split: str, gt_pairs: pl.DataFrame) -> dict:
    """Blocking quality on labelled data: recall, candidate-set sizes, reduction ratio."""
    s1, cand = entity_tables(cfg, split)
    pairs = pl.read_parquet(cfg.art / "candidates" / f"{split}_pairs.parquet")
    gt_all = gt_pairs.join(s1.select(pl.col("entity_id").alias("s1_id")), on="s1_id", how="semi")
    gt_idx = _gt_to_idx(gt_all, s1, cand)
    hit = gt_idx.join(pairs, on=["s1_idx", "cand_idx"], how="inner")
    n_true = gt_all.height
    per_s1 = s1.select(pl.col("idx").alias("s1_idx")).join(
        pairs.group_by("s1_idx").len(), on="s1_idx", how="left").fill_null(0)["len"]
    by_source = {}
    for src in (2, 3):
        tot = gt_all.filter(pl.col("cand_id").str.starts_with(f"S{src}-")).height
        got = hit.filter(pl.col("source") == src).height
        by_source[f"S{src}"] = {"true": tot, "retrieved": got, "recall": got / tot if tot else None}
    only = {}
    for name, bit in STRATEGY_BIT.items():
        with_bit = hit.filter((pl.col("blk_bits") & bit) > 0).height
        only_bit = hit.filter(pl.col("blk_bits") == bit).height
        only[name] = {"hits": with_bit, "unique_hits": only_bit}
    stats_path = cfg.report_dir / f"blocking_stats_{split}.json"
    pre = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    rep = {
        "split": split,
        "true_matches": n_true,
        "retrieved": hit.height,
        "candidate_recall": hit.height / n_true if n_true else None,
        "pre_cap_recall": (pre.get("pre_cap_gt_hits", 0) / n_true) if n_true and pre.get("pre_cap_gt_hits") else None,
        "by_source": by_source,
        "by_strategy": only,
        "avg_candidates_per_s1": float(per_s1.mean()),
        "median_candidates_per_s1": float(per_s1.median()),
        "max_candidates_per_s1": int(per_s1.max()),
        "s1_with_zero_candidates": int((per_s1 == 0).sum()),
        "total_pairs": pairs.height,
        "cartesian_pairs": s1.height * cand.height,
        "reduction_ratio": 1.0 - pairs.height / max(1, s1.height * cand.height),
    }
    (cfg.report_dir / f"candidate_recall_{split}.json").write_text(json.dumps(rep, indent=2))
    log.info("Candidate recall (%s): %.4f (%d/%d) | S2 %.4f S3 %.4f | avg %.1f med %.0f max %d | "
             "reduction %.6f", split, rep["candidate_recall"] or 0, hit.height, n_true,
             by_source["S2"]["recall"] or 0, by_source["S3"]["recall"] or 0,
             rep["avg_candidates_per_s1"], rep["median_candidates_per_s1"],
             rep["max_candidates_per_s1"], rep["reduction_ratio"])
    return rep
