"""Pairwise similarity features for (Source-1, candidate) pairs.

String similarities are computed with RapidFuzz ``cpdist`` (vectorised,
multi-threaded, C++), token-set features with Polars list operations, and
TF-IDF cosines with sparse row-wise dot products. Work is chunked by Source-1
ranges so per-group context features (rank within the candidate set) see the
complete candidate list of every Source-1 entity.

Entity IDs are never used as features.
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Jaro, JaroWinkler, Levenshtein, Prefix
from sklearn.feature_extraction.text import TfidfVectorizer

from blocking import STRATEGY_BIT
from config import Config

log = logging.getLogger("ber.features")

TEXT_COLS = ["idx", "normalized_name", "name_core", "name_acronym", "name_tokens", "core_tokens",
             "normalized_address", "address_tokens", "address_word_tokens", "numeric_tokens",
             "postal_code_candidates", "primary_postal", "building_number", "normalized_country",
             "name_length", "address_length", "name_word_count", "address_word_count"]

BIT_FEATURES = [s for s in STRATEGY_BIT]  # one indicator per blocking strategy


# --------------------------------------------------------------------------- #
# TF-IDF vectorisers (fit per split on unlabeled text only)
# --------------------------------------------------------------------------- #
def fit_vectorizers(cfg: Config, split: str, s1: pl.DataFrame, cand: pl.DataFrame) -> dict:
    path = cfg.path("features", f"{split}_tfidf.joblib")
    if path.exists() and not cfg.force:
        return joblib.load(path)
    rng = np.random.default_rng(cfg.seed)
    names = pl.concat([s1["normalized_name"], cand["normalized_name"]]).fill_null("")
    addrs = pl.concat([s1["normalized_address"], cand["normalized_address"]]).fill_null("")
    n = len(names)
    sel = rng.choice(n, size=min(n, cfg.tfidf_fit_sample), replace=False) if n > cfg.tfidf_fit_sample else None
    pick = (lambda s: s.gather(sel).to_list()) if sel is not None else (lambda s: s.to_list())
    vecs = {
        "name_char": TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True,
                                     dtype=np.float32, max_features=500_000),
        "name_word": TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b\w+\b", min_df=1,
                                     sublinear_tf=True, dtype=np.float32, max_features=1_000_000),
        "addr_char": TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2, sublinear_tf=True,
                                     dtype=np.float32, max_features=500_000),
    }
    vecs["name_char"].fit(pick(names))
    vecs["name_word"].fit(pick(names))
    vecs["addr_char"].fit(pick(addrs))
    joblib.dump(vecs, path)
    log.info("Fitted TF-IDF vectorisers for %s on %d strings", split, len(pick(names)))
    return vecs


def _row_cosine(vec: TfidfVectorizer, a: list[str], b: list[str]) -> np.ndarray:
    """Row-wise cosine of L2-normalised TF-IDF vectors; each distinct string is transformed once."""
    ua, ia = np.unique(np.asarray(a, dtype=object), return_inverse=True)
    ub, ib = np.unique(np.asarray(b, dtype=object), return_inverse=True)
    A, B = vec.transform(ua.tolist())[ia], vec.transform(ub.tolist())[ib]
    return np.asarray(A.multiply(B).sum(axis=1)).ravel().astype(np.float32)


# --------------------------------------------------------------------------- #
# Feature computation
# --------------------------------------------------------------------------- #
def _sims(a: list[str], b: list[str], w: int, prefix: str) -> dict[str, np.ndarray]:
    f32 = np.float32
    return {
        f"{prefix}_lev": process.cpdist(a, b, scorer=Levenshtein.normalized_similarity, workers=w).astype(f32),
        f"{prefix}_ratio": (process.cpdist(a, b, scorer=fuzz.ratio, workers=w) / 100).astype(f32),
        f"{prefix}_jaro": process.cpdist(a, b, scorer=Jaro.normalized_similarity, workers=w).astype(f32),
        f"{prefix}_jw": process.cpdist(a, b, scorer=JaroWinkler.normalized_similarity, workers=w).astype(f32),
        f"{prefix}_tsort": (process.cpdist(a, b, scorer=fuzz.token_sort_ratio, workers=w) / 100).astype(f32),
        f"{prefix}_tset": (process.cpdist(a, b, scorer=fuzz.token_set_ratio, workers=w) / 100).astype(f32),
        f"{prefix}_partial": (process.cpdist(a, b, scorer=fuzz.partial_ratio, workers=w) / 100).astype(f32),
    }


def _set_feats(df: pl.DataFrame, a: str, b: str, prefix: str) -> list[pl.Expr]:
    inter = pl.col(a).list.set_intersection(pl.col(b)).list.len().cast(pl.Float32)
    union = pl.col(a).list.set_union(pl.col(b)).list.len().cast(pl.Float32)
    la, lb = pl.col(a).list.unique().list.len().cast(pl.Float32), pl.col(b).list.unique().list.len().cast(pl.Float32)
    mn = pl.min_horizontal(la, lb)
    return [
        (inter / pl.when(union > 0).then(union).otherwise(None)).alias(f"{prefix}_jacc"),
        (inter / pl.when(mn > 0).then(mn).otherwise(None)).alias(f"{prefix}_overlap"),
        inter.alias(f"{prefix}_common"),
        (inter / pl.when(la > 0).then(la).otherwise(None)).alias(f"{prefix}_common_ratio"),
    ]


def compute_features(pairs: pl.DataFrame, s1: pl.DataFrame, cand: pl.DataFrame,
                     vecs: dict, cfg: Config) -> pl.DataFrame:
    """Features for one chunk of pairs (must contain complete S1 groups)."""
    w = cfg.n_workers
    s1t = s1.select(TEXT_COLS).rename({c: f"a_{c}" for c in TEXT_COLS})
    ct = cand.select(TEXT_COLS).rename({c: f"b_{c}" for c in TEXT_COLS})
    d = (pairs.join(s1t, left_on="s1_idx", right_on="a_idx", how="left")
              .join(ct, left_on="cand_idx", right_on="b_idx", how="left"))

    def L(col):
        return d[col].fill_null("").to_list()

    an, bn, ac, bc = L("a_normalized_name"), L("b_normalized_name"), L("a_name_core"), L("b_name_core")
    aa, ba = L("a_normalized_address"), L("b_normalized_address")
    feats: dict[str, np.ndarray] = {}
    feats.update(_sims(an, bn, w, "name"))
    feats.update({k: v for k, v in _sims(ac, bc, w, "core").items()
                  if k in ("core_ratio", "core_jw", "core_tset", "core_partial")})
    feats.update({k: v for k, v in _sims(aa, ba, w, "addr").items() if k != "addr_ratio"})
    feats["name_prefix"] = process.cpdist(ac, bc, scorer=Prefix.normalized_similarity, workers=w).astype(np.float32)
    feats["name_char_cos"] = _row_cosine(vecs["name_char"], an, bn)
    feats["name_word_cos"] = _row_cosine(vecs["name_word"], an, bn)
    feats["addr_char_cos"] = _row_cosine(vecs["addr_char"], aa, ba)
    ap, bp = L("a_primary_postal"), L("b_primary_postal")
    feats["postal_prefix"] = process.cpdist(ap, bp, scorer=Prefix.similarity, workers=w).astype(np.float32)
    na = "na_combined"
    feats[na] = (process.cpdist([x + " | " + y for x, y in zip(an, aa)], [x + " | " + y for x, y in zip(bn, ba)],
                                scorer=fuzz.token_set_ratio, workers=w) / 100).astype(np.float32)
    d = d.with_columns([pl.Series(k, v) for k, v in feats.items()])

    a_name_missing = pl.col("a_normalized_name").fill_null("") == ""
    b_name_missing = pl.col("b_normalized_name").fill_null("") == ""
    a_addr_missing = pl.col("a_normalized_address").fill_null("") == ""
    b_addr_missing = pl.col("b_normalized_address").fill_null("") == ""
    any_addr_missing = a_addr_missing | b_addr_missing
    postal_both = (pl.col("a_postal_code_candidates").list.len() > 0) & (pl.col("b_postal_code_candidates").list.len() > 0)
    postal_hit = pl.col("a_postal_code_candidates").list.set_intersection(pl.col("b_postal_code_candidates")).list.len() > 0
    num_inter = pl.col("a_numeric_tokens").list.set_intersection(pl.col("b_numeric_tokens")).list.len()
    ctry_both = pl.col("a_normalized_country").is_not_null() & pl.col("b_normalized_country").is_not_null()
    la, lb = pl.col("a_name_length").cast(pl.Float32), pl.col("b_name_length").cast(pl.Float32)
    ala, alb = pl.col("a_address_length").cast(pl.Float32), pl.col("b_address_length").cast(pl.Float32)
    acr_a = pl.col("a_name_acronym").fill_null("")
    acr_b = pl.col("b_name_acronym").fill_null("")
    core_a_ns = pl.col("a_name_core").fill_null("").str.replace_all(" ", "")
    core_b_ns = pl.col("b_name_core").fill_null("").str.replace_all(" ", "")

    d = d.with_columns(
        *_set_feats(d, "a_name_tokens", "b_name_tokens", "name_tok"),
        *_set_feats(d, "a_core_tokens", "b_core_tokens", "core_tok"),
        *_set_feats(d, "a_address_tokens", "b_address_tokens", "addr_tok"),
        (num_inter.cast(pl.Float32)).alias("num_common"),
        (num_inter.cast(pl.Float32) / pl.max_horizontal(
            pl.col("a_numeric_tokens").list.len(), pl.col("b_numeric_tokens").list.len(), pl.lit(1))
         ).alias("num_jacc"),
        pl.when(postal_both).then(postal_hit.cast(pl.Float32)).otherwise(None).alias("postal_match"),
        postal_both.cast(pl.Int8).alias("postal_both_present"),
        (postal_both & ~postal_hit).cast(pl.Int8).alias("postal_conflict"),
        pl.when(pl.col("a_building_number").is_not_null() & pl.col("b_building_number").is_not_null())
          .then((pl.col("a_building_number") == pl.col("b_building_number")).cast(pl.Float32))
          .otherwise(None).alias("building_match"),
        (la - lb).abs().alias("name_len_diff"),
        (pl.min_horizontal(la, lb) / pl.max_horizontal(la, lb, pl.lit(1.0))).alias("name_len_ratio"),
        (ala - alb).abs().alias("addr_len_diff"),
        (pl.col("a_name_word_count") - pl.col("b_name_word_count")).abs().cast(pl.Float32).alias("name_wc_diff"),
        ((pl.col("a_normalized_name") == pl.col("b_normalized_name")) & ~a_name_missing).cast(pl.Int8).alias("name_exact"),
        ((pl.col("a_name_core") == pl.col("b_name_core")) & ~a_name_missing).cast(pl.Int8).alias("core_exact"),
        ((pl.col("a_normalized_address") == pl.col("b_normalized_address")) & ~any_addr_missing).cast(pl.Int8).alias("addr_exact"),
        (((acr_a.str.len_chars() >= 2) & (acr_a == core_b_ns)) |
         ((acr_b.str.len_chars() >= 2) & (acr_b == core_a_ns))).cast(pl.Int8).alias("name_acronym_match"),
        pl.when(ctry_both).then((pl.col("a_normalized_country") == pl.col("b_normalized_country")).cast(pl.Float32))
          .otherwise(None).alias("country_match"),
        (~ctry_both).cast(pl.Int8).alias("country_missing"),
        (pl.col("source") == 3).cast(pl.Int8).alias("is_s3"),
        a_name_missing.cast(pl.Int8).alias("s1_name_missing"),
        b_name_missing.cast(pl.Int8).alias("cand_name_missing"),
        a_addr_missing.cast(pl.Int8).alias("s1_addr_missing"),
        b_addr_missing.cast(pl.Int8).alias("cand_addr_missing"),
        *[((pl.col("blk_bits") & bit) > 0).cast(pl.Int8).alias(f"blk_{s}") for s, bit in STRATEGY_BIT.items()],
    )
    # address similarities are meaningless when an address is missing -> NaN
    addr_cols = [c for c in d.columns if c.startswith("addr_") and c not in ("addr_exact",)]
    d = d.with_columns([pl.when(any_addr_missing).then(None).otherwise(pl.col(c)).alias(c) for c in addr_cols])
    d = d.with_columns(
        name_addr_mean=(pl.col("name_tset") + pl.col("addr_tset").fill_null(pl.col("name_tset"))) / 2,
    )
    # --- context features within each (S1) candidate group (no labels involved) ---
    d = d.with_columns(
        grp_n_cands=pl.len().over("s1_idx").cast(pl.Float32),
        grp_n_cands_src=pl.len().over("s1_idx", "source").cast(pl.Float32),
        name_jw_rank=pl.col("name_jw").rank("min", descending=True).over("s1_idx").cast(pl.Float32),
        name_jw_gap=(pl.col("name_jw").max().over("s1_idx") - pl.col("name_jw")),
        na_rank=pl.col("na_combined").rank("min", descending=True).over("s1_idx").cast(pl.Float32),
        na_gap=(pl.col("na_combined").max().over("s1_idx") - pl.col("na_combined")),
        quick_gap=(pl.col("blk_quick").max().over("s1_idx", "source") - pl.col("blk_quick")),
        grp_same_name=pl.len().over("s1_idx", "b_name_core").cast(pl.Float32),
        grp_n_core_exact=pl.col("core_exact").sum().over("s1_idx").cast(pl.Float32),
    )
    keep = ["s1_idx", "cand_idx", "source"] + feature_columns()
    return d.select([pl.col(c).cast(pl.Float32) if c in FEATURE_SET else pl.col(c) for c in keep])


def feature_columns() -> list[str]:
    return list(_FEATURES)


_FEATURES = (
    ["name_lev", "name_ratio", "name_jaro", "name_jw", "name_tsort", "name_tset", "name_partial",
     "core_ratio", "core_jw", "core_tset", "core_partial", "name_prefix", "name_char_cos", "name_word_cos",
     "name_tok_jacc", "name_tok_overlap", "name_tok_common", "name_tok_common_ratio",
     "core_tok_jacc", "core_tok_overlap", "core_tok_common", "core_tok_common_ratio",
     "name_len_diff", "name_len_ratio", "name_wc_diff", "name_exact", "core_exact", "name_acronym_match",
     "addr_lev", "addr_jaro", "addr_jw", "addr_tsort", "addr_tset", "addr_partial", "addr_char_cos",
     "addr_tok_jacc", "addr_tok_overlap", "addr_tok_common", "addr_tok_common_ratio",
     "num_common", "num_jacc", "postal_match", "postal_prefix", "postal_both_present", "postal_conflict",
     "building_match", "addr_len_diff", "addr_exact",
     "country_match", "country_missing", "na_combined", "name_addr_mean", "is_s3",
     "s1_name_missing", "cand_name_missing", "s1_addr_missing", "cand_addr_missing",
     "blk_score", "blk_quick", "blk_rank", "blk_n_strat"]
    + [f"blk_{s}" for s in BIT_FEATURES]
    + ["grp_n_cands", "grp_n_cands_src", "name_jw_rank", "name_jw_gap", "na_rank", "na_gap", "quick_gap",
       "grp_same_name", "grp_n_core_exact"]
)
FEATURE_SET = set(_FEATURES)


def build_feature_matrix(cfg: Config, split: str, pairs: pl.DataFrame, s1: pl.DataFrame,
                         cand: pl.DataFrame, tag: str | None = None) -> Path:
    """Chunked feature generation -> artifacts/features/<tag>/part-*.parquet (checkpointed).

    Chunks never split a Source-1 group; each finished chunk is written straight
    to disk so memory stays bounded. A ``_SUCCESS`` marker makes the stage resumable.
    """
    tag = tag or split
    out = cfg.art / "features" / tag
    if (out / "_SUCCESS").exists() and not cfg.force:
        log.info("[checkpoint] features %s exist", tag)
        return out
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("part-*.parquet"):
        old.unlink()
    vecs = fit_vectorizers(cfg, split, s1, cand)
    counts = pairs.group_by("s1_idx").len().sort("s1_idx")
    counts = counts.with_columns(chunk=((pl.col("len").cum_sum() - pl.col("len")) // cfg.feature_chunk_pairs))
    pairs = pairs.join(counts.select("s1_idx", "chunk"), on="s1_idx").sort("s1_idx", "source", "blk_rank")
    n_chunks = int(counts["chunk"].max() or 0) + 1 if counts.height else 0
    total = 0
    for i, chunk in enumerate(pairs.partition_by("chunk", maintain_order=True, include_key=False)):
        f = compute_features(chunk, s1, cand, vecs, cfg)
        f.write_parquet(out / f"part-{i:05d}.parquet", compression="zstd")
        total += f.height
        log.info("  features %s chunk %d/%d: %d pairs", tag, i + 1, n_chunks, f.height)
        del f
        import gc; gc.collect()
    (out / "_SUCCESS").write_text(str(total))
    log.info("Feature matrix %s: %d pairs x %d features", tag, total, len(feature_columns()))
    return out


def scan_features(path: Path) -> pl.LazyFrame:
    return pl.scan_parquet(str(Path(path) / "part-*.parquet"))
