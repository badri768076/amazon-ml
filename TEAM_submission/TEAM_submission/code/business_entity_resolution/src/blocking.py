"""Blocking keys and inverted indexes.

Each strategy turns a normalised record into zero or more *blocking keys*.
Keys are country-scoped (see :func:`scope_keys`) so records only meet inside
the same country, while records whose country is missing are still reachable.
Candidate-side keys shared by more than ``max_block_size`` records are dropped
as uninformative. Pairs are produced by hash joins on the keys - never by a
Cartesian product.

Strategies
----------
exact_name       exact normalised name (unscoped - survives wrong/missing country)
exact_core       exact name with legal forms removed (unscoped)
name_signature   sorted unique core tokens  (word-order variations)
name_token       k rarest informative core-name tokens of the Source-1 record
char_ngram       k rarest character 3-grams of the core name (typos, translit.)
addr_token       k rarest alphabetic address tokens
postal_name      postal/PIN code + first 3 letters of core name
postal_building  postal/PIN code + building number
token_postal     rarest name tokens + postal code
addr_signature   sorted unique alphabetic address tokens
(fuzzy retrieval lives in candidate_generation.py - it is index based, not key based)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import polars as pl

from config import Config

log = logging.getLogger("ber.blocking")

STRATEGIES = ["exact_name", "exact_core", "name_signature", "name_token", "char_ngram",
              "addr_token", "postal_name", "postal_building", "token_postal", "addr_signature",
              "fuzzy"]
STRATEGY_BIT = {s: 1 << i for i, s in enumerate(STRATEGIES)}

KEY_COLS = ["idx", "normalized_country", "normalized_name", "name_core", "name_signature",
            "core_tokens", "address_word_tokens", "primary_postal", "building_number",
            "address_signature"]


@dataclass
class Strategy:
    name: str
    scoped: bool
    cap_attr: str
    rarest_k_attr: str | None = None   # S1 side keeps only the k rarest keys


STRATEGY_SPECS = [
    Strategy("exact_name", False, "max_block_size_exact"),
    Strategy("exact_core", False, "max_block_size_exact"),
    Strategy("name_signature", True, "max_block_size_exact"),
    Strategy("name_token", True, "max_block_size_token", "name_tokens_per_record"),
    Strategy("char_ngram", True, "max_block_size_ngram", "ngrams_per_record"),
    Strategy("addr_token", True, "max_block_size_addr", "addr_tokens_per_record"),
    Strategy("postal_name", True, "max_block_size_combo"),
    Strategy("postal_building", True, "max_block_size_combo"),
    Strategy("token_postal", True, "max_block_size_combo", "name_tokens_per_record"),
    Strategy("addr_signature", True, "max_block_size_combo"),
]


# --------------------------------------------------------------------------- #
# Raw key generation: (idx, normalized_country, key)
# --------------------------------------------------------------------------- #
def _nonempty(col: str, min_len: int = 1) -> pl.Expr:
    return pl.col(col).is_not_null() & (pl.col(col).str.len_chars() >= min_len)


def raw_keys(lf: pl.LazyFrame, strategy: str, cfg: Config) -> pl.LazyFrame:
    b = cfg.blocking
    base = ["idx", "normalized_country"]
    if strategy == "exact_name":
        out = lf.filter(_nonempty("normalized_name", 2)).select(*base, key=pl.col("normalized_name"))
    elif strategy == "exact_core":
        out = lf.filter(_nonempty("name_core", 3)).select(*base, key=pl.col("name_core"))
    elif strategy == "name_signature":
        out = lf.filter(_nonempty("name_signature", 3)).select(*base, key=pl.col("name_signature"))
    elif strategy in ("name_token", "token_postal"):
        out = (lf.select(*base, "primary_postal", key=pl.col("core_tokens")).explode("key")
                 .filter(pl.col("key").str.len_chars() >= b.min_token_len).unique(["idx", "key"]))
        if strategy == "token_postal":
            out = out.filter(pl.col("primary_postal").is_not_null()) \
                     .with_columns(key=pl.col("key") + "@" + pl.col("primary_postal"))
        out = out.select(*base, "key")
    elif strategy == "char_ngram":
        q = b.ngram_size
        s = (lf.filter(_nonempty("name_core", q))
               .select(*base, s=pl.lit(" ") + pl.col("name_core").str.slice(0, 15).str.replace_all(" ", "_") + pl.lit(" ")))
        out = (s.with_columns(pos=pl.int_ranges(0, pl.col("s").str.len_chars() - q + 1)).explode("pos")
                .select(*base, key=pl.col("s").str.slice(pl.col("pos"), q)).unique(["idx", "key"]))
    elif strategy == "addr_token":
        out = (lf.select(*base, key=pl.col("address_word_tokens")).explode("key")
                 .filter(pl.col("key").str.len_chars() >= 3).unique(["idx", "key"]))
    elif strategy == "postal_name":
        out = (lf.filter(pl.col("primary_postal").is_not_null() & _nonempty("name_core", 3))
                 .select(*base, key=pl.col("primary_postal") + "#" + pl.col("name_core").str.slice(0, 3)))
    elif strategy == "postal_building":
        out = (lf.filter(pl.col("primary_postal").is_not_null() & pl.col("building_number").is_not_null())
                 .select(*base, key=pl.col("primary_postal") + "#" + pl.col("building_number")))
    elif strategy == "addr_signature":
        out = (lf.filter(pl.col("address_word_tokens").list.len() >= 2)
                 .select(*base, key=pl.col("address_signature")))
    else:
        raise ValueError(strategy)
    return out


def scope_keys(keys: pl.LazyFrame, side: str, scoped: bool) -> pl.LazyFrame:
    """Country-scope keys and hash them to u64.

    candidate with country c : "c|k"
    candidate without country: "?|k"
    source-1  with country c : "c|k" and "?|k" (to reach missing-country candidates)
    source-1  without country: "?|k"
    => same-country pairs meet, missing-country records are never excluded, zero redundant duplication.
    """
    c, k = pl.col("normalized_country"), pl.col("key")
    if not scoped:
        return keys.select("idx", key=k.hash(seed=17)).unique()
    if side == "cand":
        k_scoped = pl.when(c.is_null()).then(pl.lit("?|") + k).otherwise(c + "|" + k)
        return keys.select("idx", key=k_scoped.hash(seed=17)).unique()
    else:
        has = keys.filter(c.is_not_null())
        parts = [has.select("idx", k=c + "|" + k), has.select("idx", k=pl.lit("?|") + k),
                 keys.filter(c.is_null()).select("idx", k=pl.lit("?|") + k)]
        return pl.concat(parts).select("idx", key=pl.col("k").hash(seed=17)).unique()


# --------------------------------------------------------------------------- #
# Candidate-side inverted index
# --------------------------------------------------------------------------- #
@dataclass
class KeyIndex:
    strategy: Strategy
    postings: pl.DataFrame          # key(u64) -> cand idx, df
    raw_df: pl.DataFrame | None     # unscoped key string -> df (for rarest-k ranking)
    n_cand: int


def build_index(cand: pl.DataFrame, spec: Strategy, cfg: Config) -> KeyIndex:
    import gc
    cap = getattr(cfg.blocking, spec.cap_attr)
    lf = cand.lazy().select(KEY_COLS)
    rk = raw_keys(lf, spec.name, cfg)
    raw_df = None
    if spec.rarest_k_attr:
        # rarest-k ranking always uses the plain token frequency
        base = raw_keys(lf, "name_token", cfg) if spec.name == "token_postal" else rk
        raw_df = base.group_by("key").agg(df=pl.len().cast(pl.UInt32)).collect()

    scoped = scope_keys(rk, "cand", spec.scoped)
    # Memory-efficient: compute key counts with group_by, discard keys > cap before joining
    freq = scoped.group_by("key").agg(df=pl.len().cast(pl.UInt32)).filter(pl.col("df") <= cap)
    postings = (scoped.join(freq, on="key", how="inner")
                .select("key", pl.col("idx").alias("cand_idx"), "df")
                .collect())
    gc.collect()
    log.info("  index %-16s postings=%10d keys=%9d (cap=%d)", spec.name, postings.height,
             postings["key"].n_unique() if postings.height else 0, cap)
    return KeyIndex(spec, postings, raw_df, cand.height)


def query_index(s1_chunk: pl.DataFrame, index: KeyIndex, cfg: Config) -> pl.DataFrame:
    """Join a chunk of Source-1 records against one strategy index.

    Returns (s1_idx, cand_idx, contrib, bits) aggregated per pair.
    """
    spec = index.strategy
    w = cfg.blocking.weights[spec.name]
    rk = raw_keys(s1_chunk.lazy().select(KEY_COLS), spec.name, cfg)
    if spec.rarest_k_attr and index.raw_df is not None:
        k = getattr(cfg.blocking, spec.rarest_k_attr)
        cap = getattr(cfg.blocking, spec.cap_attr)
        base_key = pl.col("key").str.split("@").list.first() if spec.name == "token_postal" else pl.col("key")
        # token_postal keys are made selective by the postal code, so common tokens stay eligible
        df_cap = None if spec.name == "token_postal" else cap * 4
        rk = (rk.with_columns(base=base_key)
                .join(index.raw_df.lazy().rename({"key": "base"}), on="base", how="inner")
                .filter(pl.lit(True) if df_cap is None else pl.col("df") <= df_cap)
                .sort("idx", "df", "key")                       # deterministic tie-breaking
                .with_columns(r=pl.int_range(pl.len()).over("idx"))
                .filter(pl.col("r") < k).drop("base", "df", "r"))
    keys = scope_keys(rk, "s1", spec.scoped)
    n = float(index.n_cand)
    pairs = (keys.join(index.postings.lazy(), on="key", how="inner")
                 .group_by(pl.col("idx").alias("s1_idx"), "cand_idx")
                 .agg(contrib=(w * (n / pl.col("df").cast(pl.Float64)).log1p()).sum().cast(pl.Float32))
                 .with_columns(bits=pl.lit(STRATEGY_BIT[spec.name], dtype=pl.Int32))
                 .collect())
    return pairs
