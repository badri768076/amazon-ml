"""Metrics, EDA and error analysis.

Primary metric: F0.5 over matched IDs (micro-averaged across Source-1 entities)

    TP = predicted match IDs that are true matches
    FP = predicted match IDs that are not true matches
    FN = true match IDs not predicted (including ones blocking never retrieved)
    F0.5 = 1.25 * P * R / (0.25 * P + R)

Singleton correctness = share of Source-1 entities with *no* true match for
which the system predicts an empty match list.
"""
from __future__ import annotations

import json
import logging
from collections import Counter

import duckdb
import numpy as np
import polars as pl

from config import ALL_TABLES, Config

log = logging.getLogger("ber.eval")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def prf(tp: float, fp: float, fn: float) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f05 = 1.25 * p * r / (0.25 * p + r) if p + r else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f0_5": f05, "f1": f1, "tp": int(tp), "fp": int(fp), "fn": int(fn)}


class GroupEvaluator:
    """Vectorised entity-level evaluation for many decision rules on one pair set.

    Pairs must be sorted by Source-1 entity. ``n_true_total`` includes true
    matches that are not in the candidate set.
    """

    def __init__(self, s1_keys: np.ndarray, labels: np.ndarray, n_true_total: int,
                 singleton_entities: set, n_entities: int, s1_ids: np.ndarray | None = None):
        self.labels = labels.astype(bool)
        self.n_true_total = n_true_total
        self.n_entities = n_entities
        _, self.starts, self.group_of = np.unique(s1_keys, return_index=True, return_inverse=True)
        self.n_groups = len(self.starts)
        ids = s1_ids if s1_ids is not None else s1_keys
        first_ids = ids[self.starts]
        self.group_is_singleton = np.array([x in singleton_entities for x in first_ids])
        self.n_singletons = len(singleton_entities)

    def group_max(self, x: np.ndarray) -> np.ndarray:
        return np.maximum.reduceat(x, self.starts)[self.group_of]

    def group_rank(self, x: np.ndarray) -> np.ndarray:
        """1-based descending rank of x inside its group."""
        order = np.lexsort((-x, self.group_of))
        ranks = np.empty(len(x), dtype=np.int32)
        pos = np.arange(len(x))
        grp_start = self.starts[self.group_of[order]]
        ranks[order] = pos - grp_start + 1
        return ranks

    def evaluate(self, pred: np.ndarray) -> dict:
        tp = float(np.sum(pred & self.labels))
        fp = float(np.sum(pred & ~self.labels))
        fn = self.n_true_total - tp
        m = prf(tp, fp, fn)
        any_pred = np.logical_or.reduceat(pred, self.starts) if len(pred) else np.array([], bool)
        wrong_singletons = int(np.sum(any_pred & self.group_is_singleton))
        m["singleton_correctness"] = (1 - wrong_singletons / self.n_singletons) if self.n_singletons else None
        m["avg_matches_per_s1"] = float(pred.sum() / max(1, self.n_entities))
        m["s1_with_no_prediction"] = int(self.n_entities - any_pred.sum())
        return m


def evaluate_predictions(pred_pairs: pl.DataFrame, gt: pl.DataFrame, entities: list[str]) -> dict:
    """Set-based evaluation from (s1_id, cand_id) frames (used for final reports)."""
    ent = set(entities)
    pred = pred_pairs.filter(pl.col("s1_id").is_in(list(ent))).unique()
    gt = gt.filter(pl.col("s1_id").is_in(list(ent))).unique()
    tp = pred.join(gt, on=["s1_id", "cand_id"], how="inner").height
    m = prf(tp, pred.height - tp, gt.height - tp)
    single = ent - set(gt["s1_id"].unique().to_list())
    predicted_for = set(pred["s1_id"].unique().to_list())
    m["singleton_correctness"] = (1 - len(single & predicted_for) / len(single)) if single else None
    m["avg_matches_per_s1"] = pred.height / max(1, len(ent))
    return m


# --------------------------------------------------------------------------- #
# EDA
# --------------------------------------------------------------------------- #
def run_eda(cfg: Config) -> dict:
    """Dataset profiling with DuckDB directly on the cached Parquet files."""
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={cfg.n_workers}")
    rep: dict = {"tables": {}}
    for t in ALL_TABLES:
        path = str(cfg.raw_dir / f"{t}.parquet")
        if t.endswith("ground_truth"):
            continue
        src = f"read_parquet('{path}')"
        q = lambda sql: con.execute(sql.replace("{T}", src)).fetchall()  # noqa: E731
        (n, n_ids, miss_n, miss_a, miss_c, empty_n, empty_a) = q("""
            SELECT count(*), count(DISTINCT entity_id),
                   count(*) FILTER (WHERE business_name IS NULL),
                   count(*) FILTER (WHERE business_address IS NULL),
                   count(*) FILTER (WHERE country IS NULL OR trim(country) = ''),
                   count(*) FILTER (WHERE trim(coalesce(business_name,'')) = ''),
                   count(*) FILTER (WHERE trim(coalesce(business_address,'')) = '')
            FROM {T}""")[0]
        dup_name = q("SELECT count(*) - count(DISTINCT lower(trim(business_name))) FROM {T}")[0][0]
        dup_addr = q("SELECT count(*) - count(DISTINCT lower(trim(business_address))) FROM {T}")[0][0]
        dup_rec = q("""SELECT count(*) - count(DISTINCT (lower(trim(coalesce(business_name,''))),
                        lower(trim(coalesce(business_address,''))), lower(trim(coalesce(country,''))))) FROM {T}""")[0][0]
        countries = q("SELECT coalesce(country,'<NULL>') c, count(*) n FROM {T} GROUP BY 1 ORDER BY 2 DESC LIMIT 25")
        text = q("""
            SELECT avg(length(business_name)), median(length(business_name)), max(length(business_name)),
                   avg(length(business_address)), median(length(business_address)), max(length(business_address)),
                   avg(len(string_split(trim(business_name), ' '))), avg(len(string_split(trim(business_address), ' '))),
                   avg(CASE WHEN regexp_matches(business_name, '[0-9]') THEN 1 ELSE 0 END),
                   avg(CASE WHEN regexp_matches(business_address, '[0-9]') THEN 1 ELSE 0 END),
                   avg(CASE WHEN regexp_matches(business_name, '[^\\x00-\\x7F]') THEN 1 ELSE 0 END),
                   avg(CASE WHEN regexp_matches(business_address, '[^\\x00-\\x7F]') THEN 1 ELSE 0 END),
                   avg(CASE WHEN regexp_matches(business_address, '\\b[0-9]{5,6}\\b') THEN 1 ELSE 0 END)
            FROM {T}""")[0]
        top_name_tokens = q("""SELECT tok, count(*) n FROM (SELECT unnest(string_split(lower(business_name), ' ')) tok
                               FROM (SELECT business_name FROM {T} USING SAMPLE 200000 ROWS)) WHERE tok <> ''
                               GROUP BY 1 ORDER BY 2 DESC LIMIT 30""")
        top_addr_tokens = q("""SELECT tok, count(*) n FROM (SELECT unnest(string_split(lower(business_address), ' ')) tok
                               FROM (SELECT business_address FROM {T} USING SAMPLE 200000 ROWS)) WHERE tok <> ''
                               GROUP BY 1 ORDER BY 2 DESC LIMIT 30""")
        sample = con.execute(f"SELECT * FROM read_parquet('{path}') LIMIT 5").fetchall()
        rep["tables"][t] = {
            "rows": n, "distinct_ids": n_ids, "duplicate_ids": n - n_ids,
            "missing": {"business_name": miss_n, "business_address": miss_a, "country": miss_c},
            "empty": {"business_name": empty_n, "business_address": empty_a},
            "duplicates": {"names": dup_name, "addresses": dup_addr, "full_records": dup_rec},
            "countries": countries,
            "text": dict(zip(["name_len_mean", "name_len_median", "name_len_max", "addr_len_mean",
                              "addr_len_median", "addr_len_max", "name_words_mean", "addr_words_mean",
                              "name_has_digit", "addr_has_digit", "name_non_ascii", "addr_non_ascii",
                              "addr_has_postal_like"], [float(x) if x is not None else None for x in text])),
            "top_name_tokens": top_name_tokens, "top_address_tokens": top_addr_tokens,
            "sample_records": [list(map(str, r)) for r in sample],
        }
        log.info("EDA %-14s rows=%d dupName=%d dupAddr=%d missingCountry=%d countries=%s", t, n, dup_name,
                 dup_addr, miss_c, countries[:5])
    # character distribution (unicode blocks) on a sample of names
    from data_loader import load, parse_gt
    blocks = {}
    for t in ("train_source1", "test_source1"):
        names = load(cfg, t)["business_name"].drop_nulls()
        names = names.sample(min(50000, names.len()), seed=cfg.seed)
        c = Counter()
        for s in names:
            for ch in s:
                o = ord(ch)
                c["ascii_letter" if ch.isascii() and ch.isalpha() else "digit" if ch.isdigit() else
                  "space" if ch.isspace() else "ascii_punct" if o < 128 else
                  "latin_ext" if o < 0x250 else "devanagari" if 0x900 <= o < 0x980 else "other_unicode"] += 1
        tot = sum(c.values()) or 1
        blocks[t] = {k: v / tot for k, v in c.most_common()}
    rep["character_distribution"] = blocks
    # train vs test country sets (open-set check)
    tr = {c for c, _ in rep["tables"]["train_source1"]["countries"]}
    te = {c for t in ("test_source1", "test_source2", "test_source3") for c, _ in rep["tables"][t]["countries"]}
    rep["countries_only_in_test"] = sorted(te - tr)
    # ground truth
    gt = parse_gt(load(cfg, "train_ground_truth"))
    lens = gt["matched"].list.len()
    ex = gt.explode("matched").drop_nulls("matched")["matched"]
    per_s1_s2 = gt["matched"].list.eval(pl.element().str.starts_with("S2-").cast(pl.Int32)).list.sum()
    rep["ground_truth"] = {
        "rows": gt.height, "zero_match": int((lens == 0).sum()), "one_match": int((lens == 1).sum()),
        "multi_match": int((lens > 1).sum()), "mean_matches": float(lens.mean()), "max_matches": int(lens.max()),
        "match_count_hist": dict(Counter(min(int(x), 10) for x in lens)),
        "s2_matches": int(ex.str.starts_with("S2-").sum()), "s3_matches": int(ex.str.starts_with("S3-").sum()),
        "s1_with_multiple_s2": int((per_s1_s2 > 1).sum()),
    }
    (cfg.report_dir / "eda.json").write_text(json.dumps(rep, indent=2, default=str))
    _write_eda_md(cfg, rep)
    log.info("EDA ground truth: %s", {k: v for k, v in rep["ground_truth"].items() if k != "match_count_hist"})
    log.info("Countries only in test: %s", rep["countries_only_in_test"])
    return rep


def _write_eda_md(cfg: Config, rep: dict) -> None:
    lines = ["# EDA report", "", "| table | rows | dup ids | missing name | missing addr | missing country | "
             "dup names | dup addr | dup records |", "|---|---|---|---|---|---|---|---|---|"]
    for t, s in rep["tables"].items():
        lines.append(f"| {t} | {s['rows']:,} | {s['duplicate_ids']} | {s['missing']['business_name']:,} | "
                     f"{s['missing']['business_address']:,} | {s['missing']['country']:,} | "
                     f"{s['duplicates']['names']:,} | {s['duplicates']['addresses']:,} | {s['duplicates']['full_records']:,} |")
    lines += ["", "## Countries", ""]
    for t, s in rep["tables"].items():
        lines.append(f"- **{t}**: " + ", ".join(f"{c} ({n:,})" for c, n in s["countries"][:10]))
    lines += ["", f"Countries only in test: {rep['countries_only_in_test']}", "", "## Text", ""]
    for t, s in rep["tables"].items():
        lines.append(f"- **{t}**: " + ", ".join(f"{k}={v:.2f}" for k, v in s["text"].items() if v is not None))
    lines += ["", "## Ground truth", "", "```", json.dumps(rep["ground_truth"], indent=2), "```"]
    (cfg.report_dir / "eda.md").write_text("\n".join(lines))


# --------------------------------------------------------------------------- #
# Error analysis
# --------------------------------------------------------------------------- #
def error_analysis(cfg: Config, val: pl.DataFrame, pred_mask: np.ndarray, val_gt: pl.DataFrame,
                   s1: pl.DataFrame, cand: pl.DataFrame, n: int = 300) -> dict:
    """Write FP / FN samples with the raw records and summarise error causes."""
    v = val.with_columns(pred=pl.Series(pred_mask))
    fp = v.filter(pl.col("pred") & (pl.col("label") == 0))
    tp = v.filter(pl.col("pred") & (pl.col("label") == 1)).select("s1_id", "cand_id")
    fn_all = val_gt.join(tp, on=["s1_id", "cand_id"], how="anti")
    in_cands = v.filter(pl.col("label") == 1).select("s1_id", "cand_id", "p_final")
    fn = fn_all.join(in_cands, on=["s1_id", "cand_id"], how="left").with_columns(
        cause=pl.when(pl.col("p_final").is_null()).then(pl.lit("missed_by_blocking")).otherwise(pl.lit("below_threshold")))
    s1r = s1.select(pl.col("entity_id").alias("s1_id"), pl.col("business_name").alias("s1_name"),
                    pl.col("business_address").alias("s1_address"), pl.col("country").alias("s1_country"))
    cr = cand.select(pl.col("entity_id").alias("cand_id"), pl.col("business_name").alias("cand_name"),
                     pl.col("business_address").alias("cand_address"), pl.col("country").alias("cand_country"))
    fp_s = fp.sort("p_final", descending=True).head(n).select("s1_id", "cand_id", "p_final", "name_jw", "addr_tset") \
             .join(s1r, on="s1_id", how="left").join(cr, on="cand_id", how="left")
    fn_s = fn.head(n).join(s1r, on="s1_id", how="left").join(cr, on="cand_id", how="left")
    fp_s.write_csv(cfg.report_dir / "error_analysis_false_positives.tsv", separator="\t")
    fn_s.write_csv(cfg.report_dir / "error_analysis_false_negatives.tsv", separator="\t")
    bucket = lambda c: (pl.when(pl.col(c) >= .9).then(pl.lit(">=0.9")).when(pl.col(c) >= .7).then(pl.lit("0.7-0.9"))  # noqa: E731
                        .otherwise(pl.lit("<0.7")))
    summary = {
        "false_positives": fp.height,
        "false_negatives": fn.height,
        "fn_by_cause": dict(fn.group_by("cause").len().iter_rows()),
        "fp_by_name_similarity": dict(fp.with_columns(b=bucket("name_jw")).group_by("b").len().iter_rows()),
        "fp_same_postal": int(fp["postal_match"].fill_null(0).sum()) if fp.height else 0,
        "fp_country_mismatch": int((fp["country_match"] == 0).sum()) if fp.height else 0,
        "fp_by_source": dict(fp.group_by("source").len().iter_rows()),
        "fn_by_source": dict(fn.with_columns(src=pl.col("cand_id").str.slice(0, 2)).group_by("src").len().iter_rows()),
    }
    (cfg.report_dir / "error_analysis.json").write_text(json.dumps(summary, indent=2, default=str))
    log.info("Error analysis: %s", summary)
    return summary
