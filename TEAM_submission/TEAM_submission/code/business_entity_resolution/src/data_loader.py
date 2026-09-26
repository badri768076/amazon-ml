"""Dataset loading and validation from the local dataset or Kaggle input.

The data source is automatically discovered from the local folder:
    student_resource/student_resource/dataset
with fallback to Kaggle input directory when running on Kaggle notebooks.
Raw tables are loaded directly from the discovered dataset directory and materialised
once to ``artifacts/raw/<table>.parquet`` (only the required columns), after
which every stage reads them lazily with Polars / DuckDB.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from config import (ALL_TABLES, DATA_ROOT as DEFAULT_DATA_ROOT, GT_COLUMNS, SOURCE_COLUMNS,
                    SOURCE_PREFIX, TEST_TABLES, TRAIN_TABLES, Config, resolve_data_root)

log = logging.getLogger("ber.data")

DATA_ROOT = DEFAULT_DATA_ROOT

_SHARD_RE = re.compile(r"[-_](\d{5})[-_]of[-_](\d{5})$")
_ID_SPLIT_RE = r"[,;|\s]+"


class DataError(RuntimeError):
    """Raised for missing files, missing columns, invalid schemas or corrupt data."""


# --------------------------------------------------------------------------- #
# Reference loader (plain pandas loader directly from dataset directory)
# --------------------------------------------------------------------------- #
def load_dataset_pandas_reference(data_root: Path | str | None = None) -> dict:
    """Plain pandas loader from the dataset directory. Handy for notebooks / EDA.

    The pipeline itself uses :func:`load_tables` / :func:`materialise_raw`, which reads
    only the required columns and caches Parquet locally instead of holding everything
    in pandas.
    """
    import pandas as pd

    data_root = Path(data_root) if data_root else resolve_data_root()
    print(f"Loading dataset from: {data_root}")
    found = _discover_dataset_files(data_root)
    dataframes = {}
    for table_name, file_paths in sorted(found.items()):
        print(f"Loading {table_name} ({len(file_paths)} file(s))...")
        dfs = []
        for file in sorted(file_paths):
            f_path = Path(file)
            if f_path.suffix == ".parquet":
                dfs.append(pd.read_parquet(f_path))
            elif f_path.suffix == ".csv":
                dfs.append(pd.read_csv(f_path))
            else:
                dfs.append(pd.read_csv(f_path, sep="\t"))
        dataframes[table_name] = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    print("\nLoaded tables:")
    print(list(dataframes.keys()))
    if "train_source1" in dataframes:
        print(dataframes["train_source1"].head())
    return dataframes


load_kaggle_pandas_reference = load_dataset_pandas_reference


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _required_columns(table: str) -> list[str]:
    return GT_COLUMNS if table.endswith("ground_truth") else SOURCE_COLUMNS


def _canonical_table_name(path: str) -> str | None:
    """Map a file path to one of the canonical table names (or None).

    Handles ``train_source1.parquet``, ``train/source1.parquet``,
    ``train/train_source1.parquet``, ``student-resource-amazon-ml/train_source2.parquet``,
    ``ground_truth.tsv``, ``train_ground_truth.csv``, and sharded files.
    """
    p = Path(path)
    stem = _SHARD_RE.sub("", p.stem).lower().replace("-", "_")
    aliases = {
        "groundtruth": "ground_truth",
        "gt": "ground_truth",
        "labels": "ground_truth",
        "train_gt": "train_ground_truth",
        "train_labels": "train_ground_truth",
    }
    stem = aliases.get(stem, stem)
    for split in ("train", "test"):
        if stem.startswith(f"{split}_"):
            rest = stem[len(split) + 1:]
            stem = f"{split}_{aliases.get(rest, rest)}"
    if stem in ALL_TABLES:
        return stem
    if stem in ("ground_truth", "groundtruth", "gt", "labels"):
        return "train_ground_truth"
    base = aliases.get(stem, stem)
    parents = [x.lower() for x in p.parts[:-1]]
    for split in ("train", "test"):
        if any(x == split or x.startswith(f"{split}_") or x.endswith(f"_{split}") for x in parents):
            name = f"{split}_{base}"
            if name in ALL_TABLES:
                return name
            if base in ("ground_truth", "groundtruth", "gt", "labels"):
                return "train_ground_truth"
    return None


def _match_columns(available: list[str], required: list[str], table: str) -> dict[str, str]:
    """Case/whitespace-insensitive column matching; raises on missing columns."""
    lookup = {c.strip().lower(): c for c in available}
    mapping, missing = {}, []
    for col in required:
        if col in lookup:
            mapping[lookup[col]] = col
        else:
            missing.append(col)
    if missing:
        raise DataError(f"{table}: missing required column(s) {missing}; found {available}")
    return mapping


def _read_one_file(path: str, table: str) -> pa.Table:
    """Read a single Parquet/TSV/CSV file with only the required columns."""
    req = _required_columns(table)
    try:
        if path.endswith(".parquet"):
            with open(path, "rb") as fh:
                pf = pq.ParquetFile(fh)
                mapping = _match_columns(pf.schema_arrow.names, req, table)
                tbl = pf.read(columns=list(mapping))
        else:
            sep = "," if path.endswith(".csv") else "\t"
            df = pl.read_csv(path, separator=sep, infer_schema_length=0, quote_char='"',
                             null_values=[""], truncate_ragged_lines=False)
            mapping = _match_columns(df.columns, req, table)
            tbl = df.select(list(mapping)).to_arrow()
    except DataError:
        raise
    except Exception as exc:  # corrupted file, bad encoding, ...
        raise DataError(f"{table}: failed to read {path}: {exc}") from exc
    return tbl.rename_columns([mapping[c] for c in tbl.column_names])


def _standardise(tbl: pa.Table, table: str) -> pl.DataFrame:
    df = pl.from_arrow(tbl)
    if table.endswith("ground_truth"):
        m = df.schema["matched_entity_ids"]
        if isinstance(m, pl.List):
            df = df.with_columns(pl.col("matched_entity_ids").list.eval(
                pl.element().cast(pl.Utf8).str.strip_chars()).list.join(","))
        df = df.with_columns(
            pl.col("source1_entity_id").cast(pl.Utf8).str.strip_chars(),
            pl.col("matched_entity_ids").cast(pl.Utf8).fill_null("").str.strip_chars(),
        )
    else:
        df = df.with_columns([pl.col(c).cast(pl.Utf8) for c in SOURCE_COLUMNS]) \
               .with_columns(pl.col("entity_id").str.strip_chars())
    return df


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def _discover_dataset_files(root: Path) -> dict[str, list[str]]:
    """Scan directory recursively for canonical dataset tables."""
    root = Path(root)
    found: dict[str, list[str]] = {}
    for ext in ("parquet", "tsv", "csv", "txt"):
        for f in sorted(root.rglob(f"*.{ext}")):
            # Skip hidden files, __MACOSX metadata and AppleDouble files
            if f.name.startswith(".") or any(part.startswith(".") or part == "__MACOSX" for part in f.parts):
                continue
            try:
                rel = str(f.relative_to(root))
            except ValueError:
                rel = str(f)
            name = _canonical_table_name(rel) or _canonical_table_name(f.name)
            if name:
                found.setdefault(name, []).append(str(f))
    # Prefer parquet files when multiple formats exist for the same table
    for name, paths in list(found.items()):
        parquets = [p for p in paths if p.endswith(".parquet")]
        if parquets:
            found[name] = parquets
    return found


def _discover_dataset(cfg: Config) -> dict[str, list[str]]:
    """Resolve dataset root and discover all required tables."""
    root = Path(cfg.data_dir) if cfg.data_dir else DATA_ROOT
    if not root.exists():
        from config import resolve_data_root
        root = resolve_data_root()
        if not root.exists():
            raise DataError(
                f"Dataset directory not found at: {cfg.data_dir} (or default: {DATA_ROOT})\n"
                f"Please pass --data-dir <path_to_dataset> or place the dataset under student_resource/dataset."
            )

    log.info("Loading dataset from directory: %s", root)
    found = _discover_dataset_files(root)
    return found


def materialise_raw(cfg: Config, tables: list[str] | None = None) -> dict[str, Path]:
    """Load and convert raw tables from the dataset to local Parquet cache once (checkpointed)."""
    tables = tables or ALL_TABLES
    out = {t: cfg.raw_dir / f"{t}.parquet" for t in tables}
    todo = [t for t in tables if cfg.force or not out[t].exists()]
    if not todo:
        return out
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    found = _discover_dataset(cfg)
    log.info("Discovered tables from dataset: %s", sorted(found))
    for t in todo:
        if t not in found:
            raise DataError(
                f"Required table '{t}' not found in dataset directory ({cfg.data_dir}).\n"
                f"Discovered tables: {sorted(found.keys())}"
            )
        parts = [_read_one_file(p, t) for p in sorted(found[t])]
        tbl = pa.concat_tables(parts, promote_options="default") if len(parts) > 1 else parts[0]
        df = _standardise(tbl, t)
        df.write_parquet(out[t], compression="zstd")
        log.info("Loaded %-20s rows=%10d  (%d file%s)", t, df.height, len(parts), "s" * (len(parts) > 1))
    return out


def scan(cfg: Config, table: str) -> pl.LazyFrame:
    """Lazy Polars scan of a materialised raw table."""
    p = cfg.raw_dir / f"{table}.parquet"
    if not p.exists():
        materialise_raw(cfg, [table])
    return pl.scan_parquet(p)


def load(cfg: Config, table: str) -> pl.DataFrame:
    return scan(cfg, table).collect()


def parse_gt(gt: pl.DataFrame) -> pl.DataFrame:
    """Ground truth -> (source1_entity_id, matched: list[str]) with empties removed."""
    return gt.with_columns(
        pl.col("matched_entity_ids").fill_null("").str.replace_all(_ID_SPLIT_RE, ",").str.split(",")
    ).with_columns(
        pl.col("matched_entity_ids").list.eval(
            pl.element().str.strip_chars().filter(pl.element().str.len_chars() > 0)).alias("matched")
    ).select("source1_entity_id", "matched")


def gt_pairs(gt: pl.DataFrame) -> pl.DataFrame:
    """Exploded ground-truth pairs (s1_id, cand_id), de-duplicated."""
    return parse_gt(gt).explode("matched").drop_nulls("matched") \
        .rename({"source1_entity_id": "s1_id", "matched": "cand_id"}).unique()


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_tables(cfg: Config) -> dict:
    """Run schema/content checks on all tables and write a validation report."""
    materialise_raw(cfg)
    report: dict = {"tables": {}, "ground_truth": {}, "errors": [], "warnings": []}
    ids: dict[str, set] = {}
    for t in ALL_TABLES:
        if t.endswith("ground_truth"):
            continue
        df = load(cfg, t)
        src = t.split("_")[1]
        prefix = SOURCE_PREFIX[src]
        n = df.height
        stats = {
            "rows": n,
            "unique_entity_ids": df["entity_id"].n_unique(),
            "null_entity_ids": df["entity_id"].null_count(),
            "bad_prefix": int((~df["entity_id"].fill_null("").str.starts_with(prefix)).sum()),
            "missing": {c: int(df[c].null_count()) for c in SOURCE_COLUMNS[1:]},
            "empty_name": int((df["business_name"].fill_null("").str.strip_chars() == "").sum()),
            "empty_address": int((df["business_address"].fill_null("").str.strip_chars() == "").sum()),
            "countries": dict(df["country"].fill_null("<NULL>").value_counts(sort=True)
                              .head(30).iter_rows()),
        }
        if n == 0:
            report["errors"].append(f"{t}: table is empty")
        if stats["unique_entity_ids"] != n:
            report["errors"].append(f"{t}: {n - stats['unique_entity_ids']} duplicate entity_id(s)")
        if stats["bad_prefix"]:
            report["errors"].append(f"{t}: {stats['bad_prefix']} ids without prefix {prefix}")
        report["tables"][t] = stats
        if t.startswith("train"):
            ids[src] = set(df["entity_id"].to_list())
        log.info("%-15s rows=%d missing=%s", t, n, stats["missing"])

    gt = load(cfg, "train_ground_truth")
    parsed = parse_gt(gt)
    pairs = parsed.explode("matched").drop_nulls("matched")
    s1_ids = ids["source1"]
    cand_ids = ids["source2"] | ids["source3"]
    matched = pairs["matched"].to_list()
    lens = parsed["matched"].list.len()
    dup_within = parsed.filter(pl.col("matched").list.len() != pl.col("matched").list.unique().list.len())
    g = {
        "rows": gt.height,
        "unique_s1": parsed["source1_entity_id"].n_unique(),
        "s1_not_in_source1": int(sum(1 for x in parsed["source1_entity_id"] if x not in s1_ids)),
        "source1_without_gt_row": len(s1_ids - set(parsed["source1_entity_id"].to_list())),
        "invalid_match_ids": int(sum(1 for m in matched if m not in cand_ids)),
        "match_ids_wrong_prefix": int(sum(1 for m in matched if not m.startswith(("S2-", "S3-")))),
        "rows_with_duplicate_match_ids": dup_within.height,
        "self_matches": int(pairs.filter(pl.col("source1_entity_id") == pl.col("matched")).height),
        "match_ids_used_by_multiple_s1": int(pairs.group_by("matched").len()
                                             .filter(pl.col("len") > 1).height),
        "zero_match": int((lens == 0).sum()),
        "one_match": int((lens == 1).sum()),
        "multi_match": int((lens > 1).sum()),
        "max_matches": int(lens.max() or 0),
        "s2_matches": int(sum(1 for m in matched if m.startswith("S2-"))),
        "s3_matches": int(sum(1 for m in matched if m.startswith("S3-"))),
    }
    if g["unique_s1"] != g["rows"]:
        report["errors"].append(f"train_ground_truth: {g['rows'] - g['unique_s1']} duplicate S1 rows")
    for k in ("s1_not_in_source1", "invalid_match_ids", "match_ids_wrong_prefix", "self_matches"):
        if g[k]:
            report["warnings"].append(f"train_ground_truth: {k} = {g[k]}")
    if g["rows_with_duplicate_match_ids"]:
        report["warnings"].append(f"train_ground_truth: {g['rows_with_duplicate_match_ids']} rows "
                                  "contain duplicate match ids (de-duplicated downstream)")
    report["ground_truth"] = g
    log.info("Ground truth: zero=%d one=%d multi=%d S2=%d S3=%d invalid=%d",
             g["zero_match"], g["one_match"], g["multi_match"], g["s2_matches"],
             g["s3_matches"], g["invalid_match_ids"])

    (cfg.report_dir / "validation_report.json").write_text(json.dumps(report, indent=2))
    for w in report["warnings"]:
        log.warning(w)
    if report["errors"]:
        for e in report["errors"]:
            log.error(e)
        raise DataError("Data validation failed - see artifacts/reports/validation_report.json")
    log.info("Data validation passed (%d warnings)", len(report["warnings"]))
    return report


def resolve_gt_policy(cfg: Config, gt: pl.DataFrame) -> str:
    """Decide how Source-1 train entities absent from ground truth are treated.

    auto: if the GT file lists explicit empty rows, absent S1 ids are *unlabelled*
    (excluded); otherwise absent S1 ids are taken to have zero matches.
    """
    if cfg.gt_absent_policy != "auto":
        return cfg.gt_absent_policy
    has_empty = parse_gt(gt).filter(pl.col("matched").list.len() == 0).height > 0
    return "exclude" if has_empty else "no_match"


__all__ = ["DATA_ROOT", "DataError", "materialise_raw", "scan", "load", "parse_gt", "gt_pairs",
           "validate_tables", "resolve_gt_policy", "load_kaggle_pandas_reference",
           "TRAIN_TABLES", "TEST_TABLES"]
