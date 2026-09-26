"""Write and validate the official submission files.

output/matching_results.tsv   source1_entity_id <TAB> matched_entity_ids
output/candidate_pairs.tsv    source1_entity_id <TAB> candidate_entity_ids

* exactly one row per test Source-1 entity, sorted by id (deterministic)
* comma-separated, de-duplicated, sorted S2-/S3- ids; empty list allowed
* candidate_pairs = the exact pair set scored by the model
* every final match is contained in the entity's candidate list
"""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import polars as pl

from config import Config
from data_loader import load

log = logging.getLogger("ber.submission")

MATCH_HEADER = ["source1_entity_id", "matched_entity_ids"]
CAND_HEADER = ["source1_entity_id", "candidate_entity_ids"]


class SubmissionError(RuntimeError):
    pass


def _grouped(pairs: pl.DataFrame, all_s1: pl.Series, col: str) -> pl.DataFrame:
    agg = (pairs.select("s1_id", "cand_id").unique()
                .group_by("s1_id").agg(pl.col("cand_id").sort().alias("ids")))
    return (pl.DataFrame({"s1_id": all_s1}).join(agg, on="s1_id", how="left")
              .with_columns(pl.col("ids").fill_null(pl.lit([], dtype=pl.List(pl.Utf8))).list.join(",").alias(col))
              .select("s1_id", col).sort("s1_id"))


def _write_tsv(path: Path, header: list[str], df: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        fh.write("\t".join(header) + "\n")
        for a, b in df.iter_rows():
            fh.write(f"{a}\t{b}\n")


def write_submission(cfg: Config, scored: pl.DataFrame) -> tuple[Path, Path]:
    all_s1 = load(cfg, "test_source1")["entity_id"].unique().sort()
    matches = _grouped(scored.filter(pl.col("pred")), all_s1, "matched_entity_ids")
    cands = _grouped(scored, all_s1, "candidate_entity_ids")
    out = Path(cfg.output_dir)
    mp, cp = out / "matching_results.tsv", out / "candidate_pairs.tsv"
    _write_tsv(mp, MATCH_HEADER, matches)
    _write_tsv(cp, CAND_HEADER, cands)
    n_match = int(scored["pred"].sum())
    log.info("Wrote %s (%d rows, %d matches, %d empty) and %s (%d rows, %d candidate pairs)", mp.name,
             matches.height, n_match, int((matches["matched_entity_ids"] == "").sum()), cp.name, cands.height,
             scored.height)
    return mp, cp


def _read_tsv(path: Path, header: list[str]) -> dict[str, list[str]]:
    rows = {}
    with open(path, encoding="utf-8", newline="") as fh:
        first = fh.readline().rstrip("\n").split("\t")
        if first != header:
            raise SubmissionError(f"{path.name}: header {first} != {header}")
        for ln, line in enumerate(fh, start=2):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                raise SubmissionError(f"{path.name}:{ln}: expected 2 tab-separated columns, got {len(parts)}")
            s1, ids = parts
            if s1 in rows:
                raise SubmissionError(f"{path.name}: duplicate Source-1 row {s1}")
            rows[s1] = [x for x in ids.split(",") if x] if ids else []
    return rows


def validate_submission(cfg: Config, mp: Path, cp: Path) -> dict:
    """Internal validator: fails loudly on any contract violation."""
    s1 = set(load(cfg, "test_source1")["entity_id"].to_list())
    valid = set(load(cfg, "test_source2")["entity_id"].to_list()) | set(load(cfg, "test_source3")["entity_id"].to_list())
    m, c = _read_tsv(mp, MATCH_HEADER), _read_tsv(cp, CAND_HEADER)
    errors = []
    for name, rows in (("matching_results", m), ("candidate_pairs", c)):
        missing, extra = s1 - set(rows), set(rows) - s1
        if missing:
            errors.append(f"{name}: {len(missing)} Source-1 rows missing")
        if extra:
            errors.append(f"{name}: {len(extra)} unknown Source-1 ids")
        for k, ids in rows.items():
            if len(ids) != len(set(ids)):
                errors.append(f"{name}: duplicate ids in row {k}")
                break
        bad = [x for ids in rows.values() for x in ids if x not in valid or not x.startswith(("S2-", "S3-"))]
        if bad:
            errors.append(f"{name}: {len(bad)} ids not in test Source 2/3 (e.g. {bad[:3]})")
    not_in_cands = 0
    for k, ids in m.items():
        cset = set(c.get(k, []))
        not_in_cands += sum(1 for x in ids if x not in cset)
    if not_in_cands:
        errors.append(f"{not_in_cands} matches absent from candidate_pairs")
    rep = {"rows": len(m), "matches": sum(map(len, m.values())), "candidates": sum(map(len, c.values())),
           "empty_match_rows": sum(1 for v in m.values() if not v), "errors": errors}
    if errors:
        for e in errors:
            log.error("Submission validation: %s", e)
        raise SubmissionError("; ".join(errors))
    log.info("Internal submission validation PASSED: %s", {k: v for k, v in rep.items() if k != "errors"})
    return rep


def run_official_validator(cfg: Config, mp: Path, cp: Path) -> bool | None:
    v = Path(cfg.validator_path)
    test_dir = Path(cfg.data_dir) / "test"
    if not v.exists():
        log.info("Official validator not found at %s - skipped (internal validation already passed)", v)
        return None
    if not (test_dir / "test_source1.tsv").exists():
        if (Path(cfg.data_dir) / "test_source1.tsv").exists():
            test_dir = Path(cfg.data_dir)
        else:
            # Export test tables as TSV for the official validator if not found
            test_dir = cfg.art / "dataset_export" / "test"
            test_dir.mkdir(parents=True, exist_ok=True)
            for t in ("test_source1", "test_source2", "test_source3"):
                out_tsv = test_dir / f"{t}.tsv"
                if not out_tsv.exists():
                    load(cfg, t).write_csv(out_tsv, separator="\t")
    cmd = [sys.executable, str(v), "--matching", str(mp), "--candidate", str(cp), "--test-dir", str(test_dir)]
    log.info("Running official validator: %s", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    log.info("validator stdout:\n%s", r.stdout.strip())
    if r.returncode != 0:
        log.error("validator stderr:\n%s", r.stderr.strip())
        raise SubmissionError("Official validator failed")
    return True
