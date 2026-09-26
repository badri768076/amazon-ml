 """Pipeline entry point.

    python src/main.py --stage eda | preprocess | blocking | train | evaluate |
                               inference | submission | all

Every stage checkpoints to ``artifacts/`` and skips work that already exists
(use ``--force`` to recompute), so a failed run resumes where it stopped.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import TEST_TABLES, TRAIN_TABLES, Config, set_seeds, setup_logging  # noqa: E402

log = logging.getLogger("ber.main")
STAGES = ["eda", "preprocess", "blocking", "train", "evaluate", "inference", "submission"]


# --------------------------------------------------------------------------- #
def stage_eda(cfg: Config) -> None:
    from data_loader import materialise_raw, validate_tables
    from evaluation import run_eda
    materialise_raw(cfg)
    validate_tables(cfg)
    run_eda(cfg)


def stage_preprocess(cfg: Config) -> None:
    from preprocessing import preprocess_table
    for t in TRAIN_TABLES + TEST_TABLES:
        if not t.endswith("ground_truth"):
            preprocess_table(cfg, t)


def stage_blocking(cfg: Config) -> None:
    from candidate_generation import candidate_recall, generate_candidates
    from training_pairs import select_train_entities
    ids, gtp = select_train_entities(cfg)
    generate_candidates(cfg, "train", ids, gt=gtp)
    candidate_recall(cfg, "train", gtp)
    generate_candidates(cfg, "test")


def _evaluator(cfg: Config, df, role: str):
    import polars as pl
    from evaluation import GroupEvaluator
    gt = pl.read_parquet(cfg.art / "training" / f"{role}_gt.parquet")
    ents = pl.read_parquet(cfg.art / "training" / f"{role}_entities.parquet")["s1_id"]
    singles = set(ents.to_list()) - set(gt["s1_id"].unique().to_list())
    return GroupEvaluator(df["s1_idx"].to_numpy(), df["label"].to_numpy(), gt.height, singles, ents.len(),
                          s1_ids=df["s1_id"].to_numpy()), gt


def stage_train(cfg: Config) -> None:
    import polars as pl
    from candidate_generation import entity_tables
    from feature_engineering import build_feature_matrix
    from models import PairModels, save_importance, to_xy
    from threshold_optimization import best_threshold
    from training_pairs import build_training_sets, select_train_entities

    done = [cfg.art / "models" / "selection.joblib", cfg.art / "predictions" / "val_probs.parquet",
            cfg.art / "predictions" / "inner_probs.parquet", cfg.report_dir / "model_validation.json"]
    if all(p.exists() for p in done) and not cfg.force:
        log.info("[checkpoint] trained models and validation predictions exist")
        return
    s1, cand = entity_tables(cfg, "train")
    pairs = pl.read_parquet(cfg.art / "candidates" / "train_pairs.parquet")
    feat_dir = build_feature_matrix(cfg, "train", pairs, s1, cand)
    _, gtp = select_train_entities(cfg)
    sets = build_training_sets(cfg, feat_dir, s1, cand, gtp)
    Xtr, ytr = to_xy(sets["train"])
    Xin, yin = to_xy(sets["inner"])
    models = PairModels(cfg).fit(Xtr, ytr, Xin, yin)
    models.save("selection")
    save_importance(cfg, models)

    report = {}
    for role in ("inner", "val"):
        df = sets[role].sort("s1_idx", "cand_idx")
        X, _ = to_xy(df)
        probs = models.predict(X)
        pl.DataFrame({"s1_idx": df["s1_idx"], "cand_idx": df["cand_idx"], "s1_id": df["s1_id"],
                      "cand_id": df["cand_id"], "label": df["label"],
                      **{f"p_{k}": v for k, v in probs.items()}}).write_parquet(
            cfg.path("predictions", f"{role}_probs.parquet"))
        if role == "val":
            ev, _ = _evaluator(cfg, df, "val")
            for name, p in probs.items():
                t, m = best_threshold(p, ev, cfg)
                report[name] = {"best_threshold": t, **m}
                log.info("Validation %-5s t=%.2f  P=%.4f R=%.4f F0.5=%.4f F1=%.4f singleton=%.4f %s", name, t,
                         m["precision"], m["recall"], m["f0_5"], m["f1"], m["singleton_correctness"] or 0,
                         "(baseline)" if name == "lr" else "")
    (cfg.report_dir / "model_validation.json").write_text(json.dumps(report, indent=2))


def stage_evaluate(cfg: Config) -> None:
    import polars as pl
    from candidate_generation import entity_tables
    from ensemble import combine, select_ensemble
    from evaluation import error_analysis
    from models import ENSEMBLE_MODELS
    from threshold_optimization import decision_mask, save_rule, threshold_table

    val = pl.read_parquet(cfg.art / "predictions" / "val_probs.parquet")
    inner = pl.read_parquet(cfg.art / "predictions" / "inner_probs.parquet")
    ev, val_gt = _evaluator(cfg, val, "val")
    vp = {m: val[f"p_{m}"].to_numpy() for m in ENSEMBLE_MODELS}
    ip = {m: inner[f"p_{m}"].to_numpy() for m in ENSEMBLE_MODELS}
    res = select_ensemble(vp, ev, ip, inner["label"].to_numpy(), cfg)
    rule = dict(res["rule"])
    rule.update({"weights": res["weights"], "calibration": res["calibration"]})
    p_final = combine(res["calibrator"].transform(vp), res["weights"])
    table = threshold_table(p_final, ev, cfg, rule["relative_ratio"], rule["max_matches"])
    save_rule(cfg, rule, table)
    pl.DataFrame(res["rule_rows"]).write_csv(cfg.report_dir / "decision_rule_search.tsv", separator="\t")

    mask = decision_mask(p_final, rule["threshold"], rule["relative_ratio"], rule["max_matches"],
                         ev.group_max(p_final), ev.group_rank(p_final))
    final = ev.evaluate(mask)
    base = json.loads((cfg.report_dir / "model_validation.json").read_text())
    summary = {"ensemble": final, "rule": {k: rule[k] for k in ("threshold", "relative_ratio", "max_matches")},
               "weights": res["weights"], "calibration": res["calibration"], "single_models": base}
    (cfg.report_dir / "final_validation.json").write_text(json.dumps(summary, indent=2))
    log.info("FINAL VALIDATION (ensemble): P=%.4f R=%.4f F0.5=%.4f F1=%.4f TP=%d FP=%d FN=%d singleton=%.4f",
             final["precision"], final["recall"], final["f0_5"], final["f1"], final["tp"], final["fp"],
             final["fn"], final["singleton_correctness"] or 0)
    log.info("  vs LR baseline F0.5=%.4f", base["lr"]["f0_5"])
    s1, cand = entity_tables(cfg, "train")
    error_analysis(cfg, val.with_columns(p_final=pl.Series(p_final)).join(
        pl.read_parquet(cfg.art / "training" / "val_pairs.parquet").select(
            "s1_idx", "cand_idx", "name_jw", "addr_tset", "postal_match", "country_match", "source"),
        on=["s1_idx", "cand_idx"], how="left"), mask, val_gt, s1, cand)


def stage_inference(cfg: Config) -> None:
    from inference import predict_test
    predict_test(cfg)


def stage_submission(cfg: Config) -> None:
    import polars as pl
    from submission import run_official_validator, validate_submission, write_submission
    scored = pl.read_parquet(cfg.art / "predictions" / "test_scored.parquet")
    mp, cp = write_submission(cfg, scored)
    validate_submission(cfg, mp, cp)
    run_official_validator(cfg, mp, cp)


RUNNERS = {"eda": stage_eda, "preprocess": stage_preprocess, "blocking": stage_blocking, "train": stage_train,
           "evaluate": stage_evaluate, "inference": stage_inference, "submission": stage_submission}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Business Entity Resolution pipeline")
    ap.add_argument("--stage", required=True, choices=STAGES + ["all"])
    ap.add_argument("--data-dir", help="dataset directory (default: auto-detected from local student_resource/dataset)")
    ap.add_argument("--artifacts-dir")
    ap.add_argument("--output-dir")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                    help="training device: auto (detects RTX 4060 GPU / CUDA), cuda, or cpu")
    ap.add_argument("--rf-max-rows", type=int, help="max training rows for Random Forest (default: 300,000)")
    ap.add_argument("--max-train-s1", type=int, help="cap on labelled train Source-1 entities")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--force", action="store_true", help="recompute checkpoints of the selected stage(s)")
    a = ap.parse_args(argv)

    cfg = Config()
    for k in ("output_dir", "max_train_s1", "seed"):
        if getattr(a, k) is not None:
            setattr(cfg, k, getattr(a, k))
    if a.data_dir:
        cfg.data_dir = a.data_dir
    if a.artifacts_dir:
        cfg.artifacts_dir = a.artifacts_dir
    if a.device:
        cfg.model.device = a.device
    if a.rf_max_rows:
        cfg.model.rf_max_train_rows = a.rf_max_rows
    if a.workers:
        cfg.n_workers = a.workers
    cfg.force = a.force
    setup_logging(cfg)
    set_seeds(cfg.seed)
    cfg.path("config_used.json").write_text(cfg.to_json())
    log.info("Dataset directory: %s | data source: %s | device: %s | artifacts: %s | output: %s",
             cfg.data_dir, cfg.data_source, cfg.model.device, cfg.artifacts_dir, cfg.output_dir)

    for st in (STAGES if a.stage == "all" else [a.stage]):
        t = time.time()
        log.info("===== STAGE %s =====", st.upper())
        RUNNERS[st](cfg)
        log.info("===== STAGE %s done in %.1fs =====", st.upper(), time.time() - t)


if __name__ == "__main__":
    main()
