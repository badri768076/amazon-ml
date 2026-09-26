"""Comprehensive Model Testing & Evaluation Script.

Usage:
    python test_and_evaluate.py
    python test_and_evaluate.py --samples 5
    python test_and_evaluate.py --check-test
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

# Fix Windows terminal UTF-8 encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Add src to python path
SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

import numpy as np
import polars as pl
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from config import Config
from models import ENSEMBLE_MODELS, PairModels, to_xy


def print_banner(title: str) -> None:
    width = 75
    print("\n" + "=" * width)
    print(f" {title.upper()} ".center(width, "="))
    print("=" * width + "\n")


def print_table(headers: list[str], rows: list[list[str]], col_align: list[str] | None = None) -> None:
    """Pretty prints a markdown-style table to the console."""
    str_rows = [[str(cell) for cell in row] for row in rows]
    widths = [len(h) for h in headers]
    for r in str_rows:
        for i, val in enumerate(r):
            widths[i] = max(widths[i], len(val))

    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * widths[i] for i in range(len(headers)))
    print(header_line)
    print(sep_line)
    for r in str_rows:
        line_parts = []
        for i, val in enumerate(r):
            align = col_align[i] if col_align else "left"
            if align == "right":
                line_parts.append(val.rjust(widths[i]))
            else:
                line_parts.append(val.ljust(widths[i]))
        print(" | ".join(line_parts))


def evaluate_models(cfg: Config) -> dict:
    """Evaluates all individual models and the ensemble on the validation pairs."""
    print_banner("1. Model Evaluation on Validation Set")
    
    val_pairs_path = cfg.art / "training" / "val_pairs.parquet"
    if not val_pairs_path.exists():
        print(f"Error: {val_pairs_path} not found. Run stage_train first.")
        return {}

    t0 = time.time()
    val_df = pl.read_parquet(val_pairs_path)
    print(f"Loaded validation set: {val_df.height:,} candidate pairs in {time.time()-t0:.2f}s")
    
    X_val, y_val = to_xy(val_df)
    n_pos = int(y_val.sum())
    n_neg = len(y_val) - n_pos
    print(f"Distribution: {n_pos:,} Positive Matches ({n_pos/len(y_val)*100:.2f}%) | {n_neg:,} Negative Pairs ({n_neg/len(y_val)*100:.2f}%)\n")

    # Load trained models
    model_tag = "selection" if (cfg.art / "models" / "selection.joblib").exists() else "final"
    print(f"Loading trained models from: artifacts/models/{model_tag}.joblib ...")
    models = PairModels.load(cfg, model_tag)
    
    # Load decision rule & ensemble weights
    rule_path = cfg.art / "models" / "decision_rule.json"
    rule = json.loads(rule_path.read_text()) if rule_path.exists() else {"threshold": 0.80, "weights": {"lgbm": 0.4, "xgb": 0.6, "rf": 0.0}}
    weights = rule.get("weights", {"lgbm": 0.4, "xgb": 0.6, "rf": 0.0})
    best_threshold = rule.get("threshold", 0.80)
    print(f"Ensemble Configuration: {weights} | Threshold: {best_threshold}\n")

    # Predict with all models
    preds_raw = models.predict(X_val)
    
    # Compute ensemble probability
    p_ensemble = np.zeros(len(y_val), dtype=np.float32)
    for m_name, w in weights.items():
        if m_name in preds_raw and w > 0:
            p_ensemble += w * preds_raw[m_name]

    results_table = []
    headers = ["Model Name", "Threshold", "Precision", "Recall", "F0.5 Score", "F1 Score", "TP", "FP", "FN"]

    for name in ["lr", "rf", "lgbm", "xgb"]:
        if name in preds_raw:
            probs = preds_raw[name]
            # Use model-specific threshold or default
            thresh = 0.75 if name == "lr" else (0.60 if name == "rf" else (0.85 if name == "lgbm" else 0.80))
            y_pred = (probs >= thresh).astype(int)
            p, r, f1, _ = precision_recall_fscore_support(y_val, y_pred, average="binary", zero_division=0)
            f0_5 = (1.25 * p * r) / (0.25 * p + r) if (0.25 * p + r) > 0 else 0.0
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
            results_table.append([
                name.upper(), f"{thresh:.2f}", f"{p*100:.2f}%", f"{r*100:.2f}%",
                f"{f0_5:.4f}", f"{f1:.4f}", f"{tp:,}", f"{fp:,}", f"{fn:,}"
            ])

    # Ensemble evaluation
    y_ens_pred = (p_ensemble >= best_threshold).astype(int)
    p_ens, r_ens, f1_ens, _ = precision_recall_fscore_support(y_val, y_ens_pred, average="binary", zero_division=0)
    f0_5_ens = (1.25 * p_ens * r_ens) / (0.25 * p_ens + r_ens) if (0.25 * p_ens + r_ens) > 0 else 0.0
    tn, fp, fn, tp = confusion_matrix(y_val, y_ens_pred).ravel()
    results_table.append([
        "ENSEMBLE", f"{best_threshold:.2f}", f"{p_ens*100:.2f}%", f"{r_ens*100:.2f}%",
        f"{f0_5_ens:.4f}", f"{f1_ens:.4f}", f"{tp:,}", f"{fp:,}", f"{fn:,}"
    ])

    print_table(headers, results_table, col_align=["left", "right", "right", "right", "right", "right", "right", "right", "right"])

    # Source 2 vs Source 3 Breakdown
    print_banner("2. Performance Breakdown by Source")
    src_table = []
    src_headers = ["Source", "Total Pairs", "True Matches", "Predicted Matches", "Precision", "Recall", "F0.5"]
    for src, label in [(2, "Source 2"), (3, "Source 3")]:
        mask = (val_df["source"].to_numpy() == src)
        if mask.sum() > 0:
            y_s = y_val[mask]
            pred_s = y_ens_pred[mask]
            p_s, r_s, _, _ = precision_recall_fscore_support(y_s, pred_s, average="binary", zero_division=0)
            f0_5_s = (1.25 * p_s * r_s) / (0.25 * p_s + r_s) if (0.25 * p_s + r_s) > 0 else 0.0
            src_table.append([
                label, f"{mask.sum():,}", f"{int(y_s.sum()):,}", f"{int(pred_s.sum()):,}",
                f"{p_s*100:.2f}%", f"{r_s*100:.2f}%", f"{f0_5_s:.4f}"
            ])
    print_table(src_headers, src_table, col_align=["left", "right", "right", "right", "right", "right", "right"])

    return {"p_ensemble": p_ensemble, "val_df": val_df, "y_val": y_val, "models": models}


def show_feature_importance(cfg: Config) -> None:
    """Displays the top 15 most important features from LightGBM and XGBoost."""
    print_banner("3. Top 15 Feature Importances")
    report_file = cfg.report_dir / "feature_importance.json"
    if not report_file.exists():
        print("Feature importance file not found.")
        return
    data = json.loads(report_file.read_text())
    
    headers = ["Rank", "LightGBM (Gain)", "Gain Score", "Random Forest", "Importance"]
    lgbm_imp = data.get("lgbm_gain") or data.get("lgbm", {})
    rf_imp = data.get("rf") or data.get("xgb", {})
    
    top_lgbm = sorted(lgbm_imp.items(), key=lambda x: x[1], reverse=True)[:15]
    top_rf = sorted(rf_imp.items(), key=lambda x: x[1], reverse=True)[:15]
    
    rows = []
    for i in range(max(len(top_lgbm), len(top_rf))):
        lf_name, lf_val = top_lgbm[i] if i < len(top_lgbm) else ("", 0)
        rf_name, rf_val = top_rf[i] if i < len(top_rf) else ("", 0)
        rows.append([
            f"#{i+1}", lf_name, f"{lf_val:,.1f}", rf_name, f"{rf_val:.4f}"
        ])
    print_table(headers, rows, col_align=["right", "left", "right", "left", "right"])


def show_sample_predictions(cfg: Config, eval_data: dict, n_samples: int = 4) -> None:
    """Displays real examples of matched and non-matched entity pairs."""
    print_banner(f"4. Qualitative Inspection ({n_samples} Sample Predictions per Category)")
    
    val_df = eval_data["val_df"]
    y_val = eval_data["y_val"]
    p_ens = eval_data["p_ensemble"]
    
    s1_table_path = cfg.art / "candidates" / "train_s1.parquet"
    cand_table_path = cfg.art / "candidates" / "train_cand.parquet"
    if not s1_table_path.exists() or not cand_table_path.exists():
        print("Entity text tables not found, skipping entity string display.")
        return

    s1_df = pl.read_parquet(s1_table_path, columns=["idx", "normalized_name", "normalized_address", "normalized_country"])
    cand_df = pl.read_parquet(cand_table_path, columns=["idx", "normalized_name", "normalized_address", "normalized_country"])
    
    df_with_p = val_df.with_columns(prob=pl.Series(p_ens), true_label=pl.Series(y_val))

    categories = [
        ("High-Confidence True Matches (True Positives)", (df_with_p["prob"] > 0.90) & (df_with_p["true_label"] == 1)),
        ("High-Confidence Non-Matches (True Negatives)", (df_with_p["prob"] < 0.10) & (df_with_p["true_label"] == 0)),
        ("Borderline Cases (Probabilities ~ 0.50)", (df_with_p["prob"] >= 0.45) & (df_with_p["prob"] <= 0.55)),
    ]

    for cat_name, condition in categories:
        sub = df_with_p.filter(condition)
        if sub.height == 0:
            continue
        sample = sub.head(n_samples)
        
        # Join with text
        joined = (sample.join(s1_df.rename({"normalized_name": "s1_name", "normalized_address": "s1_addr", "normalized_country": "s1_country"}),
                              left_on="s1_idx", right_on="idx", how="left")
                        .join(cand_df.rename({"normalized_name": "c_name", "normalized_address": "c_addr", "normalized_country": "c_country"}),
                              left_on="cand_idx", right_on="idx", how="left"))
        
        print(f"\n--- {cat_name} ---")
        for i, row in enumerate(joined.iter_rows(named=True)):
            print(f"[{i+1}] Probability: {row['prob']:.4f} | Label: {row['true_label']} | Source: S{row['source']}")
            print(f"    S1   : \"{row['s1_name']}\" | Addr: \"{row['s1_addr']}\" ({row['s1_country']})")
            print(f"    Cand : \"{row['c_name']}\" | Addr: \"{row['c_addr']}\" ({row['c_country']})")
            print()


def check_submission_files(cfg: Config) -> None:
    """Verifies and summarizes the final generated submission TSVs."""
    print_banner("5. Submission Files Diagnostics")
    out_dir = Path(cfg.output_dir)
    m_path = out_dir / "matching_results.tsv"
    c_path = out_dir / "candidate_pairs.tsv"

    if not m_path.exists() or not c_path.exists():
        print(f"Submission files not found in {out_dir}. Run stage_submission to generate them.")
        return

    m_size_mb = m_path.stat().st_size / (1024 * 1024)
    c_size_mb = c_path.stat().st_size / (1024 * 1024)
    
    print(f"Found matching_results.tsv : {m_size_mb:.1f} MB ({m_path})")
    print(f"Found candidate_pairs.tsv  : {c_size_mb:.1f} MB ({c_path})")
    
    # Read first 5 lines as preview
    print("\nPreview of matching_results.tsv (first 5 rows):")
    with open(m_path, "r", encoding="utf-8") as f:
        for _ in range(6):
            print("  " + f.readline().rstrip())

    # Load validation summary if present
    final_val_path = cfg.report_dir / "final_validation.json"
    if final_val_path.exists():
        v = json.loads(final_val_path.read_text())
        ens = v.get("ensemble", {})
        print("\nSummary of Final Ensemble Metrics:")
        print(f"  - Precision : {ens.get('precision', 0)*100:.2f}%")
        print(f"  - Recall    : {ens.get('recall', 0)*100:.2f}%")
        print(f"  - F0.5 Score: {ens.get('f0_5', 0):.4f}")
        print(f"  - F1 Score  : {ens.get('f1', 0):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test and Evaluate Trained Business Entity Resolution Models")
    parser.add_argument("--samples", type=int, default=3, help="Number of qualitative sample predictions to display")
    parser.add_argument("--no-samples", action="store_true", help="Skip qualitative sample printing")
    args = parser.parse_args()

    cfg = Config()
    eval_data = evaluate_models(cfg)
    if eval_data:
        show_feature_importance(cfg)
        if not args.no_samples:
            show_sample_predictions(cfg, eval_data, n_samples=args.samples)
    check_submission_files(cfg)
    print_banner("Evaluation Complete! Safe & Ready for Submission")


if __name__ == "__main__":
    main()
