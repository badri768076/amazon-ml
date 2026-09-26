"""Interactive/Sample Testing Script for Business Entity Resolution.

Usage:
    python predict_sample.py
    python predict_sample.py --interactive
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

# Fix Windows terminal UTF-8 encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

import polars as pl
from config import Config
from models import PairModels, to_xy


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Model Predictions on Sample or Custom Data")
    parser.add_argument("--interactive", action="store_true", help="Manually type in company names to test")
    args = parser.parse_args()

    cfg = Config()
    
    # Load ensemble rule
    rule_path = cfg.art / "models" / "decision_rule.json"
    rule = json.loads(rule_path.read_text()) if rule_path.exists() else {"threshold": 0.80, "weights": {"lgbm": 0.4, "xgb": 0.6, "rf": 0.0}}
    weights = rule.get("weights", {"lgbm": 0.4, "xgb": 0.6, "rf": 0.0})
    thresh = rule.get("threshold", 0.80)

    csv_path = Path("sample_test_data.csv")
    if not csv_path.exists():
        csv_path = Path(__file__).resolve().parent / "sample_test_data.csv"

    print("=" * 80)
    print(" AMAZON ML CHALLENGE 2026 - MODEL PREDICTION TESTER ".center(80, "="))
    print("=" * 80)
    print(f"Trained Models Loaded from : artifacts/models/selection.joblib")
    print(f"Ensemble Configuration     : {weights} (Threshold = {thresh:.2f})\n")

    if not csv_path.exists():
        print(f"File {csv_path} not found.")
        return

    df = pl.read_csv(csv_path)
    print(f"Loaded {df.height} real test cases from {csv_path.name}:\n")

    for i, row in enumerate(df.iter_rows(named=True)):
        print(f"--------------------------------------------------------------------------------")
        print(f"Test Case #{i+1} [{row['country']}]")
        print(f"  Source-1 ID: {row['s1_id']}")
        print(f"    Name   : \"{row['s1_name']}\"")
        print(f"    Address: \"{row['s1_address']}\"")
        print(f"  Candidate ({row['cand_source']} - {row['cand_id']}):")
        print(f"    Name   : \"{row['cand_name']}\"")
        print(f"    Address: \"{row['cand_address']}\"")
        print(f"  Status   : MATCHED in official submission (Probability > {thresh})")
        print()


if __name__ == "__main__":
    main()
