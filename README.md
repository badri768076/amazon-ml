# Amazon ML Challenge - Business Entity Resolution

This repository contains the end-to-end Machine Learning solution for the Amazon ML Challenge on Business Entity Resolution.

## Project Structure
```text
.
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── blocking.py               # Blocking strategies & index generation
│       │   ├── candidate_generation.py   # Multi-stage candidate retrieval
│       │   ├── config.py                 # Central pipeline configuration & path auto-detection
│       │   ├── data_loader.py            # TSV/Parquet loader & DuckDB EDA
│       │   ├── ensemble.py               # Model weighting & decision rules
│       │   ├── evaluation.py             # F0.5 & PR evaluation metrics
│       │   ├── feature_engineering.py    # 80+ lexical, phonetic & token features
│       │   ├── inference.py              # Test set prediction & ranking
│       │   ├── main.py                   # Pipeline orchestrator (--stage flags)
│       │   ├── models.py                 # LightGBM, XGBoost, CatBoost & Logistic Regression
│       │   ├── preprocessing.py          # Vectorized Polars normalization
│       │   ├── submission.py             # TSV formatting & validation check
│       │   ├── threshold_optimization.py # F0.5 threshold search
│       │   └── training_pairs.py         # Hard negative mining & pair creation
│       ├── predict_sample.py             # Quick inference test on sample pairs
│       ├── test_and_evaluate.py          # Validation evaluation & model diagnostics
│       ├── export_samples.py             # Diagnostic sample exporter
│       ├── requirements.txt              # Python dependencies
│       └── README.md                     # Technical pipeline docs
├── output/                               # Generated submission TSVs (git-ignored)
│   ├── candidate_pairs.tsv
│   └── matching_results.tsv
├── student_resource/                     # Competition datasets & validator (git-ignored)
├── Documentation_template.md             # Competition documentation & approach report
├── sample_test_data.csv                  # Sample test records for quick testing
└── README.md
```

## Setup & Quickstart

### 1. Requirements
Install the project dependencies:
```bash
pip install -r code/business_entity_resolution/requirements.txt
```

### 2. Testing Predictions
Run a fast sanity check on sample data:
```bash
python code/business_entity_resolution/predict_sample.py
```

### 3. Model Evaluation & Diagnostics
Evaluate the trained ensemble against the validation split:
```bash
python code/business_entity_resolution/test_and_evaluate.py
```

### 4. Running the Full Pipeline
To re-run any stage or the complete pipeline:
```bash
python code/business_entity_resolution/src/main.py --stage all
```

### 5. Packaging Submission Zip
To package the required files for competition submission:
```powershell
Compress-Archive -Path code, output, Documentation_template.md -DestinationPath TEAM_submission.zip -Force
```
