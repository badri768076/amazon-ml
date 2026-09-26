"""Central configuration for the Business Entity Resolution pipeline.

Every path, seed, limit and model parameter lives here. Values can be
overridden with environment variables (prefix ``BER_``) or from the command
line in ``main.py``; nothing else in the project hard-codes paths.
"""
from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# code/business_entity_resolution/src/config.py -> project root = code/business_entity_resolution
PROJECT_DIR = Path(__file__).resolve().parents[1]
# TEAM_submission/ (two levels above the project dir)
SUBMISSION_ROOT = PROJECT_DIR.parents[1]


def resolve_data_root(custom_path: str | Path | None = None) -> Path:
    """Intelligently locate the dataset root directory on the local system or Kaggle."""
    candidates: list[Path] = []
    if custom_path:
        candidates.append(Path(custom_path))
    env_dir = os.environ.get("BER_DATA_DIR")
    if env_dir:
        candidates.append(Path(env_dir))

    for parent in [Path.cwd(), PROJECT_DIR, *PROJECT_DIR.parents]:
        candidates.extend([
            parent / "student_resource" / "student_resource" / "dataset",
            parent / "student_resource" / "dataset",
            parent / "dataset",
        ])
    candidates.extend([
        Path("/kaggle/input/datasets/adityakollapudi/student-resource-amazon-ml"),
        Path("/kaggle/input/student-resource-amazon-ml"),
    ])

    for c in candidates:
        if c.exists() and c.is_dir():
            has_files = (
                any(c.glob("*source*.tsv")) or any(c.glob("*source*.parquet"))
                or (c / "train").exists() or (c / "test").exists()
                or (c / "student_resource" / "dataset").exists()
            )
            if has_files:
                if (c / "dataset").is_dir() and ((c / "dataset" / "train").exists() or (c / "dataset" / "test").exists()):
                    return (c / "dataset").resolve()
                if (c / "student_resource" / "dataset").is_dir():
                    return (c / "student_resource" / "dataset").resolve()
                return c.resolve()

    for c in candidates:
        if c.exists():
            return c.resolve()
    return (PROJECT_DIR.parents[-1] / "student_resource" / "student_resource" / "dataset").resolve()


def resolve_validator_path(custom_path: str | Path | None = None) -> Path:
    """Intelligently locate the official submission validator script."""
    candidates: list[Path] = []
    if custom_path:
        candidates.append(Path(custom_path))
    env_v = os.environ.get("BER_VALIDATOR")
    if env_v:
        candidates.append(Path(env_v))

    for parent in [Path.cwd(), PROJECT_DIR, *PROJECT_DIR.parents]:
        candidates.extend([
            parent / "student_resource" / "student_resource" / "utils" / "validate_submission.py",
            parent / "student_resource" / "utils" / "validate_submission.py",
            parent / "utils" / "validate_submission.py",
        ])
    for c in candidates:
        if c.exists() and c.is_file():
            return c.resolve()
    return (SUBMISSION_ROOT / "utils" / "validate_submission.py").resolve()


def detect_device() -> str:
    """Detect if NVIDIA GPU (CUDA) is available for 4060 GPU acceleration."""
    env_dev = os.environ.get("BER_DEVICE")
    if env_dev:
        return env_dev.lower().strip()
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    try:
        import xgboost as xgb
        dm = xgb.DMatrix(np.zeros((1, 1), dtype=np.float32), label=np.zeros(1))
        bst = xgb.train({"device": "cuda", "tree_method": "hist"}, dm, num_boost_round=1)
        return "cuda"
    except Exception:
        pass
    return "cpu"


# Resolved local or Kaggle dataset directory
DATA_ROOT = resolve_data_root()


def _env(name: str, default):
    """Read ``BER_<NAME>`` from the environment, cast to the default's type."""
    raw = os.environ.get(f"BER_{name.upper()}")
    if raw is None:
        return default
    if isinstance(default, bool):
        return raw.strip().lower() in {"1", "true", "yes", "y"}
    if isinstance(default, int) and not isinstance(default, bool):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    if default is None:
        return None if raw.strip().lower() in {"", "none", "null"} else raw
    return raw


# --------------------------------------------------------------------------- #
# Table / schema constants
# --------------------------------------------------------------------------- #
SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
TRAIN_TABLES = ["train_source1", "train_source2", "train_source3", "train_ground_truth"]
TEST_TABLES = ["test_source1", "test_source2", "test_source3"]
ALL_TABLES = TRAIN_TABLES + TEST_TABLES
SOURCE_PREFIX = {"source1": "S1-", "source2": "S2-", "source3": "S3-"}


@dataclass
class BlockingConfig:
    # Candidate-side block size cap: keys shared by more candidates than this
    # are considered uninformative (too common) and dropped.
    max_block_size_exact: int = _env("max_block_exact", 5000)
    max_block_size_token: int = _env("max_block_token", 1000)
    max_block_size_ngram: int = _env("max_block_ngram", 600)
    max_block_size_addr: int = _env("max_block_addr", 600)
    max_block_size_combo: int = _env("max_block_combo", 1500)
    # Number of rarest keys per record used for token / n-gram strategies.
    name_tokens_per_record: int = 3
    addr_tokens_per_record: int = 3
    ngrams_per_record: int = 4
    ngram_size: int = 3
    min_token_len: int = 2
    # Enabled blocking strategies; char_ngram is disabled by default to prevent 500M row OOM
    enabled_strategies: list[str] = field(default_factory=lambda: [
        "exact_name", "exact_core", "name_signature", "name_token",
        "addr_token", "postal_name", "postal_building", "token_postal", "addr_signature"
    ])
    # Final cap on candidates kept per Source-1 entity *per candidate source*.
    max_candidates_per_source: int = _env("max_cands_per_source", 25)
    # Two-stage capping: keep the top ``pre_rank_keep`` pairs per (S1, source) by
    # blocking score, re-rank them with a cheap RapidFuzz similarity, then keep
    # ``max_candidates_per_source``.
    pre_rank_keep: int = _env("pre_rank_keep", 200)
    # Strategy weights for the blocking score (score = sum w * log1p(N/df)).
    weights: dict = field(default_factory=lambda: {
        "exact_name": 6.0,
        "exact_core": 5.0,
        "name_signature": 4.0,
        "name_token": 1.0,
        "char_ngram": 0.6,
        "addr_token": 0.5,
        "postal_name": 3.0,
        "postal_building": 2.0,
        "token_postal": 2.0,
        "addr_signature": 3.0,
        "fuzzy": 2.0,
    })
    # Fuzzy fallback (TF-IDF char n-gram retrieval + RapidFuzz re-rank).
    fuzzy_enabled: bool = _env("fuzzy_enabled", False)
    fuzzy_min_candidates: int = 3          # run fallback when a source has fewer candidates
    fuzzy_top_k_retrieve: int = 40         # sparse cosine retrieval depth
    fuzzy_top_k_keep: int = 8              # kept after RapidFuzz re-rank
    fuzzy_min_score: float = 0.35          # cosine floor for retrieval
    fuzzy_max_postings: int = 20000        # drop n-grams shared by more candidates than this
    fuzzy_query_batch: int = 512
    s1_chunk_size: int = _env("s1_chunk", 5000)


@dataclass
class ModelConfig:
    device: str = _env("device", "auto")   # "auto" | "cuda" | "cpu"
    lr_C: float = 1.0
    lr_max_iter: int = 2000
    lgbm_params: dict = field(default_factory=lambda: {
        "objective": "binary", "learning_rate": 0.05, "num_leaves": 63,
        "min_child_samples": 40, "feature_fraction": 0.8, "bagging_fraction": 0.8,
        "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1,
    })
    lgbm_rounds: int = 2000
    xgb_params: dict = field(default_factory=lambda: {
        "objective": "binary:logistic", "eval_metric": "logloss", "eta": 0.05,
        "max_depth": 8, "subsample": 0.8, "colsample_bytree": 0.8,
        "min_child_weight": 5, "lambda": 1.0, "tree_method": "hist",
    })
    xgb_rounds: int = 2000
    early_stopping_rounds: int = 100
    rf_params: dict = field(default_factory=lambda: {
        "n_estimators": 200, "max_depth": 16, "min_samples_leaf": 5,
        "max_features": "sqrt", "max_samples": 0.5, "class_weight": None,
    })
    rf_max_train_rows: int = _env("rf_max_train_rows", 300_000)     # RF subsample cap (lowered from 1.5M for safe local RAM usage)
    ensemble_weight_step: float = 0.1
    calibration: str = "auto"              # "none" | "platt" | "isotonic" | "auto"
    refit_on_full: bool = True             # refit on train+val after selection


@dataclass
class Config:
    # ---------------- data ----------------
    data_dir: str = field(default_factory=lambda: _env("data_dir", str(resolve_data_root())))

    @property
    def local_data_dir(self) -> str:
        return self.data_dir

    @local_data_dir.setter
    def local_data_dir(self, val: str) -> None:
        self.data_dir = val

    @property
    def data_source(self) -> str:
        d = str(self.data_dir)
        return "kaggle" if ("/kaggle" in d or Path("/kaggle").exists()) else "local"

    # ---------------- paths ----------------
    artifacts_dir: str = _env("artifacts_dir", str(PROJECT_DIR / "artifacts"))
    output_dir: str = _env("output_dir", str(SUBMISSION_ROOT / "output"))
    validator_path: str = field(default_factory=lambda: _env("validator", str(resolve_validator_path())))

    # ---------------- run control ----------------
    seed: int = _env("seed", 42)
    val_ratio: float = _env("val_ratio", 0.2)
    inner_holdout_ratio: float = 0.1                           # early stopping / calibration
    max_train_s1: int | None = _env("max_train_s1", 100_000)   # 100k S1 entities (~2.5M pairs) fits in 13-16GB RAM
    gt_absent_policy: str = _env("gt_absent_policy", "auto")   # "auto" | "no_match" | "exclude"
    neg_hard_per_s1: int = 20                                  # hardest negatives kept per S1
    neg_random_per_s1: int = 5                                 # extra random in-block negatives
    n_workers: int = _env("workers", max(1, os.cpu_count() or 1))
    feature_chunk_pairs: int = _env("feature_chunk", 200_000)  # 200k pairs for safe laptop RAM usage
    tfidf_fit_sample: int = 600_000
    force: bool = False                                        # recompute checkpoints

    # ---------------- decision rule search ----------------
    thresholds: list = field(default_factory=lambda: (
        [round(x, 2) for x in np.arange(0.10, 0.951, 0.05)] + [0.97, 0.99]))
    relative_ratios: list = field(default_factory=lambda: [0.0, 0.5, 0.7, 0.85])
    max_matches_options: list = field(default_factory=lambda: [0, 3, 10])   # 0 = unlimited

    # ---------------- postal patterns (generic, country-agnostic) ----------------
    postal_regex: str = r"\b\d{5,6}\b"

    blocking: BlockingConfig = field(default_factory=BlockingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    # ---------------- derived paths ----------------
    @property
    def art(self) -> Path:
        return Path(self.artifacts_dir)

    def path(self, *parts: str) -> Path:
        p = self.art.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def raw_dir(self) -> Path:
        return self.art / "raw"

    @property
    def norm_dir(self) -> Path:
        return self.art / "normalized"

    @property
    def report_dir(self) -> Path:
        d = self.art / "reports"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_logging(cfg: Config, level: int = logging.INFO) -> logging.Logger:
    log_path = cfg.path("logs", "pipeline.log")
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
                        force=True)
    return logging.getLogger("ber")
