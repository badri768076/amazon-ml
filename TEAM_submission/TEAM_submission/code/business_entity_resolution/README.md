# Business Entity Resolution — Amazon ML Challenge 2026

A scalable, reproducible, checkpointed pipeline that decides which **Source 2** and
**Source 3** records describe the same real-world business as each **Source 1**
entity, and writes the two official submission files:

```
output/matching_results.tsv
output/candidate_pairs.tsv
```

The approach: **multi-strategy blocking → two-stage candidate capping → ~80 pairwise
similarity features → Logistic Regression baseline + LightGBM / XGBoost / Random Forest
ensemble → entity-level decision rule tuned for F0.5**.

---

## 1. Project overview

| Stage | Module | What it does |
|---|---|---|
| Load & validate | `data_loader.py` | Supplied Kaggle dataset (Parquet/TSV/CSV), required columns only, schema + content checks |
| EDA | `evaluation.py` (`run_eda`) | DuckDB queries straight on Parquet: sizes, missing values, duplicates, countries, text stats, ground-truth distribution |
| Normalise | `preprocessing.py` | Conservative name/address/country normalisation as vectorised Polars expressions |
| Block | `blocking.py` | 10 key-based strategies, country-scoped, frequency-capped inverted indexes |
| Candidates | `candidate_generation.py` | Union, fuzzy fallback (sparse TF-IDF + RapidFuzz), two-stage capping, recall measurement |
| Pairs | `training_pairs.py` | Source-1 entity-level split, positives + hard negatives |
| Features | `feature_engineering.py` | Name, address, numeric/postal, country, cross-field, blocking and group-context features |
| Models | `models.py` | LR baseline, LightGBM, XGBoost, Random Forest |
| Ensemble | `ensemble.py` | Weighted probabilities, simplex weight search, optional Platt/isotonic calibration |
| Decision | `threshold_optimization.py` | Threshold / relative-to-best / top-K rule chosen on validation F0.5 |
| Inference | `inference.py` | Refit on all labelled data, score test candidates, apply rule |
| Submission | `submission.py` | Deterministic TSVs + internal validator + official validator hook |
| Orchestration | `main.py` | `--stage` CLI, checkpoints, logging |
| Settings | `config.py` | Every path, seed, cap, parameter and grid |

## 2. Problem statement

Source 1 is the de-duplicated reference. For every Source 1 entity the system must output
**all** matching S2/S3 records — zero, one or many. One-to-one matching is never assumed and
`argmax` is never used on its own. Wrong merges are costly, so the target metric is
**F0.5** (precision weighted over recall).

## 3–4. Dataset source (Local System & Kaggle)

The data loader automatically auto-detects the dataset whether running locally or on Kaggle:

- **Local system (default):** `student_resource/student_resource/dataset` (or `student_resource/dataset`, `dataset/`)
- **Kaggle fallback:** `/kaggle/input/datasets/adityakollapudi/student-resource-amazon-ml` (or `/kaggle/input/student-resource-amazon-ml`)
- **Custom override:** `--data-dir <path>` or `BER_DATA_DIR=<path>`

No external downloads or internet connections are required.

## 5. Dataset structure

| Table | Columns |
|---|---|
| `train_source1`, `train_source2`, `train_source3`, `test_source1`, `test_source2`, `test_source3` | `entity_id`, `business_name`, `business_address`, `country` |
| `train_ground_truth` | `source1_entity_id`, `matched_entity_ids` (comma-separated S2/S3 ids, may be empty) |

ID prefixes: `S1-` Source 1, `S2-` Source 2, `S3-` Source 3. `country` is an **open set**:
the test data also contains France, which never appears in training.

## 6. Dataset loading & Hardware Acceleration

The data loader (`src/data_loader.py`):

* Auto-detects the local dataset directory and handles nested, sharded, TSV, CSV, or Parquet tables.
* Filters out Mac metadata (`__MACOSX`, `._*`) and unneeded columns to keep RAM bounded.
* Reads directly using Polars streaming.
* Materialises each table once to `artifacts/raw/<table>.parquet` for lazy reads via Polars / DuckDB.
* **GPU Acceleration (RTX 4060):** XGBoost automatically uses CUDA (`--device auto` or `--device cuda`), speeding up training from minutes to seconds, with automatic fallback to CPU if CUDA is not available.
* Reference loader: `data_loader.load_dataset_pandas_reference()` (also aliased as `load_kaggle_pandas_reference()`).

```bash
# Run with automatic dataset detection and GPU acceleration:
python src/main.py --stage all
```

## 7–8. Environment setup and installation (Local Laptop with RTX 4060 GPU)

On Windows (PowerShell):
```powershell
cd code\business_entity_resolution
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src\main.py --stage all
```

On Linux / macOS:
```bash
cd code/business_entity_resolution
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/main.py --stage all
```

Outputs go to `TEAM_submission/output/` (`matching_results.tsv` and `candidate_pairs.tsv`).
Intermediate files and checkpoints go to `code/business_entity_resolution/artifacts/`.

## 9. Configuration

Everything lives in `src/config.py`. Common overrides are available as environment variables
(`BER_<NAME>`) or CLI flags:

| Setting | Env / CLI | Default |
|---|---|---|
| dataset dir | `BER_DATA_DIR`, `--data-dir` | Auto-detected from `student_resource/dataset` |
| hardware device | `BER_DEVICE`, `--device` | `auto` (detects RTX 4060 GPU / CUDA) |
| artifacts dir | `BER_ARTIFACTS_DIR`, `--artifacts-dir` | `artifacts/` |
| output dir | `BER_OUTPUT_DIR`, `--output-dir` | `TEAM_submission/output` |
| seed | `BER_SEED`, `--seed` | 42 |
| validation ratio | `BER_VAL_RATIO` | 0.2 |
| labelled S1 cap | `BER_MAX_TRAIN_S1`, `--max-train-s1` | 100,000 (fits in 16GB RAM) |
| RF max train rows | `BER_RF_MAX_ROWS`, `--rf-max-rows` | 300,000 (safe memory cap) |
| candidates per source | `BER_MAX_CANDS_PER_SOURCE` | 25 (so ≤ 50 per S1) |
| pre-rank depth | `BER_PRE_RANK_KEEP` | 200 |
| block-size caps | `BER_MAX_BLOCK_{EXACT,TOKEN,NGRAM,ADDR,COMBO}` | 5000 / 1000 / 600 / 600 / 1500 |
| S1 chunk size | `BER_S1_CHUNK` | 5000 |
| feature chunk | `BER_FEATURE_CHUNK` | 200 000 pairs (memory-optimized) |
| workers | `BER_WORKERS`, `--workers` | all CPU cores |
| ground-truth policy | `BER_GT_ABSENT_POLICY` | `auto` (see §16) |
| official validator | `BER_VALIDATOR` | Auto-detected from `student_resource/utils/` |

## 10. EDA

`python src/main.py --stage eda` validates every table (writing
`artifacts/reports/validation_report.json`, and failing on duplicate IDs, bad prefixes, missing
columns or corrupt files), then profiles the data with DuckDB:

* row counts, missing / empty names, addresses and countries,
* duplicate names, addresses, full records and entity IDs,
* train vs. test country distributions, plus the countries that appear only in test,
* name/address length, word counts, digit and non-ASCII shares, postal-like tokens, top tokens,
  and a Unicode-block character distribution,
* ground truth: zero / one / multi-match counts, S2 vs S3 matches, the match-count histogram,
  and S1 entities with several S2 matches.

Results are in `artifacts/reports/eda.json` and `eda.md`.

## 11. Preprocessing

Rules are written once as Polars expressions (Rust regex, no Python loops). The scalar
`normalize_name()`, `normalize_address()` and `normalize_country()` run the same expressions,
so single records and batches always agree.

* **Names:** NFKC → lowercase → strip Latin accents only (Devanagari and other scripts are
  untouched) → collapse dotted acronyms (`S.A.S.` → `sas`, `J.P.` → `jp`) → drop the `M/s`
  honorific → punctuation to spaces, `&` → `and` → token-level mapping of legal forms and safe
  abbreviations (`pvt` → `private`, `ltd` → `limited`, `corp` → `corporation`, `mfg` →
  `manufacturing`, …). For example, `ABC Technologies Pvt. Ltd.` becomes
  `abc technologies private limited`.
* **Core name:** the normalised name without legal-form and filler tokens (`abc technologies`).
  Used for blocking and for features.
* **Addresses:** the same cleaning, plus PIN repair (`560 001` → `560001`), ZIP+4 → ZIP,
  `#` → `number`, letter/digit splitting (`plot12` → `plot 12`) and street-type abbreviations
  (`rd` → `road`, `av`/`ave` → `avenue`, `nr` → `near`, `opp` → `opposite`, …).
* **Country:** alias map to ISO-like codes (`USA`, `United States` → `us`; `India`, `IND` → `in`;
  `France` → `fr`, …). **Unknown values pass through unchanged (open set)**, and missing
  stays null.
* **Derived fields:** `normalized_name`, `normalized_address`, `normalized_country`, `name_core`,
  `name_signature` (sorted tokens), `name_acronym`, `name_tokens`, `address_tokens`,
  `numeric_tokens`, `postal_code_candidates`, `primary_postal`, `building_number`,
  `address_signature`, lengths and word counts. Character n-grams are not stored per row;
  the blocking index and the TF-IDF matrices hold them.

No external geocoding or business data is used.

## 12. Blocking

No S1×S2 or S1×S3 Cartesian product is ever built. Each strategy emits hashed keys. The
candidate side becomes an inverted index, and keys shared by more candidates than a cap are
dropped as uninformative. Source 1 chunks are hash-joined against each index.

| # | Strategy | Key |
|---|---|---|
| 1 | Country scoping | every key below except 2a/2b is prefixed with the country (see below) |
| 2a | `exact_name` | exact normalised name (not country-scoped, so it survives a wrong or missing country) |
| 2b | `exact_core` | exact core name (not country-scoped) |
| 3 | `name_token` | the 3 rarest informative core tokens of the S1 record |
| 4 | `char_ngram` | the 4 rarest character 3-grams of the core name (typos, transliteration) |
| 5 | `addr_token` | the 3 rarest alphabetic address tokens |
| 6 | `postal_building` | postal/PIN + building number |
| 7 | `postal_name`, `token_postal`, `name_signature`, `addr_signature` | postal + name prefix, rare name token + postal, sorted name tokens (word order), sorted address tokens |
| 8 | `fuzzy` | TF-IDF char-3-gram sparse cosine retrieval (country-partitioned), re-ranked by RapidFuzz `token_set_ratio`. It runs only for S1 records with fewer than 3 candidates from a source |

**Country scoping without excluding missing countries.** A candidate with country *c* emits
`c|k` and `*|k`; a candidate with no country emits `?|k` and `*|k`. A Source 1 record with
country *c* queries `c|k` and `?|k`; one with no country queries `*|k`. Same-country records
meet, cross-country records don't, and records without a country are never excluded.
Unseen countries such as France need no special handling.

## 13. Candidate generation

For each Source 1 entity, every strategy's pairs are unioned. Duplicates are merged, and each
pair gets a blocking score Σ wₛ·log(1 + N/df) plus a bitmask of the strategies that found it.
Then, per (S1, source):

1. keep the top `pre_rank_keep` (200) by blocking score,
2. re-rank with a cheap RapidFuzz score (name token-set, name ratio, address token-set,
   blocking score),
3. keep the top `max_candidates_per_source` (25).

The result, `artifacts/candidates/<split>_pairs.parquet`, is **exactly** the pair set that is
featurised, scored and written to `candidate_pairs.tsv`.

## 14. Candidate recall

`artifacts/reports/candidate_recall_train.json` reports:

* recall = true matches retrieved / all true matches, overall and per source,
* pre-cap recall, which shows how much the cap costs,
* hits and unique hits per strategy,
* mean, median and max candidates per S1, and S1 entities with zero candidates,
* reduction ratio = 1 − pairs / (|S1| × |S2 + S3|).

## 15. Feature engineering (≈ 80 features, no IDs)

* **Name:** Levenshtein, Indel ratio, Jaro, Jaro-Winkler, token-sort, token-set, partial ratio
  on the full and core names; prefix similarity; char-n-gram TF-IDF cosine; word TF-IDF cosine;
  token Jaccard / overlap / common count / common ratio; length difference and ratio;
  word-count difference; exact and exact-core match; acronym match.
* **Address:** Levenshtein, Jaro, Jaro-Winkler, token-sort / set / partial, char TF-IDF cosine,
  token Jaccard / overlap / common; numeric-token overlap and Jaccard; postal exact, prefix,
  both-present and conflict; building-number match; length difference; exact match. These are
  set to NaN when either address is missing.
* **Cross-field:** country match (NaN if either is missing), country-missing flag, combined
  name + address token-set, name/address mean, source indicator (S3), missing name / address
  flags.
* **Blocking:** score, quick score, rank, number of strategies, one flag per strategy.
* **Group context** (per S1, label-free): number of candidates, similarity rank and gap to the
  best candidate, candidates sharing the same name, and the number of exact-core matches.
  These help separate "several true duplicates" from "one match plus look-alikes".

Features are computed in chunks that never split an S1 group, and each chunk is written to
`artifacts/features/<split>/part-*.parquet`.

## 16. Training

* **Labelled scope:** `gt_absent_policy=auto`. If the ground-truth file lists explicit empty
  rows, S1 entities missing from it are treated as unlabelled and excluded; otherwise they are
  treated as having zero matches.
* **Split:** by **Source 1 entity**: 80 % train, 20 % validation, and a further 10 % of the
  training entities as an inner hold-out for early stopping and calibration. Pairs are never
  split randomly. An assertion checks that no S1 entity appears in both train and validation.
* **Positives:** ground-truth pairs retrieved by blocking.
* **Negatives:** hard negatives only, meaning in-block non-matches (same country, similar name
  or address, shared postal code or tokens). The 20 hardest by quick score are kept, plus 5
  random in-block negatives per S1. There are no random cross-product negatives. Validation
  keeps the full candidate set.
* **Models:** Logistic Regression baseline (scaled, NaN → −1); LightGBM and XGBoost (native NaN
  handling, early stopping on the inner hold-out); Random Forest (NaN → −1). All use the same
  split and data.

## 17. Ensemble

`P_final = w1·P_LGBM + w2·P_XGB + w3·P_RF`. Equal weights are evaluated first, then all 66
weight vectors on a 0.1-step simplex, each at its own best threshold. The best validation F0.5
wins. Platt and isotonic calibration (fitted on the inner hold-out) are also tried and kept
only if they raise validation F0.5 by more than 0.001.

## 18–19. Validation and F0.5 optimisation

Metrics are entity-level and micro-averaged over matched IDs. Ground-truth matches that
blocking missed count as **false negatives**, so the validation score is honest end to end.

```
F0.5 = 1.25·P·R / (0.25·P + R)
```

The decision rule keeps candidate *j* when `p_j ≥ t`, `p_j ≥ r·max(p)` and `rank_j ≤ K`, with
t ∈ {0.10, 0.15, …, 0.95, 0.97, 0.99}, r ∈ {0, 0.5, 0.7, 0.85} and K ∈ {∞, 3, 10}. All
candidates that pass are kept (multi-match). If none pass, the entity gets an empty list
(singleton). Ties are broken by precision, then by the simpler rule. Reports:
`threshold_search.tsv` (P, R, F0.5, FP, FN, singleton correctness and average matches per S1
for every threshold), `decision_rule_search.tsv`, `ensemble_weight_search.tsv`,
`model_validation.json` (per-model and LR-baseline scores), `final_validation.json`,
`error_analysis*.{json,tsv}` and `feature_importance.json`.

## 20. Test inference

`--stage inference` refits LightGBM, XGBoost and RF on all labelled entities, using the
early-stopped iteration counts scaled to the larger set. When calibration was selected, it
keeps the selection models instead. It then builds test features for the exact candidate set,
applies the ensemble and the rule, and saves `artifacts/predictions/test_scored.parquet`.

## 21. Submission format

```
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812
S1-00003
```

`candidate_pairs.tsv` has the same layout with `candidate_entity_ids`. Both files are
tab-separated, have exactly one row per test Source 1 entity sorted by ID, contain sorted and
de-duplicated S2/S3 test IDs, allow empty lists, and are deterministic.

## 22. Submission validation

`--stage submission` writes both files and runs an internal validator. It fails on missing or
duplicate S1 rows, wrong headers or column counts, IDs that don't exist in test S2/S3, duplicate
IDs inside a row, and any match missing from `candidate_pairs.tsv`. If the challenge validator
exists at `utils/validate_submission.py` (or `BER_VALIDATOR`), the pipeline also runs:

```bash
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

If the test directory is not already a standalone folder, the test tables are exported to
`artifacts/dataset_export/test/` for the validator.

## 23. Hardware requirements

* Minimum: 4 cores, 16 GB RAM, ~10 GB of disk for artifacts.
* Recommended for the full multi-million-row data: 16+ cores, 32–64 GB RAM.

Memory is kept low by column-pruned Parquet, u64-hashed keys, block-size caps, S1 chunking
(`BER_S1_CHUNK`), feature chunking to disk (`BER_FEATURE_CHUNK`) and an RF row cap. To use
less RAM, reduce `BER_S1_CHUNK`, `BER_FEATURE_CHUNK` or `BER_MAX_TRAIN_S1`.

## 24. Runtime considerations

Measured on a 2-core cloud VM with a synthetic dataset shaped like the challenge (40 k S1,
350 k S2 + S3; 2 M train and 1 M test candidate pairs):

* blocking ≈ 1.5 min for train and 40 s for test,
* features ≈ 20 k pairs/s,
* the full run finishes in about 13 min with a peak RSS of 3.5 GB.

Time scales roughly linearly with the number of candidate pairs (≤ 50 per S1) and inversely
with the number of cores. RapidFuzz `cpdist`, Polars and LightGBM/XGBoost are all
multi-threaded. The most effective levers are `BER_MAX_TRAIN_S1` (for example 200 000),
`BER_MAX_CANDS_PER_SOURCE` and more cores.

**Checkpointing:** every stage writes its outputs to `artifacts/` (raw and normalised tables,
TF-IDF indexes, candidate pairs, features, models, validation predictions, ensemble weights,
the decision rule and test predictions) and skips work that already exists. A failed run
resumes where it stopped. `--force` recomputes the selected stage(s). After changing an
upstream setting, re-run downstream stages with `--force`, or delete `artifacts/`.

## 25. Reproducibility

A single seed (`BER_SEED`) drives Python, NumPy, the split, negative sampling and every model
(LightGBM runs with `deterministic=True`). Every rank uses an explicit tie-break, so identical
inputs give byte-identical output files; this was verified by running the pipeline twice and
comparing MD5 checksums. The configuration used is saved to `artifacts/config_used.json`, logs
go to `artifacts/logs/pipeline.log`, and dependency versions are pinned in `requirements.txt`.

## 26. Known limitations

* The metric is assumed to be micro P/R/F0.5 over matched IDs. If the official scorer
  aggregates differently (for example macro per entity), the threshold search should use that
  definition instead (`evaluation.GroupEvaluator`).
* Transliteration between scripts (Devanagari ↔ Latin) is only handled through char-n-gram and
  fuzzy similarity. There is no transliteration model, since none is allowed from external data.
* Chains (same brand, many branches) depend on address evidence. A very noisy or missing
  address can put the true branch below the per-source cap.
* Abbreviation dictionaries are deliberately small and conservative. Rare local abbreviations
  are left to the similarity features.
* TF-IDF vocabularies are fitted per split on unlabelled text (train on train, test on test),
  so unseen test vocabulary such as French words gets meaningful weights. This is transductive
  but uses no labels.
