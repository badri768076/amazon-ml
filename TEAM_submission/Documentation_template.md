# Amazon ML Challenge 2026 — Business Entity Resolution: Documentation

**Team:** `<TEAM NAME>` · **Code:** `code/business_entity_resolution/` · **Reproduce:** `python src/main.py --stage all`

> **Before submitting:** every result cell marked **⟨fill⟩** must be copied from
> `code/business_entity_resolution/artifacts/reports/` after the pipeline has run on the
> official dataset. The file that holds each number is named in the table. Numbers under
> *"Smoke test (synthetic)"* come from a synthetic dataset built to mimic the challenge schema
> and noise. They show the pipeline works end to end. They are **not** challenge results.

---

## 1. Methodology

```
Kaggle dataset → load (required columns, cached Parquet) → validation → EDA (DuckDB)
 → conservative normalisation (Polars) → 10 country-scoped blocking strategies + fuzzy fallback
 → union → two-stage capping (blocking score → RapidFuzz quick score, ≤25 per source)
 → candidate recall → positives + hard negatives (S1-entity split 80/20)
 → ~80 similarity features → Logistic Regression baseline → LightGBM, XGBoost, Random Forest
 → weighted ensemble (weights by validation F0.5) → decision rule (threshold / relative / top-K)
 → zero / single / multi-match per S1 → test inference → submission validation → 2 TSV files
```

A matcher that is precise but only sees a small candidate pool is as useless as one that sees
everything but can't discriminate. The design therefore treats **candidate recall** (blocking)
and **precision at the entity level** (classifier + decision rule) as separate, separately
measured problems. The final decision is tuned directly on the competition metric, F0.5.

## 2. Dataset

* **Source:** Supplied challenge dataset located at `student_resource/dataset` (with automatic resolution for local laptop execution and fallback for Kaggle environments), loaded directly without external downloads or third-party APIs.
* **Training files:** `train_source1`, `train_source2`, `train_source3`, `train_ground_truth`.
* **Test files:** `test_source1`, `test_source2`, `test_source3`.
* **Columns:** `entity_id`, `business_name`, `business_address`, `country`. Ground truth:
  `source1_entity_id`, `matched_entity_ids`. ID prefixes are `S1-`, `S2-` and `S3-`.
* **Noise:** in names, abbreviations, legal-suffix variants (Pvt Ltd / Private Limited /
  Inc. / SARL), typos, punctuation, DBA/trade names, transliteration and word order. In
  addresses, missing components, abbreviations (Rd / Road, Av. / Avenue), PIN/ZIP formatting
  (`560 001`, ZIP+4), landmarks ("Near Bus Stand"), numbering differences and reordered
  components.
* **Countries:** an open set. Training has India and the US; test also has France. Country
  values are also missing or spelled several ways (`US`, `USA`, `United States`).

| Measured on the official data (`reports/eda.json`) | Value |
|---|---|
| Rows train S1 / S2 / S3 | ⟨fill⟩ |
| Rows test S1 / S2 / S3 | ⟨fill⟩ |
| Missing name / address / country (per table) | ⟨fill⟩ |
| Duplicate names / addresses / full records | ⟨fill⟩ |
| Train S1 with 0 / 1 / >1 matches | ⟨fill⟩ |
| Matches to S2 / S3 | ⟨fill⟩ |
| Countries only in test | ⟨fill⟩ |

## 3. Preprocessing

All rules are vectorised Polars expressions. The scalar functions `normalize_name`,
`normalize_address` and `normalize_country` run the same expressions, so there is no drift
between single records and batches.

* **Name:** NFKC → lowercase → strip Latin diacritics only (non-Latin scripts preserved) →
  collapse dotted acronyms (`S.A.S.` → `sas`) → drop the `M/s` honorific → punctuation to space,
  `&` → `and` → token map of legal forms and unambiguous abbreviations (`pvt` → `private`,
  `ltd` → `limited`, `corp` → `corporation`, `mfg` → `manufacturing`, …).
  **Core name** = the name with legal and filler tokens removed.
* **Address:** the same cleaning, plus PIN repair (`560 001` → `560001`), ZIP+4 → ZIP,
  `#` → `number`, letter-digit splitting and street-type abbreviations (`rd`, `st`, `ave`,
  `blvd`, `nr`, `opp`, `ngr`, …). Numeric tokens, postal candidates (5–6 digits), primary postal
  code and building number are extracted.
* **Country:** alias map (`USA` / `United States` → `us`, `India` / `IND` → `in`, `France` →
  `fr`, …). Unknown values pass through (open set); missing stays null.
* **Why conservative:** every rule only unifies spellings of the same thing. Nothing merges
  distinct words, because an aggressive rule would create false merges, which F0.5 punishes most.

## 4. Blocking

No Cartesian product is ever built. Each strategy produces hashed keys. The candidate side is
indexed, keys shared by more records than a cap are dropped, and S1 chunks are hash-joined
against each index.

| Strategy | Key | Handles |
|---|---|---|
| Country scoping | country prefix on every scoped key; `?`/`*` variants for missing countries | cross-country false candidates, without excluding missing countries |
| exact_name | normalised name (unscoped) | clean matches, wrong or missing country |
| exact_core | name without legal forms (unscoped) | legal-suffix variants |
| name_signature | sorted unique core tokens | word order |
| name_token | 3 rarest core tokens of the S1 record | partial names, DBA variants |
| char_ngram | 4 rarest char-3-grams of the core name | typos, transliteration, spacing |
| addr_token | 3 rarest alphabetic address tokens | renamed businesses at the same place |
| postal_name | postal code + first 3 letters of the core name | chains, noisy names |
| postal_building | postal code + building number | same premises |
| token_postal | rare name token + postal code | common names in one locality |
| addr_signature | sorted alphabetic address tokens | reordered addresses |
| fuzzy | TF-IDF char-3-gram sparse cosine (country-partitioned) + RapidFuzz re-rank | S1 records with < 3 candidates from a source |

**Scoring and capping:** the pair score is Σ wₛ·log(1 + N/df). The top 200 per (S1, source) by
that score are re-ranked by a cheap RapidFuzz score (name token-set 0.35, name ratio 0.15,
address token-set 0.20, address ratio 0.10, postal match 0.10, building match 0.05, blocking
score 0.05). The top 25 per source are kept. That capped set is exactly what the model scores
and what is written to `candidate_pairs.tsv`.

## 5. Candidate recall

| Metric (`reports/candidate_recall_train.json`) | Official data | Smoke test (synthetic, 40 k S1 × 350 k S2+S3) |
|---|---|---|
| Candidate recall (all / S2 / S3) | ⟨fill⟩ | 0.911 / 0.908 / 0.914 |
| Pre-cap recall | ⟨fill⟩ | 0.977 |
| Avg / median / max candidates per S1 | ⟨fill⟩ | 50 / 50 / 50 |
| Candidate reduction ratio | ⟨fill⟩ | 0.99986 (2.0 M pairs vs 14 B Cartesian) |

The synthetic set is deliberately harsh on chains. With a small name vocabulary, the median S1
has ~90 candidates with an identical core name per source, so the cap from 200 to 25 costs
recall. On the 10× smaller synthetic set, recall was 0.985. With 25 per source, the quick score
beat a pure blocking-score cap by 5 recall points (0.857 → 0.911). `BER_MAX_CANDS_PER_SOURCE`
trades recall for runtime.

## 6. Feature engineering (≈ 80 features)

* **Name:** Levenshtein, Indel ratio, Jaro, Jaro-Winkler, token-sort / set / partial on the
  full and core names; prefix similarity; char-(2–4)-gram and word TF-IDF cosine; token
  Jaccard / overlap / common count / common ratio (full and core); length difference and ratio;
  word-count difference; exact and exact-core match; acronym match.
* **Address:** Levenshtein, Jaro, Jaro-Winkler, token-sort / set / partial, char-3-gram TF-IDF
  cosine, token Jaccard / overlap / common, numeric-token overlap and Jaccard, postal exact,
  prefix, both-present and conflict, building-number match, length difference, exact match.
  All are NaN when an address is missing.
* **Country:** exact match (NaN when either is missing) and a missing flag.
* **Cross-field:** combined name + address token-set; name/address mean; source indicator;
  missing name / address flags.
* **Blocking:** score, quick score, rank, number of strategies, one indicator per strategy.
* **Group context** (label-free, per S1): number of candidates, similarity rank and gap to the
  best candidate, count of candidates with the same name, and count of exact-core matches.
  These help tell real duplicates apart from look-alikes.

Raw entity IDs are never used as features.

## 7. Models

| Model | Setup |
|---|---|
| Logistic Regression (baseline) | StandardScaler, NaN → −1, C = 1 |
| LightGBM | 63 leaves, lr 0.05, bagging / feature fraction 0.8, early stopping on the inner hold-out, deterministic |
| XGBoost | hist, depth 8, eta 0.05, subsample / colsample 0.8, early stopping on the inner hold-out |
| Random Forest | 200 trees, depth 16, min leaf 5, sqrt features, 50 % bootstrap, ≤ 1.5 M rows |

| Validation (`reports/model_validation.json`), each at its own best threshold | Official P / R / F0.5 | Smoke test P / R / F0.5 |
|---|---|---|
| Logistic Regression | ⟨fill⟩ | 0.9952 / 0.8732 / 0.9681 |
| LightGBM | ⟨fill⟩ | 0.9980 / 0.8854 / 0.9733 |
| XGBoost | ⟨fill⟩ | 0.9977 / 0.8848 / 0.9729 |
| Random Forest | ⟨fill⟩ | 0.9979 / 0.8696 / 0.9693 |

## 8. Ensemble

`P_final = w1·P_LGBM + w2·P_XGB + w3·P_RF`. Equal weights come first as the reference. Then all
66 weight vectors on a 0.1-step simplex are scored, each at its own best threshold, and the best
validation F0.5 is chosen. Platt and isotonic calibration (fitted on the inner hold-out) are
tried and kept only for a gain above 0.001 F0.5.

| `reports/ensemble_selection.json` | Official | Smoke test |
|---|---|---|
| Equal-weights F0.5 | ⟨fill⟩ | 0.9717 |
| Selected weights (LGBM / XGB / RF) | ⟨fill⟩ | 0.8 / 0.2 / 0.0 |
| Selected calibration | ⟨fill⟩ | none (Platt 0.9732, isotonic 0.9730 vs 0.9736) |

## 9. Training

* **Positives:** ground-truth pairs (S1↔S2, S1↔S3) that blocking retrieved.
* **Hard negatives:** non-matching pairs retrieved by blocking. They share a country, name
  tokens, n-grams, address tokens, postal code or building number with the S1 record, so they
  are exactly the confusions the model has to resolve. The 20 hardest per S1 (by quick score)
  are kept, plus 5 random in-block negatives. There are no random cross-product negatives.
* **Final model:** after selection, LightGBM, XGBoost and RF are refit on all labelled entities
  with the early-stopped iteration counts, scaled for the larger set.

## 10. Validation

Splitting happens at the **Source 1 entity level**, before any pair is built: 80 % train and
20 % validation, with 10 % of the training entities as an inner hold-out for early stopping and
calibration. An assertion checks that no S1 entity appears in both train and validation, and
validation labels are never used as features. Validation keeps the full candidate set, and
counts ground-truth matches that blocking missed as false negatives.

## 11. Evaluation

Precision, recall, F0.5 = 1.25·P·R / (0.25·P + R), F1, TP, FP and FN are computed over matched
IDs, micro-averaged over S1 entities. **Singleton correctness** is the share of S1 entities with
no true match that receive an empty list.

| `reports/final_validation.json` | Official | Smoke test |
|---|---|---|
| Precision | ⟨fill⟩ | 0.9984 |
| Recall | ⟨fill⟩ | 0.8855 |
| **F0.5** | ⟨fill⟩ | **0.9736** |
| F1 | ⟨fill⟩ | 0.9386 |
| TP / FP / FN | ⟨fill⟩ | 8165 / 13 / 1056 |
| Singleton correctness | ⟨fill⟩ | 0.9990 |

Smoke-test held-out *test* split, including France, which is absent from training: overall
F0.5 0.978 (P 0.998, R 0.906); India 0.976, US 0.982, **France 0.977**.

## 12. Threshold optimisation

The rule keeps candidate *j* if `p_j ≥ t`, `p_j ≥ r·max(p)` and `rank_j ≤ K`. The search covers
t ∈ {0.10 … 0.95, 0.97, 0.99}, r ∈ {0, 0.5, 0.7, 0.85} and K ∈ {∞, 3, 10}. Every candidate that
passes is kept (multi-match); if none passes the list is empty (singleton). For every threshold,
`reports/threshold_search.tsv` lists P, R, F0.5, FP, FN, singleton correctness and average
matches per S1.

| | Official | Smoke test |
|---|---|---|
| Selected t / r / K | ⟨fill⟩ | 0.90 / 0 / ∞ |
| F0.5 at t = 0.50 (for reference) | ⟨fill⟩ | 0.9682 (FP 115) |
| F0.5 at selected rule | ⟨fill⟩ | 0.9736 (FP 13) |

## 13. Scalability

* Parquet is read column-pruned with PyArrow and cached once. Stages scan it lazily with
  Polars; EDA runs DuckDB directly on the Parquet files.
* Blocking keys are hashed to u64. Frequency caps bound block sizes, S1 is processed in chunks
  of 5 000, and pairs come from hash joins only — never a Cartesian product.
* The fuzzy fallback uses a sparse TF-IDF matrix with bounded posting lists and batched sparse
  matrix products, and only runs for under-served S1 records.
* Features are computed with multi-threaded RapidFuzz `cpdist`, Polars list/set operations and
  sparse row-wise cosines (each distinct string transformed once). Chunks of 400 k pairs are
  streamed to disk.
* Training sets are built part by part, so only the sampled training pairs and the validation
  pairs are held in memory.
* Every expensive artifact is checkpointed, so runs are resumable.
* Measured on the synthetic 10× set (2-core VM): 13 min end to end, peak RSS 3.5 GB, with
  2 M train and 1 M test candidate pairs.

## 14. Error analysis

`reports/error_analysis.json` groups errors by cause, and `error_analysis_false_{positives,negatives}.tsv`
lists the worst cases with their raw records.

| | Official | Smoke test |
|---|---|---|
| FN missed by blocking / below threshold | ⟨fill⟩ | 824 / 232 |
| FP with name similarity ≥ 0.9 | ⟨fill⟩ | 9 of 13 |
| FP sharing the postal code | ⟨fill⟩ | 11 of 13 |

In the smoke test, most false negatives come from blocking, where chain businesses with
identical names exceed the cap. The remaining false positives are almost all *same name, same
postal code, different branch or building*. That is the expected hard case and the reason the
rule prefers a high threshold.

## 15. Limitations

* The official aggregation is assumed to be micro over matched IDs. If it differs, the evaluator
  (`evaluation.GroupEvaluator`) is the single place to change.
* Cross-script transliteration is only covered through n-gram and fuzzy similarity, because no
  external transliteration resource is allowed.
* Chains with noisy or missing addresses can lose the true branch at the per-source cap.
* Abbreviation dictionaries are intentionally small.
* TF-IDF is fitted per split on unlabelled text (transductive, but label-free).

## 16. Future improvements

* A learned blocking re-ranker (a tiny GBM on the quick features) instead of the fixed
  quick-score weights, and an adaptive per-S1 cap based on score gaps.
* Char-level multilingual embeddings trained only on challenge text, used as an extra retrieval
  channel and feature.
* Joint decisions per S1: consistency between S2 and S3 matches, and transitivity through
  S2↔S3 similarity.
* Separate thresholds per source and per country, if validation supports them.
* Automatic abbreviation mining from aligned ground-truth pairs.
