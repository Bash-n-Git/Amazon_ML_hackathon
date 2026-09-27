# Business Entity Resolution — v9 (final submission code)

Amazon ML Challenge 2026. **Public leaderboard: 0.984953** (macro F0.5). Validation (out-of-fold, 1.47M training
businesses, every competitor present): 0.9893.

This folder is the exact code that produced `output/v9/matching_results.tsv` and `output/v9/candidate_pairs.tsv`
(git commit `1682591`). Only the provided challenge files are used — no external data, APIs or lookups. The one
pretrained model is `intfloat/multilingual-e5-small` (MIT licence, 118M parameters).

## Pipeline

```
raw TSVs
 → normalise + Indic→Latin transliteration (table learned from training pairs only)      normalize.py, translit.py
 → candidate search, per country, GPU: char-3gram TF-IDF, 4 retrievers                    blocking.py, build_candidates.py
     (name+address fwd k50, address fwd k50, blank-address name fwd k30, name+address reverse k3)
     → 99.4% of true matches found
 → stage 1: LightGBM on blocking features, keep top 25 per business → 99.0% kept           pipeline.py stage1 / stage1rest
 → pair features: 52 base + 20 v2 (fuzzy house numbers, address-noise undo, Indic
     phonetic key, invented-name score, tie-safe ranks)                                     features.py, features_v2.py
 → stage 2: LightGBM on 100% of training businesses (3,561 trees), 3-fold out-of-fold       pipeline.py stage2full(_oof)
 → cross-encoder (multilingual-e5-small) on uncertain pairs 0.01 < p < 0.99,
     trained on fold-0 pairs, scores folds 1-2 and test                                     cross_encoder.py
 → stage 3: LightGBM stacking (business context, record competition, group consistency,
     cross-encoder score), trained on folds 1-2                                             pipeline.py stage3v8
 → stage 4: features recomputed from stage-3 scores, stacked again                          pipeline.py stage4
 → decision: each record goes to at most one business (its highest score), then p ≥ 0.7
 → output/v9/matching_results.tsv + candidate_pairs.tsv (the 25 candidates stage 2-4 scored)
```

## Run order (from `src/`, Windows, RTX 4090 Laptop 16 GB, 32 GB RAM)

```bash
pip install -r ../requirements.txt
pip install torch==2.14.0+cu126 --index-url https://download.pytorch.org/whl/cu126

# data preparation + candidates
python translit.py                                   # Indic→Latin table from training pairs
python build_candidates.py train 0.3                 # 30% training sample (forward), all records (reverse)
python build_candidates.py train rest 0.3            # the other 70% of training businesses
python build_candidates.py test

# stage 1 + features
python pipeline.py stage1                            # trains stage 1, prunes train sample + test to 25/business
python pipeline.py stage1rest                        # prunes the other 70% with the saved stage-1 model
python pipeline.py features train test trainrest
python pipeline.py features2 train test trainrest

# stage 2 on 100% of training businesses (run name v6)
set BER_RUN=v6
python pipeline.py stage2full                        # fold-0 validation + final model + test scores
python pipeline.py stage2full_oof 1,2                # out-of-fold scores for folds 1-2
python pipeline.py predict greedy0.75                # caches stage-2 test scores

# cross-encoder (model A)
python cross_encoder.py train                        # fold-0 uncertain pairs
python cross_encoder.py score_test
python cross_encoder.py score_train                  # folds 1-2

# stage 3 (v8) and stage 4 (v9)
python pipeline.py stage3v8
set BER_OUT_NAME=v8
python pipeline.py predict s3greedy0.7               # stage-3 test scores (cached as test_scores3.parquet)
python pipeline.py stage4                            # writes output/v9/

# check the submission
cd ../../../student_resource
python utils/validate_submission.py --matching ../output/v9/matching_results.tsv --candidate ../output/v9/candidate_pairs.tsv --test-dir dataset/test
```

Data is read from `<repo root>/student_resource/dataset/{train,test}`; intermediate files go to `<repo root>/work`,
outputs to `<repo root>/output`. Every step caches its results and skips finished pieces on a rerun.

## Runtime (measured)

| Step | Device | Time |
|---|---|---|
| Normalise + transliterate | CPU | ~15 min |
| Candidate search (train 30% + 70%, test) | GPU | ~2 h 45 min |
| Stage 1 (+ rest) | CPU | ~1 h |
| Features + features v2 | CPU | ~45 min |
| Stage 2 on 100% of training businesses (3 folds + final) | CPU | ~5 h |
| Cross-encoder train + score | GPU | ~1 h 15 min |
| Stage 3 + stage 4 | CPU | ~1 h 30 min |
| **Total** | | **~12–13 h** |

Peak RAM ~25 GB for the Python process. The cross-encoder training on GPU is not bit-for-bit reproducible;
all LightGBM models use fixed seeds.

## Key numbers

| | |
|---|---|
| Candidates found by the search | 99.40% of true matches |
| Kept after stage 1 (25 per business) | 99.01% |
| Validation macro F0.5 (stage 4) | 0.9893 (precision 0.998, recall 0.971) |
| Public leaderboard | 0.984953 |
