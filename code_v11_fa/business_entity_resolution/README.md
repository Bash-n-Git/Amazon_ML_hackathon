# Business Entity Resolution — v11_fa

Amazon ML Challenge 2026. **Public leaderboard: 0.985835** (macro F0.5). Validation (out-of-fold, all 2.2M training
businesses, every competitor present): 0.9898.

Only the provided challenge files are used — no external data, APIs or lookups. Pretrained model:
`intfloat/multilingual-e5-small` (MIT licence, 118M parameters).

## What changed from v9 (0.984953)

| Change | Why |
|---|---|
| Stacking (stages 3–4) trained on **all 2.2M** training businesses instead of 1.47M | +50% stacking data |
| Cross-encoder **B** (trained on 2M pairs of folds 1–2) scores fold 0; A scores folds 1–2; test = mean of A and B | every training row gets an out-of-fold cross-encoder score |
| Cross-encoder band widened **0.01–0.99 → 0.001–0.999** | ~2× more pairs get a transformer second opinion |
| New features: token-IDF weighted name/address overlap, rarest shared token, PIN-code conflict, model-vs-cross-encoder disagreement | rare-token evidence (aliases, generic names) |
| **Test predictions = average of the 3 validated inner-fold models** (no separately refit final model) | what is validated is exactly what predicts the test set |

## Pipeline

```
raw TSVs → normalise + Indic→Latin transliteration (learned from training pairs)       normalize.py, translit.py
→ candidate search per country (GPU, char-3gram TF-IDF, 4 retrievers) → 99.4% recall      blocking.py, build_candidates.py
→ stage 1 LightGBM → top 25 per business → 99.0% recall                                  pipeline.py stage1 / stage1rest
→ 72 pair features                                                                        features.py, features_v2.py
→ stage 2 LightGBM on 100% of training businesses, out-of-fold for all 3 folds           pipeline.py stage2full(_oof)
→ cross-encoders A (fold 0 pairs) and B (folds 1-2 pairs), band 0.001-0.999              cross_encoder.py
→ stage 3 + stage 4 LightGBM stacking (+ IDF features), 3-fold, all businesses           pipeline.py stage4
→ test = mean of the 3 fold models per stage, scored per country                          pipeline.py predict4
→ one-owner rule + p ≥ 0.7 → output/v11_fa/{matching_results,candidate_pairs}.tsv
```

## Run order (from `src/`)

```bash
pip install -r ../requirements.txt
pip install torch==2.14.0+cu126 --index-url https://download.pytorch.org/whl/cu126

# 1. data + candidates
python translit.py
python build_candidates.py train 0.3
python build_candidates.py train rest 0.3
python build_candidates.py test

# 2. stage 1 + pair features
python pipeline.py stage1
python pipeline.py stage1rest
python pipeline.py features train test trainrest
python pipeline.py features2 train test trainrest

# 3. stage 2 on 100% of training businesses (run name v6)
set BER_RUN=v6
python pipeline.py stage2full
python pipeline.py stage2full_oof 1,2
python pipeline.py predict greedy0.75          # caches stage-2 test scores

# 4. cross-encoders A and B
python cross_encoder.py train                  # A: fold-0 uncertain pairs
python cross_encoder.py score_test
python cross_encoder.py score_train            # A scores folds 1-2
python cross_encoder.py train_b                # B: folds 1-2 uncertain pairs
python cross_encoder.py score_fold0_b
python cross_encoder.py score_test_b
python cross_encoder.py combine                # fold 0 from B, folds 1-2 from A; test = mean(A, B)
python cross_encoder.py score_wide             # extra band 0.001-0.01 and 0.99-0.999, same out-of-fold rule
python cross_encoder.py combine_wide

# 5. stacking (stages 3-4) on all businesses, fold-averaged; then test scoring per country
set BER_ALLFOLDS=1
set BER_EXTRA=1
set BER_CE_WIDE=1
set BER_FOLDAVG=1
set BER_SKIP_TEST=1
python pipeline.py stage4                      # saves 3 stage-3 + 3 stage-4 fold models (s3cv_/s4cv_v11_fa_f*.txt)
python pipeline.py predict4 v11_fa 0.7         # writes output/v11_fa/

# 6. check
cd ../../../student_resource
python utils/validate_submission.py --matching ../output/v11_fa/matching_results.tsv --candidate ../output/v11_fa/candidate_pairs.tsv --test-dir dataset/test
```

`pipeline.py` also contains later experiments (v12/v13/v14 switches: `BER_CE3`, `BER_CEB`, `BER_S3BIG`); they are
off unless their environment variables are set, so the commands above reproduce v11_fa exactly.

Data is read from `<repo root>/student_resource/dataset/{train,test}`; intermediate files go to `<repo root>/work`,
outputs to `<repo root>/output`. Every step caches its results and skips finished pieces on a rerun.

## Runtime (measured, RTX 4090 Laptop 16 GB, 24-core CPU, 32 GB RAM)

| Step | Device | Time |
|---|---|---|
| Normalise + transliterate | CPU | ~15 min |
| Candidate search (train + test) | GPU | ~2 h 45 min |
| Stage 1 (+ rest) | CPU | ~1 h |
| Pair features | CPU | ~45 min |
| Stage 2 on 100% of training businesses | CPU | ~5 h |
| Cross-encoders A + B (train + score, incl. wide band) | GPU | ~2 h 30 min |
| Stages 3–4, 3-fold, all businesses | CPU | ~1 h |
| Test scoring per country | CPU | ~20 min |
| **Total** | | **~13–14 h** |

Peak RAM ~27 GB. Cross-encoder training on GPU is not bit-for-bit reproducible; LightGBM models use fixed seeds.

## Key numbers

| | |
|---|---|
| Candidates found by the search | 99.40% of true matches |
| Kept after stage 1 (25 per business) | 99.01% (candidate_pairs.tsv: 43,313,600 test pairs) |
| Validation macro F0.5 | 0.9898 (precision 0.999, recall 0.971) |
| Public leaderboard | **0.985835** |
