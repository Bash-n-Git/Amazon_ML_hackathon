# Business Entity Resolution — final submission

Amazon ML Challenge 2026. **Final version: v15 — public leaderboard 0.986253** (macro F0.5).
Validation (out-of-fold, every competitor present): 0.9899 (v13_fa, US + India); US-only stacking 0.9902 on the US.

Only the provided challenge files are used — no external data, APIs or lookups. Pretrained model:
`intfloat/multilingual-e5-small` (MIT licence, 118M parameters).

## What changed from v9 (0.984953) to v15 (0.986253)

| Change | Why |
|---|---|
| Stacking (stages 3–4) trained on **all 2.2M** training businesses instead of 1.47M | +50% stacking data |
| Cross-encoder **B** (trained on 2M pairs of folds 1–2) scores fold 0; A scores folds 1–2; test = mean of A and B | every training row gets an out-of-fold cross-encoder score |
| Cross-encoder band widened **0.01–0.99 → 0.001–0.999** | ~2× more pairs get a transformer second opinion |
| New features: token-IDF weighted name/address overlap, rarest shared token, PIN-code conflict, model-vs-cross-encoder disagreement | rare-token evidence (aliases, generic names) |
| **Test predictions = average of the 3 validated inner-fold models** (no separately refit final model) | what is validated is exactly what predicts the test set |
| Three ~2M-pair small cross-encoders (B, c02, c01): **every fold scored out-of-fold by a strong model**; test = mean of 3 | two-thirds of training rows had been scored by the weaker 1M-pair model A (log-loss 0.197 → ~0.180) |
| Second opinion from a **larger cross-encoder** `intfloat/multilingual-e5-base` (MIT, 278M), 3 fold models, core band 0.01–0.99, as an extra feature | different errors from the small model |
| **France scored by stacking trained on US businesses only** (US + India unchanged) | France has no labels; US-only stacking is as good on the US (0.9902) and avoids India-specific patterns — LB +0.00005 |

## Pipeline

```
raw TSVs → normalise + Indic→Latin transliteration (learned from training pairs)       normalize.py, translit.py
→ candidate search per country (GPU, char-3gram TF-IDF, 4 retrievers) → 99.4% recall      blocking.py, build_candidates.py
→ stage 1 LightGBM → top 25 per business → 99.0% recall                                  pipeline.py stage1 / stage1rest
→ 72 pair features                                                                        features.py, features_v2.py
→ stage 2 LightGBM on 100% of training businesses, out-of-fold for all 3 folds           pipeline.py stage2full(_oof)
→ small cross-encoders B, c02, c01 (one per fold, ~2M pairs each), band 0.001-0.999      cross_encoder.py
→ large cross-encoder multilingual-e5-base (one per fold), band 0.01-0.99                cross_encoder.py
→ stage 3 + stage 4 LightGBM stacking (+ IDF features, both cross-encoders), 3-fold      pipeline.py stage4
→ test = mean of the 3 fold models per stage, scored per country                          pipeline.py predict4
→ France (a country with no training labels) re-scored by stacking trained on US businesses only
→ one-owner rule + p ≥ 0.7 → output/v15/{matching_results,candidate_pairs}.tsv
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

# 4b. (v12+) three ~2M-pair cross-encoders so every fold is scored out-of-fold by a strong model; test = mean of 3
python cross_encoder.py train_fold model_c02 0,2
python cross_encoder.py score_wide_fold model_c02 1 ce3_train_f1.parquet
python cross_encoder.py score_wide_test model_c02 ce3_test_c02.parquet
python cross_encoder.py train_fold model_c01 0,1
python cross_encoder.py score_wide_fold model_c01 2 ce3_train_f2.parquet
python cross_encoder.py score_wide_test model_c01 ce3_test_c01.parquet
python cross_encoder.py combine3

# 4c. (v13+) large cross-encoder multilingual-e5-base (MIT, 278M) as a second opinion on the core band
set BER_CE_MODEL=intfloat/multilingual-e5-base
set BER_CE_MAXLEN=64
python cross_encoder.py train_fold_cap cbase_f0 1,2 1200000 & python cross_encoder.py score_band_fold cbase_f0 0 cb_train_f0.parquet & python cross_encoder.py score_band_test cbase_f0 cb_test_m0.parquet
python cross_encoder.py train_fold_cap cbase_f1 0,2 1200000 & python cross_encoder.py score_band_fold cbase_f1 1 cb_train_f1.parquet & python cross_encoder.py score_band_test cbase_f1 cb_test_m1.parquet
python cross_encoder.py train_fold_cap cbase_f2 0,1 1200000 & python cross_encoder.py score_band_fold cbase_f2 2 cb_train_f2.parquet & python cross_encoder.py score_band_test cbase_f2 cb_test_m2.parquet
python cross_encoder.py combine_base
set BER_CE_MODEL=
set BER_CE_MAXLEN=

# 5. stacking (stages 3-4), fold-averaged; then test scoring per country
#    v11_fa:      BER_ALLFOLDS=1 BER_EXTRA=1 BER_CE_WIDE=1 BER_FOLDAVG=1 BER_SKIP_TEST=1   -> predict4 v11_fa
#    v12_fa:      BER_ALLFOLDS=1 BER_EXTRA=1 BER_CE3=1     BER_FOLDAVG=1 BER_SKIP_TEST=1   -> predict4 v12_fa
#    v13_fa:      BER_EXTRA=1 BER_CE3=1 BER_CEB=1 BER_FOLDAVG=1 BER_SKIP_TEST=1            -> predict4 v13_fa
#    v13_big2_fa: v13_fa settings + BER_S3BIG=1 BER_SEEDS=2                                 -> predict4 v13_big2_fa
#    v15:         v13_fa for US + India; France from US-only stacking (BER_ONLY=US), see 5b
#    Commands below: v15 (the final submission)
set BER_EXTRA=1
set BER_CE3=1
set BER_CEB=1
set BER_FOLDAVG=1
set BER_SKIP_TEST=1
python pipeline.py stage4                      # saves 3 stage-3 + 3 stage-4 fold models (s3cv_/s4cv_v13_fa_f*.txt)
python pipeline.py predict4 v13_fa 0.7         # writes output/v13_fa/

# 5b. v15: stacking trained on US businesses only, used for France (unseen country); US + India kept from v13_fa
set BER_ALLFOLDS=1
set BER_ONLY=US
python pipeline.py stage4                      # saves s3cv_/s4cv_v13all_us_fa_f*.txt
set BER_ALLFOLDS=
set BER_ONLY=
set BER_PRED_COUNTRIES=France
set BER_MERGE_BASE=v13_fa
python pipeline.py predict4 v13all_us_fa 0.7   # writes output/v13all_us_fa_france+v13_fa/
move "..\..\..\output\v13all_us_fa_france+v13_fa" "..\..\..\output\v15"

# 6. check
cd ../../../student_resource
python utils/validate_submission.py --matching ../output/v15/matching_results.tsv --candidate ../output/v15/candidate_pairs.tsv --test-dir dataset/test
```

Every experiment is behind an environment switch that is off by default; the commands above reproduce v15
exactly (steps 4b/4c build the cross-encoders; step 5 the v13_fa stacking; step 5b the US-only stacking for France).

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
| Cross-encoders: A + B, 2 more small (c02, c01), 3 large (e5-base), train + score | GPU | ~6 h |
| Stages 3–4, 3-fold | CPU | ~40 min |
| Test scoring per country | CPU | ~20 min |
| **Total** | | **~16–17 h** |

Peak RAM ~27 GB. Cross-encoder training on GPU is not bit-for-bit reproducible; LightGBM models use fixed seeds.

## Key numbers

| | |
|---|---|
| Candidates found by the search | 99.40% of true matches |
| Kept after stage 1 (25 per business) | 99.01% (candidate_pairs.tsv: 43,313,600 test pairs) |
| Validation macro F0.5 | 0.9899 (precision 0.999, recall 0.971) |
| Public leaderboard | **0.986253** (v15; v13_fa 0.986203) |
