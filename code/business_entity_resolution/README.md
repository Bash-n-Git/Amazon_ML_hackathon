# Business Entity Resolution — pipeline

Status: first full run complete (2026-09-25). Out-of-fold macro F0.5 on train: **0.9758**
(one-record-one-business + threshold 0.7; precision 0.991, recall 0.950).

## Layout

```
src/
  config.py            paths (override with BER_DATA / BER_WORK / BER_OUT env vars)
  io_utils.py          TSV loading; ground truth as pairs; writes submission TSVs
  metric.py            exact macro F0.5 (per S1 entity, singletons included); `python metric.py` self-tests
  normalize.py         normalisation (casefold, junk strip, Latin accent strip, suffix/address words);
                       prepared() = normalised + romanised table, cached as work/<split>_prep.parquet
  translit.py          Indic->Latin token table learned ONLY from train ground-truth pairs
  blocking.py          candidate generation: char-3gram TF-IDF retrievers, exact cosine top-k on GPU (fp16)
  build_candidates.py  runs blocking per split/country -> work/cands/
  blocking_lab.py      blocking recall + miss diagnosis on a train sample
  features.py          pair features (rapidfuzz similarities, number overlap, noise flags, competition)
  decide.py            decision layer: per-record column normalisation + per-S1 expected-F0.5 set selection
  pipeline.py          stage-1 prune -> features -> stage-2 LightGBM -> decision -> output files
  baseline_exact.py    Step-1 baseline (exact name key within country)
```

## Run end to end

```
pip install -r requirements.txt
pip install torch==2.14.0+cu126 --index-url https://download.pytorch.org/whl/cu126   # GPU search
cd src
python translit.py                      # learn Indic->Latin table from train pairs
python build_candidates.py train 0.3    # blocking: 30% S1 sample forward, all records reverse
python build_candidates.py test
python pipeline.py stage1
python pipeline.py features
python pipeline.py stage2               # prints decision-layer comparison on out-of-fold predictions
python pipeline.py predict greedy0.7    # writes ../../../output/matching_results.tsv + candidate_pairs.tsv
                                        # (no argument = best rule in work/pipe/stage2_eval.json)
```

Every step is resumable: finished pieces are cached under `work/` and skipped on a rerun.
`predict` caches test scores in `work/pipe/test_scores.parquet`, so changing the decision rule
afterwards takes seconds (delete that file after retraining stage 2).

## Runtime (measured, first full run)

Hardware: RTX 4090 Laptop GPU (16 GB), 24-core / 32-thread CPU, 32 GB RAM, Windows 11.

| Step | Runs on | Time |
|---|---|---|
| prepare + transliterate (train, test) | CPU | ~15 min |
| `build_candidates.py train 0.3` | GPU | ~35 min |
| `build_candidates.py test` | GPU | ~55 min |
| `pipeline.py stage1` | CPU | ~35 min |
| `pipeline.py features` | CPU | ~15 min |
| `pipeline.py stage2` (3 folds + final model) | CPU | ~1 h 45 min |
| `pipeline.py predict` | CPU | ~35 min |
| **Total** | | **~4.5 h** |

The candidate search falls back to CPU without CUDA but is then ~30x slower (~20 h for test).
Peak RAM ~17 GB for the Python process; steps are chunked so they fit a 32 GB machine.

`candidate_pairs.tsv` is the stage-1 top-25 list per S1: exactly the set the final model scores.

Data is expected at `<repo root>/student_resource/dataset/{train,test}`; intermediate files go to
`<repo root>/work`, outputs to `<repo root>/output`. Hardware used: RTX 4090 Laptop (16 GB), 32 GB RAM.

## Blocking retrievers (all within country)

| retriever | direction | k | purpose |
|---|---|---|---|
| comb_fwd  | S1 -> records | 50 | name + address in one vector; separates same-name branches |
| addr_fwd  | S1 -> records | 50 | renamed businesses / aliases |
| blank_fwd | S1 -> blank-address records | 30 | copies whose address was deleted |
| comb_rev  | record -> S1 | 3 | crowded names; ranks each record against all S1s |

Recall of true matches (30% train S1 sample): union of retrievers **0.9940**; after the stage-1
model keeps the top 25 per S1: **0.9901**. (`name_rev` and `name_fwd` were measured and dropped:
they added ~0.02% unique recall.)

## Models

- Stage 1: LightGBM on 13 blocking features (retriever ranks/scores), 3-fold out-of-fold; keeps top 25 per S1.
- Stage 2: LightGBM (learning rate 0.1, 127 leaves) on 52 pair features, 3-fold out-of-fold for
  evaluation, then refit on all train pairs. Validation logloss ~0.0187.
- Decision: each test record is kept only for its highest-scoring S1 (a record belongs to at most one
  S1 in the training truth), then pairs with probability >= 0.7 are accepted.

## Data use

Only the provided challenge files are used. No external data, APIs or lookups. The transliteration
table is learned from train ground-truth pairs.
