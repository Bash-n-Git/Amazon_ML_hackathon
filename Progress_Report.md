# Progress Report — Business Entity Resolution

> Status as of 2026-09-25, 23:30. Plain-language summary of everything built and measured so far.

---

## 1. Planning and research (done)

| What | Where |
|---|---|
| Research doc: how to reach maximum accuracy, with sources | [Claude doc](https://claude.ai/code/artifact/89a7deb9-4365-4c77-9d06-b79c40a30b46) |
| Plain-language version of that doc | `Max_Accuracy_Plan_Simple.md` |
| Whole pipeline as flowcharts | `Pipeline_Flowchart.md` |

Main ideas we adopted:
- Only compare records from the **same country** (true for 100% of real matches).
- **Search in both directions** (business → records, and record → businesses), because 31% of
  businesses share their name with another branch.
- Decide each business's answer **as a whole list** (expected F0.5), not with one fixed score cut-off.
- Convert Indian-script names to English letters using a table **learned from the training data**
  (no Google or other external services; those are banned).

---

## 2. Environment (fixed)

| Problem | Fix |
|---|---|
| `gpu_env` had the **CPU-only** PyTorch, so the GPU was never used (`Test.py` failed) | Installed `torch 2.14.0+cu126` into `gpu_env`. `Test.py` now passes on the RTX 4090 |
| Search was far too slow on CPU (address search ~44 ms per business ≈ 20 h for test) | Moved the exact search to the GPU in half precision: **~30× faster** |
| GPU "out of memory" errors although memory was free | A PyTorch memory setting misbehaves on Windows. Removed it and used smaller blocks |
| RAM running out | Load only needed columns, process one country at a time, integer IDs instead of text |

---

## 3. Code built

All code is in `code/business_entity_resolution/src/`.

| File | What it does |
|---|---|
| `config.py` | Folder paths |
| `io_utils.py` | Reads the TSV files; writes submission files in the exact required format |
| `metric.py` | **Exact F0.5 scorer** (same rules as the challenge). Self-test reproduces the official example (0.714) |
| `normalize.py` | Cleans text: lowercase, removes junk (`<<`, `--`), fake accents (`cénter` → `center`), unifies `Rd/Road`, `R./Rue` and similar |
| `translit.py` | Learns the Indian-script → English-letters table from training pairs |
| `blocking.py` | Finds look-alike candidates (the shortlist) with 5 search paths, on the GPU |
| `blocking_lab.py` | Measures how many real copies the shortlist catches, and explains the misses |
| `build_candidates.py` | Runs the shortlist search for train and test, saving each country to disk |
| `features.py` | Turns each (business, candidate) pair into ~45 numbers (name, address, house-number similarity, noise flags, competition) |
| `decide.py` | Settles conflicts (one record → one business) and picks the best list per business |
| `pipeline.py` | Runs everything: quick model → features → main model → decision → submission |
| `baseline_exact.py` | First simple baseline |
| `README.md`, `requirements.txt` | How to reproduce; pinned versions (needed for the final zip) |

---

## 4. Results measured so far

### Baseline submission
- Simple exact-name matching: **macro F0.5 = 0.370** on the full training set (all-empty would score 0.056).
- `output/matching_results.tsv` + `candidate_pairs.tsv` **pass the official validator** (with ID checks).
  It's ready to upload as our first leaderboard score.

### Data facts discovered
| Fact | Why it matters |
|---|---|
| Real matches always share the country (100% of 7.6M) | Safe to search within country only |
| 31% of businesses share a name with another business | The name alone can't decide; address and reverse search needed |
| Test has 5.8 messy records per business vs 4.7 in train | Probably more fake records in test, so we must be stricter |
| Indian-script vocabulary is tiny: ~1,500 distinct words | The learned table covers **93–95%** of them; no external tool needed |
| Records with an unknown Indian-script word are real copies only **27%** of the time (vs 80%) | A strong "fake record" signal, now a feature |
| 24% of Indian names in Source 2 use an Indian script (9 scripts, Hindi most common) | Transliteration is essential |
| The generator adds fake accents (`cénter`, `Àmicale`) and filler words ("services", "center") | Accents now stripped; filler handled by the model |

### Shortlist (blocking) quality, measured on 5,000 businesses per country
| Measure | Result | Target |
|---|---|---|
| **Real copies caught by the shortlist** | **99.25%** (US 99.4%, India 99.1%) | ≥ 98.5% ✅ |
| Candidates per business | ~129 (cut to 25 by the quick model) | — |

Best single search: record → business with name+address combined (97.9% alone).
Two searches added nothing new (name-only forward, exact key), so they were dropped.
What's still missed (0.75%): mostly blank-address copies, and renamed businesses.

---

## 5. First full run — results (2026-09-25 evening)

### Scores (validation on training data, out-of-fold)

| Stage | Result |
|---|---|
| Shortlist catches true matches | **99.40%** |
| After cutting each shortlist to 25 | **99.01%** (the ceiling on recall) |
| Main model error (logloss, 3 folds) | 0.0187 |
| **Macro F0.5, best rule** | **0.9758** (baseline was 0.370) |
| Precision / recall | 0.991 / 0.950 |
| US / India | 0.9763 / 0.9749 |

Decision rules compared (F0.5 reweighted to the test country mix):

| Rule | F0.5 |
|---|---|
| **One record -> one business, then threshold 0.7** | **0.9756** |
| Threshold 0.7 | 0.9754 |
| Expected-F0.5 list selection | 0.9753 |
| Threshold 0.5 | 0.9731 |
| Threshold 0.3 | 0.9655 |

### Submission files (both pass the official validator, incl. the ID-existence check for v1)

| Folder | Rule | Matches | Empty lists |
|---|---|---|---|
| `output/v2_greedy0.7/` **<- upload first** | one record -> one business + 0.7 | 5,801,062 (3.35/business) | 97,392 (5.6%) |
| `output/v1_thr0.7/` | threshold 0.7 | 5,811,072 | 96,854 |

v2 drops 10,010 matches where one record was claimed by two businesses; at most one of each can be
right. Train validation understates this rule (only 30% of businesses are in it), so the leaderboard
is its real test: upload v2, and v1 if a second submission is cheap, and compare.

Expect the leaderboard score to be **lower than 0.976**: 15% of test is France (never seen in training)
and test has more fake records per business (5.8 vs 4.7).

### What went wrong and was fixed during the run
- Background jobs were killed 3 times for low memory (other apps hold ~14 GB). Every step is now
  chunked and resumable; peak Python memory ~17 GB.
- One feature step briefly needed ~21 GB (20 list calculations running in parallel) -> chunks cut to 500k pairs.
- XGBoost on the GPU was tested for the main model: slower than LightGBM on CPU (0.6 vs 0.22 s/round).
- Main model sped up: learning rate 0.05 -> 0.1, 5 -> 3 folds (~25 min per fold).

## 6. Next steps (iteration 2)

1. **Upload v2** and read the public leaderboard score; compare with our 0.976 validation.
2. **Recall is the gap** (0.950 vs 0.990 ceiling): study the missed matches by type (renamed business,
   blank address, transliteration).
3. Fix the rank tie-breaking in stage 1 (ties broken by row order; small, known issue).
4. France checks: legal suffixes (SARL, SAS, SCI), accents, street words — no training labels, so
   inspect predictions by hand.
5. Faster main model (depth-wise XGBoost or CatBoost on GPU) to make iterations cheaper.
6. Final zip + filled-in `Documentation_template.md`.

## 7. Resource use

| Resource | Measured |
|---|---|
| Disk (`work/`) | ~25 GB (~12 GB is a search cache we can delete later) |
| RAM | peak ~17 GB for Python; other apps hold ~14 GB |
| GPU | ~5 GB during candidate search; unused by the models |
| Full run time | ~4.5 h end to end (see code README) |
