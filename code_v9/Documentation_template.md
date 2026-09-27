# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary
A multi-stage pipeline: high-recall candidate generation with character-trigram TF-IDF retrievers searched exactly
on the GPU, a LightGBM pruning model, a LightGBM matcher on 72 engineered pair features trained on all 2.2M
training businesses, a multilingual transformer cross-encoder that re-reads the uncertain pairs, and two rounds of
LightGBM stacking that use competition between businesses for the same record. The key ideas were measured
properties of the data — a record belongs to at most one business, a blank address signals a true copy, and the
generator's systematic noise (typos in house numbers, "St" → "Saint", invented alias names, transliteration into
nine Indian scripts) — each turned into a feature or a decision rule. Validation macro F0.5 0.9893; public
leaderboard 0.984953.

---

## 2. Methodology

### 2.1 Problem Analysis
Measured on the full training data (2.2M Source-1, 10.3M Source-2/3 records):

- **Every Source-2/3 record matches at most one Source-1 business** (0 of 7.6M matched IDs reused) → a
  one-owner constraint for decisions and "competition" features.
- **Matches always share the country** → all search and matching runs within country.
- **Match-set sizes:** mean 3.46, 74% of businesses have 2–6 copies, only 5.6% are singletons → recall matters;
  an all-empty submission scores 0.056.
- **Blank addresses mark true copies:** 4.4% of matched records vs 0.3% of unmatched distractors.
- **26% of Source-2/3 records are distractors** (belong to nobody); test has more (5.8 vs 4.7 records per business).
- **Generator noise operators** (reverse-engineered from matched pairs): character typos; dropped/added legal
  suffixes and tokens; domain-style names (`maurewilliamscolombier.com`); invented alias names (`haloxylo`);
  full and partial transliteration into Indian scripts (24% of Indian names in Source 2); "St" written as
  "Saint"; house-number corruption (`7970→8970`, `244→00244`); state abbreviations (`TN`↔`Tamil Nadu`); fake accents.
- **31% of Source-1 names are shared by several branches** → the name alone cannot decide.
- **France** (15% of test) never appears in training; an unseen-country test (train on US, score India) costs
  3–5 F0.5 points, mostly precision.

### 2.2 Solution Strategy

**Approach Type:** Blocking + learned pruning + gradient-boosted matcher + transformer cross-encoder + stacking
(Hybrid)  
**Core Innovation:** Treating matching as a competition problem: a record can belong to only one business, so the
final stages score each pair in the context of every other business claiming the same record and every other
candidate of the same business, combined with a multilingual cross-encoder applied only where the matcher is
uncertain (≈6% of pairs), which keeps the transformer affordable on 43M test pairs.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** character-trigram TF-IDF vectors of the normalised, transliterated name and address,
  searched with exact cosine top-k (sparse × dense on the GPU, fp16), per country:

  | Retriever | Direction | k |
  |---|---|---|
  | name + address combined | business → records | 50 |
  | address only | business → records | 50 |
  | name, among blank-address records only | business → records | 30 |
  | name + address combined | record → businesses (reverse) | 3 |

  A **stage-1 LightGBM** on the retrievers' ranks and scores keeps the top 25 candidates per business.
- **Candidate pairs generated:** 43,313,600 test pairs in `candidate_pairs.tsv` (25 per Source-1 business) —
  exactly the set the matching models score. Reduction versus all within-country pairs: > 99.99%.
- **How you ensured true matches were not lost:** recall was measured on labelled training businesses before any
  modelling. The union of the four retrievers finds **99.40%** of true matches; after stage-1 pruning **99.01%**
  remain. The blank-address retriever and the reverse search were added specifically for the misses found in
  this analysis (blank-address copies; businesses whose name is shared by several branches). Retrievers that added
  no unique recall (name-only forward, exact key) were removed.

---

## 4. Matching Model

**Features used (72 pair features + stacking features):**
- **Name features:** fuzzy ratios (ratio, partial, token-set, token-sort), Jaro-Winkler, Levenshtein on the
  suffix-free key, token containment in both directions, Jaccard, exact-key match, Indic phonetic key
  (`sree`=`shree`, `jai`=`jay`), "invented name" score (share of the record's name tokens found in any business name).
- **Address features:** token-set / partial ratios on a noise-undone address (`saint→street`, `fourth→4th`,
  filler words removed), typo-tolerant comparison of the address words, fuzzy house numbers (leading zeros
  stripped, digit edit distance, containment, unmatched-number counts), containment and Jaccard.
- **Other:** blank-address flag, source, unmapped-Indic-token count, domain/junk/parenthesis noise flags, blocking
  ranks/scores, stage-1 probability, rank and score gaps within the business's candidates (tie-safe).
- **Cross-encoder:** `intfloat/multilingual-e5-small` (MIT, 118M parameters) fine-tuned as a pair classifier on
  the raw `name: … address: …` of both records in their original scripts, applied to pairs with 0.01 < p < 0.99.
- **Stacking (stages 3–4):** the business's context (rank, gap to best, sum, count of confident candidates,
  second-best score), record-side competition (number of businesses claiming the record, best competing score,
  gap), group consistency (similarity of the record to the business's other confident records, copies per source).

**Model type:** LightGBM throughout — stage 2 (learning rate 0.1, 127 leaves, 3,561 trees on 32.6M training
pairs with easy negatives down-sampled 10× and re-weighted), stages 3–4 (63 leaves); plus the transformer
cross-encoder. All scores feeding a later stage are out-of-fold.  
**Threshold selection method:** each record is assigned to its highest-scoring business (one-owner rule), then pairs
with probability ≥ 0.7 are accepted; threshold and rule chosen by macro F0.5 on out-of-fold validation over
1.47M businesses with every competitor present (the score is flat between 0.65 and 0.80, so the choice is not
fragile).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** 0.9893 validation (precision 0.998, recall 0.971); **0.984953** public leaderboard.
- **Common false positives (wrong merges):** distractor records that copy a real business's name at a nearby
  address, and records belonging to another branch with the same name. The one-owner rule and the record-side
  competition features removed most of them (precision 0.991 → 0.998 over the iterations).
- **Common false negatives (missed matches):** (1) ~1.0% of true matches never reach the matcher (not retrieved,
  or cut at 25 candidates); (2) blank-address copies of businesses whose name is shared by several branches —
  in 77% of these cases another business has an identical raw name, so name + address cannot decide; (3) renamed
  businesses (invented alias names) whose address was also corrupted.

---

## 6. Conclusion
The largest gains came from understanding the data generator rather than from bigger models: the one-owner
constraint, the blank-address signal and the noise operators each became a feature or rule, and a small
multilingual cross-encoder used only on uncertain pairs added +0.009 on the leaderboard. Lessons: validate the
exact model that produces the test predictions, keep training and test inputs from the same process, and trust a
validation that mirrors the test (every competitor present) over a convenient one.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/` — all source in `src/`, `README.md` (full run order and runtimes),
`requirements.txt` (pinned). Main entry points (from `src/`):

| File | Role |
|---|---|
| `normalize.py`, `translit.py` | normalisation; Indic→Latin table learned from training pairs |
| `blocking.py`, `build_candidates.py` | GPU TF-IDF retrievers; candidate lists per split and country |
| `features.py`, `features_v2.py` | pair features |
| `cross_encoder.py` | cross-encoder training and scoring |
| `pipeline.py` | `stage1`, `stage1rest`, `features`, `features2`, `stage2full`, `stage2full_oof`, `predict`, `stage3v8`, `stage4` |
| `decide.py`, `metric.py`, `io_utils.py` | decision rules, exact macro F0.5 scorer, TSV writers |

`python pipeline.py stage4` (with `BER_RUN=v6`) writes `output/v9/matching_results.tsv` and `output/v9/candidate_pairs.tsv` (copied to `output/` in the zip).

### B. Additional Results

| Version | Change | Validation F0.5 | Public LB |
|---|---|---|---|
| baseline | exact name key within country | 0.370 | — |
| v1 | first full pipeline, threshold 0.7 | 0.9757 | 0.967441 |
| v2 | + one-owner rule | 0.9758 | 0.968368 |
| v3 | + 20 noise-targeted features | 0.9798 | 0.97378 |
| v6 | + stage 2 on 100% of training businesses | 0.9810 | 0.976162 |
| v8 | + cross-encoder, record competition, group consistency | 0.9890 | ~0.9845 |
| **v9** | **+ second stacking round** | **0.9893** | **0.984953** |

| Pipeline stage | Share of true matches kept |
|---|---|
| Candidate search (4 retrievers) | 99.40% |
| After stage-1 pruning (25 per business) | 99.01% |
| Final predictions (recall) | 97.1% |
