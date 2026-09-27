# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary
A multi-stage pipeline: high-recall candidate generation with character-trigram TF-IDF retrievers searched exactly
on the GPU, a LightGBM pruning model, a LightGBM matcher on 72 engineered pair features trained on all 2.2M
training businesses, multilingual transformer cross-encoders (three small models and one larger one) that re-read the uncertain pairs,
and two rounds of LightGBM stacking that use competition between businesses for the same record;
test predictions are the average of the validated fold models. The key ideas were measured
properties of the data — a record belongs to at most one business, a blank address signals a true copy, and the
generator's systematic noise (typos in house numbers, "St" → "Saint", invented alias names, transliteration into
nine Indian scripts) — each turned into a feature or a decision rule. Validation macro F0.5 0.9899; public
leaderboard 0.986253 (final version v15).

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
uncertain (≈13% of pairs, band 0.001 < p < 0.999), which keeps the transformer affordable on 43M test pairs. Every
score passed to a later stage is out-of-fold, and the test set is scored by the average of the same fold models that
were validated.

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
  the raw `name: … address: …` of both records in their original scripts, applied to pairs with 0.001 < p < 0.999.
  Three models, one per fold (B, c02, c01), each trained on ~2M uncertain pairs of the other two folds, so every
  training pair is scored out-of-fold by an equally strong model; test = mean of the three.
- **Second cross-encoder:** `intfloat/multilingual-e5-base` (MIT, 278M parameters), also one model per fold
  (1.2M training pairs each, max length 64), applied to pairs with 0.01 < p < 0.99; its score is an extra
  stacking feature (a larger model makes different errors).
- **Rare-token evidence:** token-IDF weighted name and address overlap, the rarest shared token, number of rare
  shared tokens, PIN-code conflict, and the disagreement between the matcher and the cross-encoder.
- **Stacking (stages 3–4):** the business's context (rank, gap to best, sum, count of confident candidates,
  second-best score), record-side competition (number of businesses claiming the record, best competing score,
  gap), group consistency (similarity of the record to the business's other confident records, copies per source).

**Model type:** LightGBM throughout — stage 2 (learning rate 0.1, 127 leaves, 3,561 trees on 32.6M training
pairs with easy negatives down-sampled 10× and re-weighted), stages 3–4 (63 leaves, 3-fold on the 1.47M training
businesses of folds 1–2); plus the transformer cross-encoders. All scores feeding a later stage are out-of-fold, and the test set is
scored by the **average of the 3 fold models** of each stage — the exact models that were validated (no separately
refit final model).  
**Unseen country (France):** France appears only in the test set. Its pairs are scored by stages 3–4 trained on US
businesses only (validated: 0.9902 on the US, equal to the US + India model), so no India-specific pattern is
transferred; US and India use the US + India models.  
**Threshold selection method:** each record is assigned to its highest-scoring business (one-owner rule), then pairs
with probability ≥ 0.7 are accepted; threshold and rule chosen by macro F0.5 on out-of-fold validation over
1.47M training businesses with every competitor present (the score is flat between 0.65 and 0.80, so the choice is not
fragile).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** 0.9899 validation (precision 0.999, recall 0.971); **0.986253** public leaderboard (v15).
- **Common false positives (wrong merges):** distractor records that copy a real business's name at a nearby
  address, and records belonging to another branch with the same name. The one-owner rule and the record-side
  competition features removed most of them (precision 0.991 → 0.998 over the iterations).
- **Common false negatives (missed matches):** (1) ~1.0% of true matches never reach the matcher (not retrieved,
  or cut at 25 candidates); (2) blank-address copies of businesses whose name is shared by several branches —
  in 77% of these cases another business has an identical raw name, so name + address cannot decide; (3) renamed
  businesses (invented alias names) whose address was also corrupted.

---

### Robustness checks
- **Language probe:** 80 labelled training businesses (with copies and hardest decoys) translated word-by-word into
  German, Italian, Spanish, UK English and South-Asian variants and run through the full pipeline: consistently
  translated names/addresses score within 0.005–0.03 of the English/Indian originals; the weak spots are copies that
  mix languages ("Groß Select" vs "GREAT SELECT") and legal forms missing from the suffix list (S.R.L., GmbH).
- **French legal forms:** dotted forms (S.A.R.L., E.U.R.L.) are matched at least as often as plain ones (57.5% vs 54.8%
  of records), so punctuation in suffixes is not a source of French errors.

---

## 6. Conclusion
The largest gains came from understanding the data generator rather than from bigger models: the one-owner
constraint, the blank-address signal and the noise operators each became a feature or rule, and a small
multilingual cross-encoder used only on uncertain pairs added +0.009 on the leaderboard (+0.001 more from three equally strong small models and a larger second model). Lessons: validate the
exact model that produces the test predictions (a memory-saving refit that was never validated cost 0.0035 on the
leaderboard until test predictions were switched to the average of the validated fold models), keep training and
test inputs from the same process, and trust a validation that mirrors the test (every competitor present).

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
| `pipeline.py` | `stage1`, `stage1rest`, `features`, `features2`, `stage2full`, `stage2full_oof`, `predict`, `stage3v8`, `stage4`, `predict4` |
| `decide.py`, `metric.py`, `io_utils.py` | decision rules, exact macro F0.5 scorer, TSV writers |

`python pipeline.py stage4` then `python pipeline.py predict4 v13_fa 0.7` (with `BER_RUN=v6 BER_EXTRA=1 BER_CE3=1
BER_CEB=1 BER_FOLDAVG=1 BER_SKIP_TEST=1`) give v13_fa; a second `stage4` with `BER_ALLFOLDS=1 BER_ONLY=US` and
`predict4 v13all_us_fa 0.7` with `BER_PRED_COUNTRIES=France BER_MERGE_BASE=v13_fa` re-score France and write v15
(`matching_results.tsv` and `candidate_pairs.tsv`)
(copied to `output/` in the zip). Full run order in the code README.

### B. Additional Results

| Version | Change | Validation F0.5 | Public LB |
|---|---|---|---|
| baseline | exact name key within country | 0.370 | — |
| v1 | first full pipeline, threshold 0.7 | 0.9757 | 0.967441 |
| v2 | + one-owner rule | 0.9758 | 0.968368 |
| v3 | + 20 noise-targeted features | 0.9798 | 0.97378 |
| v6 | + stage 2 on 100% of training businesses | 0.9810 | 0.976162 |
| v8 | + cross-encoder, record competition, group consistency | 0.9890 | ~0.9845 |
| v9 | + second stacking round | 0.9893 | 0.984953 |
| v9x_fa | + wider cross-encoder band, rare-token features, fold-averaged test predictions | 0.9895 | 0.98536 |
| v11_fa | + stacking on all 2.2M businesses, cross-encoder B | 0.9898 | 0.985835 |
| v12_fa | + three equally strong small cross-encoders (one per fold) | 0.9900 | — |
| v13_fa | + larger cross-encoder (e5-base) as a second opinion | 0.9899 | 0.986203 |
| v13_big2_fa | + bigger stacking models, 2 seeds | 0.9899 | — (no gain, not submitted) |
| v13all_fa | v13 stacking on all 2.2M businesses | 0.9901 | — (below +0.001, not submitted) |
| **v15** | **France scored by stacking trained on US businesses only** | **0.9902 (US)** | **0.986253** |

| Pipeline stage | Share of true matches kept |
|---|---|
| Candidate search (4 retrievers) | 99.40% |
| After stage-1 pruning (25 per business) | 99.01% |
| Final predictions (recall) | 97.1% (precision 99.9%) |
