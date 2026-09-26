# Plan 2 — Iteration 2: from 0.968 to 0.985+

> **Status:** written 2026-09-26 00:15 IST, after the first leaderboard score. Deadline Sep 27, 21:00 IST (~45 h).
> **Target:** public leaderboard **≥ 0.985** (top submissions are > 0.987).
> Every number below is measured unless marked ❓. Analysis output: `work/error_analysis_v1.txt`.
> Plan_1.md still holds the data facts and pitfalls; this file is what changes and why.

---

## 1. Where we stand

| | Score |
|---|---|
| Baseline (exact name match) | 0.370 |
| **v2 on our validation** (out-of-fold, train) | **0.9758** |
| **v2 on the public leaderboard** | **0.9684** |
| Gap | **−0.0074** |
| Top of leaderboard | > 0.987 |
| **Needed to reach 0.985** | **+0.017** |

---

## 2. Why our validation said 0.976 when the test said 0.968

Our validation did not look like the test. Three differences, all known before submitting, but I reported
the validation number without estimating how much they would cost. That was the mistake.

| Difference | Train validation | Test | Effect |
|---|---|---|---|
| **France** | 0% of businesses | **15%** | Model never saw French names/addresses. On test, France has **2× as many uncertain pairs** (1.7% of pairs with 0.2 ≤ p < 0.7 vs ~0.9% for US/India). |
| **Fake records** | 4.7 messy records per business | **5.8** | More distractors → more chances for wrong matches. US/India uncertain-zone share is ~20% higher on test than in validation. |
| **Who competes for a record** | only 30% of businesses were in the validation sample | all businesses | 26% of our validation "wrong matches" belong to a business outside the sample; on test that owner is present. The threshold (0.7) was tuned in the wrong environment. |

Implied split ❓ (no test labels): if US/India had scored on test exactly as in validation (0.976/0.975),
France would be ~0.93. More likely US/India dropped a little too (≈0.970) and France is ≈0.945.
Either way **France is worth roughly half of the gap**, and extra distractors the other half.

**Fix to the process:** build a validation that imitates the test (Section 6) and never quote a
validation number again without its expected test gap.

---

## 3. Where the points are lost (validation, v2, 662k businesses)

### 3.1 Missed matches cost 2.7× more than wrong matches

| If we fixed… | Validation F0.5 | Gain |
|---|---|---|
| nothing | 0.9758 | — |
| every **wrong** match (precision) | 0.9818 | +0.0060 |
| every **missed** match (recall) | 0.9920 | **+0.0162** |

→ **Iteration 2 is mainly a recall iteration**, while keeping precision ≥ 0.99.

### 3.2 At which stage are true matches lost? (2,291,019 true pairs)

| Stage | Lost pairs | Share of true pairs |
|---|---|---|
| Never found by the candidate search (blocking) | 13,717 | 0.60% |
| Cut by stage 1 (top 25) | 9,027 | 0.39% |
| **Found, scored, but p < 0.7** | **89,713** | **3.92%** |
| Taken by another business (one-record-one-business) | 1,968 | 0.09% |

The scored-but-missed are spread across all probabilities: 26.7k have p ≤ 0.1, 23.5k have p in (0.5, 0.7].
**The model is the bottleneck, not the search.**

### 3.3 What the missed matches look like

| | Correct matches | Missed (scored < 0.7) | Wrong matches |
|---|---|---|---|
| Record has blank address | 2.1% | **46.6%** | 26.1% |
| Name similarity ≥ 90 | 81% | 68% | 65% |
| Address similarity ≥ 90 | 79% | **28%** | 49% |
| Indian script in name | 7.4% | 2.9% | 3.2% |

Wrong matches: 64% are pure distractors (belong to nobody), 26% belong to a business outside the
validation sample, 10% belong to another sampled business.

---

## 4. The model's mistakes — root causes

### M1. Blank-address copies of businesses that share a name (~43k missed pairs, 1.9% of true pairs)
Blank-address misses: **only 23% have a unique business name**; 61% have a name shared by 3+ businesses
(e.g. "new delhi solutions" — 118 businesses). Correctly matched blank records: 89% unique.
A blank "Prime Money" with three "Prime Money" branches carries **no information** about which branch.
- ~33k of these are **genuinely ambiguous → largely irreducible** (≈ 0.003 of F0.5). Accept it.
- ~10k have a **unique name** and are still scored low (e.g. "plumbers local union 750" → "… inc", p = 0.47).
  **Fixable.**

### M2. Numbers are compared exactly — one wrong digit reads as "different address" (part of ~27k)
56% of non-blank misses have **both** name and address token-similarity ≥ 80, yet mean p = 0.36. Examples:
`7970 → 8970`, `2535 → 535`, `244 → 00244`, `3500 → 3879 3883`, `17 5 → h number 301 17 5`.
Our number features are `addr_num_jac` and `addr_num_first_eq`: **exact match only**. The generator
corrupts digits (typo, drop, leading zeros, extra numbers), so a true copy looks like a neighbour.

### M3. Generator noise we never undo in addresses
- **"St" → "Saint"**: ~112k S2/S3 US records say `Palmer Saint`, `Fourth Saint` (S1 has "Saint" only in
  real names like Saint Louis: 12k). We never map it back.
- Ordinal words: `4th` ↔ `fourth`. City typos: `seattle/settle`, `graford/graord`. Inserted words:
  `township`, `po box 1146`, `unit unit`.

### M4. Renamed businesses (aliases) score low (~12.6k missed, 26% of non-blank misses)
Name completely different, address matches: `twisted pizza → haloxylo`, `sfa optimal triton inc → synfluxsol`,
`prudent affiliate pvt ltd → tavozeph`. The alias names are **made-up words** that appear nowhere else.
The model has no feature saying "this name looks invented", so it cannot tell an alias from a
real different business at the same address.

### M5. Indian romanisation variants (part of name-high/address-low misses)
`sree ↔ shree`, `jai ↔ jay`. Same name, different spelling conventions. No phonetic key for these.

### M6. Branch competition is only seen through blocking ranks
When several businesses share a name (31% of S1 names), the model sees competition only through the
reverse-search rank (`blk_comb_rev_*`), not through **its own scores**. It does not know "this record's
best other claimant scored 0.95, I scored 0.60" — the strongest signal for the one-owner rule.

### M7. France: never trained, never checked
French legal forms (SARL, SAS, SCI, EURL), street words (`rue/r.`, `bd`, `av.`, `chemin`, `allée`),
accents, `bis/ter` house numbers, region names. The normaliser handles some (`R.`/`Rue`), not all, and
nothing has been inspected on French predictions.

### M8. Trained on only 30% of training businesses
Stage 2 saw 662k of 2.2M businesses. More data usually helps GBMs on hard, rare cases (aliases,
digit corruption). Held back by runtime (Section 7).

### M9. Rank tie-breaking (known from run 1)
`s1_rank_best` / `cmp_rank_in_s1` break exact ties by row order → identical S2/S3 copies get arbitrary
ranks; the model relies on rank 1 heavily. Switch to `rank("min")`.

---

## 5. Improvements, ranked by expected gain per hour

Expected gains are validation F0.5 estimates ❓ from the loss sizes above; every item is measured
before it is kept. Order = do-first.

| # | Change | Targets | Est. gain | Cost |
|---|---|---|---|---|
| **I1** | **Fuzzy number features**: strip leading zeros; digit edit distance; one number contained in another; best match over all number pairs; count of unmatched numbers | M2 | **+0.003–0.005** | 2 h |
| **I2** | **Second-pass (stacked) model with score-based competition**: per record, p-rank among all its claimants, gap to best other claimant; per business, p-rank and gap among its candidates; number of confident candidates | M6, M1 | **+0.002–0.004** | 3 h |
| **I3** | **Undo address noise**: `saint→street` (both sides), ordinals `4th↔fourth`, drop `po box …`, `township`, repeated tokens; city-typo tolerant city feature | M3 | +0.001–0.002 | 1.5 h |
| **I4** | **Alias features**: share of the record's name tokens that appear in any S1 name (invented-word score); character-bigram "language" score; `address strong & name unrelated` flag | M4 | +0.001–0.003 | 2 h |
| **I5** | **France hardening**: legal forms, street words, `bis/ter`, accents both folded and kept; then hand-inspect 200 French predictions at 0.3 < p < 0.9 | M7 | +0.002–0.004 on test (France ≈ 15%) | 3 h |
| **I6** | **Indic phonetic key**: collapse `sh/s`, `ee/i`, `aa/a`, `oo/u`, `v/w`, `j/z`, doubled letters; feature = phonetic-key equality/similarity | M5 | +0.0005–0.001 | 1 h |
| **I7** | **Train on more businesses** (60–100% instead of 30%), with **negative down-sampling** (keep all positives + hard negatives, drop easy negatives with stage-1 p ≈ 0) | M8 | +0.001–0.003 | runtime (see B1) |
| **I8** | Keep **top 40** instead of 25 after stage 1 (pruning costs 0.39% of true pairs) | Stage 2 | +0.0005–0.001 | runtime |
| **I9** | Tie-break fix (`rank("min")`) | M9 | small, removes noise | 0.5 h |
| **I10** | Threshold / rule chosen on the **test-like validation** (Section 6), per country incl. a France-specific threshold | Sections 2, M7 | +0.001–0.002 on test | 0.5 h |

Sum of the midpoints ≈ **+0.015–0.020**, which is what 0.985 needs. Not all of it will materialise,
so I1, I2, I5 carry the plan; the rest are insurance.

**Deliberately not doing now:** a fine-tuned transformer encoder (big cost, uncertain gain on an
orthographic problem); GPU training (measured slower, see Progress_Report); record-to-record clustering
(complex; revisit only if I1–I5 stall).

---

## 6. Validation v2 — make it look like the test

1. **Full competition:** evaluate on businesses whose *all* competitors are present — score the whole
   train candidate set for the evaluated countries rather than a 30% slice, or at minimum evaluate the
   one-owner rule only on records whose true owner is in the sample.
2. **Distractor ratio:** re-weight (or add) unmatched records so validation has ~5.8 records per business,
   like test; report F0.5 at both ratios.
3. **Unseen-country proxy for France:** train on US only → score India, and India only → score US.
   The drop tells us how much a new country costs and whether a change helps generalisation.
4. **Report every experiment as:** validation F0.5 (test mix), cold-country F0.5, and expected LB =
   validation − measured gap. Calibrate the gap after each upload.

---

## 7. Blockers and risks

| # | Blocker | Impact | Mitigation |
|---|---|---|---|
| **B1** | **Runtime**: full pipeline ~4.5 h; stage 2 alone ~1 h 45 min (25 min/fold). More data (I7) makes it worse. | Few iterations left before the deadline | Negative down-sampling (easy negatives are ~85% of rows); 3 folds; reuse cached candidates/features (only recompute changed feature groups); experiment on a 10% slice first |
| **B2** | **Memory**: commit limit 38 GB, other apps hold ~14 GB; three background kills in run 1 | Crashes, lost hours | Keep 500k-pair chunks; close Chrome during runs; never run two heavy jobs at once |
| **B3** | **No France labels** | Cannot measure 15% of the test | Cold-country proxy (Section 6); hand inspection; conservative France threshold |
| **B4** | **Validation ≠ test** (Section 2) | Tuning in the wrong environment | Validation v2 (Section 6) before tuning thresholds |
| **B5** | **Irreducible ambiguity**: ~33k blank-address copies of shared names | Caps recall; ≈ 0.003 F0.5 | Accept; do not chase it — any guess costs more precision than it gains |
| **B6** | **Leaderboard submissions** — daily limit unknown ❓ | Can't A/B everything on the LB | Check the portal limit; upload only candidates that beat v2 on validation v2 |
| **B7** | **Time**: ~45 h left, and the final zip + documentation need ~3 h | Running out of time | Freeze experiments by Sep 27, 15:00 IST; zip + docs 15:00–18:00; keep v2 as the safe fallback |
| **B8** | Reproducibility audit of the final zip | Disqualification risk if it doesn't rerun | README already has run order + timings; rerun predict from cache before submitting |

Checked and ruled out: **no leakage** in row order (Spearman −0.002) or ID numbers (Pearson 0.001).
Top scores come from modelling, so they are reachable.

---

## 8. Execution order

| When (IST) | Work | Deliverable |
|---|---|---|
| Sep 26, 00:30–04:00 | I9 tie fix, I3 address noise, I1 number features, I6 phonetic key — all in `normalize.py`/`features.py` | features v2 |
| 04:00–06:00 | Validation v2 (Section 6) + negative down-sampling (B1) | trustworthy, faster evaluation |
| 06:00–09:00 | Retrain stage 2 on features v2 → measure each feature group's gain | **submission v3** if it beats v2 |
| 09:00–13:00 | I2 stacked model with score-based competition | **submission v4** |
| 13:00–17:00 | I5 France hardening + I4 alias features + I10 per-country thresholds | **submission v5** |
| 17:00–Sep 27 09:00 | I7/I8 bigger training set + wider shortlist (overnight run) | **submission v6** |
| Sep 27, 09:00–15:00 | Final tuning on validation v2; pick the best by validation *and* LB | final choice |
| Sep 27, 15:00–18:00 | Freeze. Documentation_template.md, zip, final validator run | **final zip** |

---

## 9. Experiment log (fill in as we go)

| Version | Change | Val F0.5 (test mix) | Cold-country | Public LB | Kept? |
|---|---|---|---|---|---|
| v1 | threshold 0.7 | 0.9754 | — | 0.96744 | no |
| v2 | one-record-one-business + 0.7 | 0.9756 | — | **0.9684** | current best |
| v3 | features v2 (I1, I3, I4 vocab, I5 street words, I6, I9) + easy-negative down-sampling | **0.9796** (macro 0.9798; P 0.992, R 0.958; logloss 0.0155 vs 0.0187) | — | **0.97378** | **yes — current best** (gap to val 0.0060, was 0.0074) |
| v4 | + stage-3 stack, S1-side score competition only (I2, `output/v3_s3`) | 0.9798 (macro 0.9800) — **+0.0002, noise** | — | 0.97313 | no (singletons +0.016, 1–3 groups −0.002) |
| exp | **Cold-country test** (v3 features): US-only model → India / India-only → US | India **0.9285** (vs 0.9790 in-domain); US **0.9556** (vs 0.9803) | −0.051 / −0.025 | — | finding: unseen country loses **precision** (0.992 → 0.95); best threshold rises to 0.8–0.9. France (15%) ⇒ most of the LB gap |
| exp | **Self-training simulation** (India as unseen): US-only → + pseudo-labelled India, 1 and 2 rounds | — | 0.9285 → **0.9323** (r1) → **0.9334** (r2); best threshold stays 0.8 | — | small gain (+0.005 of a 0.05 gap): ~2–3% of confident pseudo-positives are wrong and get learned back |
| v3_fr0.85 | v3 scores, France threshold 0.85 (others 0.7) — no retraining | — | — | | candidate (expect +0.0005–0.001) |
| v5 | v3 features + France self-training (1 round), plain and France threshold 0.8 | — | — | | running |
| v5 | v3 features + France self-training (1 round); `output/v5`, `output/v5_fr0.8` | — | sim +0.005 | | ready, not uploaded (5/day limit) |
| **v6** | **100% of train S1s** (32.6M kept rows, 3,561 rounds), `output/v6` (greedy0.75), `output/v6_fr0.85` | **0.9808** (macro 0.9810 on 735k S1s; P 0.994, R 0.957) | — | | **0.976162** | best of iteration 2 |
| v7 | v6 + full out-of-fold scores → record-side score competition (unbiased now) | | | | folds 1–2 training |
