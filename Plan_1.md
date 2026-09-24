# Plan 1 — Amazon ML Challenge 2026: Business Entity Resolution

> **Status:** Iteration 1 (pre-code). Created 2026-09-25.
> **This is a living document.** When an approach fails, we do not delete it — we move it
> to the *Iteration Log* with the measured number and the reason, then write Plan 2's
> section. Every claim here should eventually carry a measured value, not a guess.
>
> **Legend:** ✅ measured on full data · 🔬 measured on a sample · ❓ unverified assumption

---

## 0. The task in one paragraph

Three lists of businesses, no shared IDs. **Source 1** is the clean deduplicated master
(one row = one real business). **Source 2** and **Source 3** are noisy and may contain the
same business, spelled differently, with a mangled or missing address, or not at all. For
every Source-1 entity, output the set of S2/S3 records that are the *same real business*.
Only `business_name`, `business_address`, `country` are available. Scored on
**macro F0.5** (precision-weighted). External lookups are disqualifying.

---

## 1. Hard facts (measured, not assumed)

### 1.1 Scale ✅

| | Source 1 | Source 2 | Source 3 |
|---|---|---|---|
| **Train** | 2,206,821 | 5,034,616 | 5,285,603 |
| **Test** | **1,732,544** | 4,887,274 | 5,082,317 |

26.4M rows, ~2.5 GB TSV. Naive comparison space ≈ **1.7 × 10^13 pairs**.
At 20 candidates/entity, inference is **~35M pairs**.

> ⚠️ Our own `Amazon_ML_Challenge_..._Analysis.md` reasons throughout with
> *"suppose 100,000 Source-1 records."* Reality is **20× that.** Every capacity estimate
> in that doc must be re-derived before it is trusted.

### 1.2 Ground-truth match-set sizes ✅ (all 2,206,821 train rows)

```
size 0 (singleton)   123,247    5.58%
size 1               119,157    5.40%
size 2               375,212   17.00%
size 3               530,841   24.05%   <- mode
size 4               484,115   21.94%
size 5               321,957   14.59%
size 6               164,868    7.47%
size 7                63,968    2.90%
size 8+               23,456    1.06%

mean = 3.46   max = 11   |   74% of entities have 2-6 matches
```

Consequences:
- **All-empty submission scores 0.056.** We cannot play safe; we must commit to ~3.5 matches/entity.
- **`argmax` (one best match per entity) is structurally capped near 0.5.** Never use it.
- Singletons are **5.58%**, not a central concern. (Our analysis doc §21/§39 over-weights them.)

### 1.3 Country distribution ✅ — the train/test shift

| | Train | Test |
|---|---|---|
| US | 1,323,633 (60%) | 663,106 (38%) |
| India | 883,188 (40%) | 809,986 (47%) |
| **France** | **0 (0%)** | **259,452 (15%)** |

Two distinct problems: **(a)** France is a cold start — 15% of test entities from an unseen
country; **(b)** even the US/India ratio inverts, so an unweighted validation split will lie to us.

### 1.4 Data quality ✅

| | empty name | empty address | avg name len | avg addr len |
|---|---|---|---|---|
| train S1 | 0.00% | 0.00% | 24.0 | 52.1 |
| train S2 | 0.00% | 3.36% | 30.3 | 47.8 |
| train S3 | 0.00% | 3.33% | 28.0 | 48.2 |
| test S1 | 0.00% | 0.00% | 23.9 | 57.3 |

Names are never empty. **Addresses are the only field that goes missing, and only in S2/S3.**

---

## 2. 🔑 The two findings that drive the whole design

Neither appears in any supplied document or in our own prior analysis.

### 2.1 Matches form a perfect partition — zero ID reuse ✅

```
distinct matched S2/S3 IDs:     7,638,365
appearing in >1 S1 match set:           0   (0.0000%)
maximum reuse:                          1
```

**Every S2/S3 record belongs to at most one Source-1 entity. Zero exceptions in 7.6M IDs.**

This converts the problem from *"independently score 35M pairs"* into a **constrained
assignment problem**, and it is the single strongest signal available:

- If `S2-X` scores 0.95 for `S1-A` and 0.60 for `S1-B`, `S1-B` must not receive it — no
  threshold tuning needed to know that.
- Post-process: sort all candidate edges by score descending, accept an edge only if that
  S2/S3 record is still unclaimed (greedy exclusive assignment).
- This eliminates a whole class of false positives **for free** — precisely the errors F0.5
  punishes hardest, and precisely the generic-name-collision trap ("ABC Trading").

**Action:** implement as a post-processing pass immediately after the first scored model.
Measure the F0.5 delta. Expected to be one of our largest single gains.

> ❓ **Must verify on test:** we cannot confirm the constraint holds in the test split, but a
> generator that produced it for 7.6M train IDs almost certainly did the same for test.
> Fallback if the exclusive pass *hurts* validation: apply it as a soft feature
> (`n_other_S1_competing_for_this_record`, `margin_over_second_best_claimant`) instead of a
> hard constraint.

### 2.2 A blank address is *evidence of a match*, not missing data ✅

| | matched | unmatched (distractor) |
|---|---|---|
| Source 2 | **4.47%** | 0.28% |
| Source 3 | **4.36%** | 0.30% |

**A blank address makes a record ~15× more likely to be a true match.**

Why: the data is synthetic. A true copy was made from an S1 seed and noise operators were
applied — "delete the address" is one of them. Distractors were generated separately and
kept their addresses.

Consequences:
- **Never drop, impute, or deprioritize blank-address records.** Feed `address_is_empty`
  in as an explicit feature.
- Blocking **must have a name-only path**, because for these records the address carries
  zero information and any address-gated rule silently discards them.

### 2.3 Distractor structure ✅

10,320,219 S2+S3 train records; 7,638,365 matched → **2,681,854 distractors (26.0%)**.
Split almost exactly evenly: **1,340,997** in S2 vs **1,340,857** in S3. Decoys were
injected deliberately and symmetrically. Expect ~26% decoys in test too.

---

## 3. The noise model — reverse-engineering the generator

**This is the highest-ROI work available.** The data was machine-generated from a finite
list of perturbation rules. Every operator we identify and invert is recall gained almost free.

### 3.1 Worked example — real group `S1-965667`

```
S1  Maure Williams Colombier Inc    | 85 Wayne Avenue, Ticonderoga, NY
----------------------------------------------------------------------------
S2  Maure Wilblims Colombier Inc    | (empty)          <- char-level typo
S2  Maure Williams Colombier        | (empty)          <- legal suffix dropped
S3  Maure Williams Inc Center       | (empty)          <- token dropped + word added
S3  maurewilliamscolombier.com      | Wayne Ave, Ticonderoga Townshiip, New York
S3  Dréxkor                         | 85 Wanye Avenue, Ticonderoga Townshiip, New York
```

The last two rows are the ones that break naive pipelines:

- **`maurewilliamscolombier.com`** — name squashed to a domain. **No whitespace => token
  Jaccard = 0.0.** Recoverable only via a space-stripped form + character n-grams.
  Its address also lost the house number (`85`).
- **`Dréxkor`** — a totally unrelated alias. **Zero name overlap.** Recoverable *only*
  through the address — which itself has a typo (`Wanye`), an expanded state
  (`NY` -> `New York`), and an inserted word (`Townshiip`, itself misspelled).

> **Design conclusion:** blocking must be an **OR over independent name and address paths**,
> never an AND. An AND rule loses `Dréxkor` (name useless) *and* the three blank-address
> rows (address useless) — 4 of 5 true matches in this single group.

### 3.2 Second example — transliteration (`S1-55344266`, "Raj Investments LLP", Chennai)

```
S2  Raj Investments LLP                        | 6(29), C.I.T. COLONY, ... Tamil Nadu
S2  ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி  | 6(29), C.I.T. COLONY, ... Tamil Nadu
S3  ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி  | ... Chennai, தமிழ்நாடு
S3  Raj Investments எல்எல்பி                  | ... Chennai, TN
```

**Full and partial transliteration into Indic scripts is a first-class operator.** Note
`Raj Investments எல்எல்பி` — Latin and Tamil mixed in one string. The state appears three
ways: `Tamil Nadu` / `தமிழ்நாடு` / `TN`.

### 3.3 Measured operator frequencies 🔬 (400k-row sample per file)

| Operator | S2 | S3 |
|---|---|---|
| Non-Latin-script name | **8.98%** | 4.65% |
| Double-space injection | 11.00% | 10.92% |
| Parentheses inserted | 4.98% | 5.09% |
| Domain-style name (`.com`/`.in`/`.net`/`.org`) | 3.15% | 3.14% |
| Empty address | 3.33% | 3.29% |
| Junk prefix (`--`, `<<`, `>>`) | 0.90% | 0.88% |

### 3.4 Per-source formatting conventions 🔬

- **S1** addresses: `Title Case`
- **S2** addresses: `ALL CAPS`
- **S3** addresses: `Mixed Case`

Systematic. Feed `source_pair` as a feature so the model can learn separate boundaries for
S1<->S2 vs S1<->S3.

### 3.5 Operator catalogue — to be completed in Step 2

Confirmed so far. **Fill in frequencies and inversion status as we measure them.**

| # | Operator | Field | Freq | Invertible? | Handled by |
|---|---|---|---|---|---|
| 1 | Character typo (transpose/substitute) | name, addr | ❓ | no — tolerate | char n-gram, edit distance |
| 2 | Legal suffix dropped/changed | name | ❓ | yes | suffix normalizer |
| 3 | Token dropped | name | ❓ | no — tolerate | token containment, not Jaccard |
| 4 | Token inserted | name | ❓ | no — tolerate | token containment |
| 5 | Domain-form name | name | 3.15% 🔬 | yes | space-stripped form |
| 6 | Full alias / rebrand (zero overlap) | name | ❓ | **no** | address-only path |
| 7 | Transliteration to Indic script | name, addr | 4.6-9.0% 🔬 | partly | learned translit map |
| 8 | Partial (mixed-script) transliteration | name | ❓ | partly | per-token translit |
| 9 | Address deleted | addr | ~4.4% ✅ | **no** | name-only path + feature |
| 10 | House number dropped | addr | ❓ | no | numeric-token feature, absent-safe |
| 11 | Street type abbreviated (`Avenue`->`Ave`) | addr | ❓ | yes | street-type normalizer |
| 12 | State abbrev <-> expanded (`NY`<->`New York`, `TN`<->`Tamil Nadu`) | addr | ❓ | yes | state map (learned) |
| 13 | Component reordered | addr | ❓ | yes | order-invariant token set |
| 14 | Word inserted (`Township`) | addr | ❓ | no | token containment |
| 15 | Case change (per-source) | addr | ~100% 🔬 | yes | casefold |
| 16 | Double space | both | ~11% 🔬 | yes | whitespace collapse |
| 17 | Parentheses inserted | name | ~5% 🔬 | yes | punctuation strip |
| 18 | Junk prefix | name | ~0.9% 🔬 | yes | prefix strip |
| 19 | Accent/diacritic change | both | ❓ | yes | NFKD fold — **but keep raw** |
| 20 | `null` literal in address | addr | ❓ | yes | treat as missing token |

---

## 4. The metric, and what it licenses

```
F0.5 = (1.25 × P × R) / (0.25 × P + R)     per S1 entity, then plain-averaged
```

Worked at the modal case (4 true matches):

| Action | P | R | F0.5 |
|---|---|---|---|
| all 4, clean | 1.00 | 1.00 | **1.000** |
| 3 of 4, no junk | 1.00 | 0.75 | **0.938** |
| all 4 **+ 1 false** | 0.80 | 1.00 | **0.833** |
| only 2 of 4, no junk | 1.00 | 0.50 | **0.833** |

- Missing one true match costs **0.06**.
- Adding one false match costs **0.17** — nearly **3×** worse.
- One FP ≈ missing *half* the true matches.

**Calibration rule:** we can afford to drop our single weakest candidate almost freely.
We cannot afford to add a speculative one. But note the floor: all-empty = 0.056, so
timidity is not a strategy — this is a tie-breaker, not a licence to abstain.

Because scoring is a **macro-average over entities**, an entity with 8 matches counts
exactly as much as a singleton. Do not optimise a pair-level objective and assume it transfers.

---

## 5. Pipeline architecture

```
  load (polars, sep="\t")
        |
        v
  NORMALIZE  -> keep raw + normalized + structured, never overwrite raw
        |
        v
  BLOCKING  (union of independent paths — OR, never AND)
        |-- char 3/4-gram TF-IDF on name        -> sparse top-k
        |-- char 3/4-gram TF-IDF on address     -> sparse top-k
        |-- numeric-token inverted index (house no. / PIN)
        |-- name-only path (for blank-address records)
        +-- space-stripped name exact/prefix    (domain forms)
        |
        v
  candidate_pairs.tsv   <- the audited artefact; must be the LAST stage before scoring
        |
        v
  PAIR FEATURES  (vectorized; ~35M rows)
        |
        v
  LightGBM / CatBoost  -> P(match)
        |
        v
  EXCLUSIVE ASSIGNMENT  (§2.1 partition constraint)
        |
        v
  THRESHOLD (tuned on true macro-F0.5, country-reweighted)
        |
        v
  aggregate per S1 + fill every missing S1 with an empty list
        |
        v
  matching_results.tsv  -> validate_submission.py -> upload
```

### 5.1 Feature list (v1)

Name: exact, normalized-exact, space-stripped-exact, token Jaccard, **token containment**
(asymmetric — handles dropped/inserted tokens), char 3-gram cosine, edit similarity,
Jaro-Winkler, longest-common-token-prefix, length delta, token-count delta.

Address: normalized-exact, token Jaccard, token containment, char 3-gram cosine,
**house-number equality** (3-state: equal / differ / absent), street-token overlap,
city equality, state equality (post-map), numeric-token set overlap, length delta.

Structural: `address_is_empty` (§2.2), `source_pair` (S1S2 / S1S3), `country`,
`same_script` / `script_mismatch`, `name_has_digits`, blocking-rule-that-fired (one-hot),
**candidate rank**, **score margin over 2nd best**, `n_candidates_for_this_S1`,
`n_S1_competing_for_this_record` (§2.1 soft form).

---

## 6. ⚠️ Pitfalls — things to be wary of while coding

Ordered by how much damage they do.

### 6.1 Brute force is not a starting point here — it is a wall
The instinct on a new problem is "compare everything, optimise later." At
1.73M × 10M ≈ **1.7 × 10^13 pairs**, that is not a slow first draft; it never finishes.
Even 1% of it is 170 billion comparisons.
**Rule: no code path may ever materialise a pair without a blocking key having produced it.**
If a function signature takes two full dataframes and returns a cross product, it is wrong.
The brute-force "baseline" is only legal on a **tiny sampled subset** (e.g. 2,000 S1 × 20,000 S2)
and *only* to sanity-check feature code — never as a candidate generator.

### 6.2 Blocking sets a hard recall ceiling — measure it *before* modelling
`Recall_final <= Recall_blocking`, always. A true pair excluded at blocking is
**unrecoverable** by any downstream model.
**Rule: never train a matcher before printing blocking recall on a labelled sample.**
Report it together with candidates-per-S1, or the number is meaningless — 100% recall at
5,000 candidates/entity is not a result. Track both: **recall** *and* **reduction ratio**.

### 6.3 AND-ing blocking rules silently destroys recall
`name_similar AND address_similar` looks precise and is the natural thing to write. In the
worked example (§3.1) it loses **4 of 5** true matches. Blocking is a **union**; precision is
the *matcher's* job, not blocking's. Keep the two stages' goals strictly separate:
blocking = permissive, matcher = conservative.

### 6.4 Over-normalizing collapses distinct businesses
Normalize enough to remove formatting noise, not so much that different entities become
identical. Stripping every suffix makes `ABC Trading Ltd` and `ABC Trading Pvt Ltd`
indistinguishable — and §47 of the analysis doc notes those may be *different* entities.
**Rule: normalization adds columns; it never overwrites `business_name` / `business_address`.**
Let the model see raw and normalized and decide.

### 6.5 Do not treat blank addresses as missing data
The reflex is `dropna()` or impute. Here blank is a **15× positive signal** (§2.2).
`dropna()` on address deletes ~4.4% of true matches and is an unrecoverable recall loss.

### 6.6 Token-based similarity fails on domain-form names
`maurewilliamscolombier.com` has no whitespace => Jaccard = 0.0 against anything.
3.15% of S2/S3 names are domain-style. **Always compute a space-stripped name form and
character n-grams alongside token metrics.**

### 6.7 Jaccard punishes dropped/inserted tokens; use containment too
`Maure Williams Colombier Inc` vs `Maure Williams Inc Center` — symmetric Jaccard is
mediocre, but asymmetric **containment** (`|A∩B| / min(|A|,|B|)`) stays high. Operators 3
and 4 in the catalogue are exactly this. Ship both metrics.

### 6.8 Validation must mirror the test distribution, or it will lie
Train is 60/40 US/India with **no France**; test is 38/47/15. An unweighted random split
over-weights US by ~1.6×. Worse, nothing in training measures France performance at all.
**Rule: (a)** reweight the validation split to the test country mix; **(b)** run a
train-US -> validate-India holdout to *estimate* cold-country degradation. That second
number is our France risk, and the **public leaderboard will hide it until the private reveal.**

### 6.9 Never one-hot or filter on `{US, India}`
The README warns twice. Any `if country in ('US','India')` branch silently mishandles
259,452 test entities. Country-agnostic features (char n-grams, numeric tokens, generic
token overlap) are the backbone; country-specific rules are a bonus layer with a default path.

### 6.10 Pair-level accuracy is a meaningless metric here
~26% of S2/S3 are distractors and most candidate pairs are negatives, so
"predict everything non-match" scores high accuracy and zero usefulness.
**Only ever report macro-F0.5 computed exactly as the scorer does**, per entity then averaged.
Build that evaluator in Step 1 and use nothing else for decisions.

### 6.11 Threshold 0.5 is arbitrary
Sweep 0.05->0.95 against real macro-F0.5. Because F0.5 is precision-heavy the optimum will
likely sit high — but **measure it**, do not assume it. Consider a per-country and
per-source-pair threshold once the single global one is tuned.

### 6.12 Format bugs lose competitions; malformed CSV is the #1 rejection
Tab-separated, UTF-8, exactly **1,732,544** rows, empty string (not `NaN`, not `"none"`)
for singletons, no duplicate IDs in a list, S2-/S3- only.
**Always** `df.to_csv(path, sep='\t', index=False, encoding='utf-8')`.
**Always** run `utils/validate_submission.py` before uploading. Run it once with
`--check-ids` (off by default; needs several GB) before the final submission.

### 6.13 `candidate_pairs.tsv` must be the LAST blocking stage
Not an early broad pass we later filter. It must be *exactly* what the model ran inference
over. Every ID in `matching_results.tsv` must appear in it or the validator warns and the
audit flags a pipeline bug.

### 6.14 Memory — this does not fit in pandas comfortably
2.5 GB raw; the `--check-ids` validator pass alone costs several GB. Use **polars** and
sparse matrices, stream where possible, and persist normalized frames to parquet so we never
re-normalize. Watch for the 35M-row feature matrix: float32, not float64.

### 6.15 Reproducibility is graded
Top teams' zips are audited. Seed everything, pin versions in `requirements.txt`, and keep
the pipeline runnable end-to-end from `code/business_entity_resolution/`. Do not let the
winning result live only in a notebook's kernel state.

### 6.16 Do not reach for an LLM/VLM
This is an **orthographic** problem (typos, abbreviations, transliteration, token drops),
not a semantic one. A generative model adds latency, VRAM pressure, and reproducibility risk
for little gain. Embeddings are a *feature source*, added late, measured against the cost.
Licence constraint anyway: **MIT/Apache-2.0, <=8B params.**

### 6.17 Beware transitivity and false chains
`A~B`, `B~C` does **not** imply `A~C` in a noisy ER system. Use cross-source consistency as
a *feature*, not as a merge rule. The §2.1 exclusivity constraint is the safe structural
prior; free-form clustering is not.

### 6.18 External data is disqualifying — including "harmless" helpers
No geocoding, no postal-code tables, no company registries, no web lookups. A
transliteration or state-abbreviation map **learned from the provided training pairs** is
fine and is the right way to build one. A downloaded gazetteer is not. Document the
provenance of every map we build, because the audit will ask.

---

## 7. Work plan

### Step 1 — End-to-end skeleton (do first, before any accuracy work)
- [ ] Polars loaders with explicit `sep="\t"`; persist to parquet
- [ ] **Exact scorer**: per-entity macro-F0.5, identical semantics to the challenge
- [ ] Validation split, country-reweighted to the test mix
- [ ] Trivial baseline: exact normalized name + city
- [ ] Write both TSVs; `validate_submission.py` -> **PASS**
- [ ] **Deliverable: a submittable artefact.** Prove the plumbing before improving the model.

### Step 2 — Noise-operator catalogue (highest ROI)
- [ ] Sample ~5k ground-truth groups; diff each S2/S3 record against its S1 parent
- [ ] Complete the §3.5 table with real frequencies
- [ ] Derive the transliteration map and state-abbreviation map **from training pairs only**
- [ ] Quantify: what share of true matches are name-recoverable / address-recoverable / both

### Step 3 — Normalization layer
- [ ] Casefold, NFKD, whitespace collapse, junk-prefix and paren strip
- [ ] Country-aware legal-suffix normalizer (US / India / **France**: `SARL`, `S.A.S`, `SCI`)
- [ ] Space-stripped name form (domain operator)
- [ ] Address component parser -> house no. / street / city / state / postal
- [ ] Transliteration folding, per token
- [ ] Assert: raw columns untouched (§6.4)

### Step 4 — Blocking (measure before proceeding)
- [ ] char 3/4-gram TF-IDF top-k on name; same on address
- [ ] Numeric-token inverted index
- [ ] Name-only path for blank-address records (§2.2)
- [ ] Space-stripped exact/prefix path
- [ ] **Report blocking recall + candidates-per-S1 + reduction ratio.** Gate: recall >= 0.97
      at a tractable candidate count, else iterate here and nowhere else.

### Step 5 — Matcher
- [ ] Vectorized pair features (§5.1), float32
- [ ] LightGBM baseline, then CatBoost; compare
- [ ] Hard-negative mining: top-scoring false positives -> retrain
- [ ] Threshold sweep on true macro-F0.5

### Step 6 — Exploit the partition constraint (§2.1)
- [ ] Greedy exclusive assignment by descending score
- [ ] Measure F0.5 delta vs. independent thresholding
- [ ] If it hurts, fall back to the soft-feature form

### Step 7 — Robustness & submission
- [ ] train-US -> validate-India cold-country estimate (France proxy)
- [ ] Per-country / per-source-pair thresholds
- [ ] Optional: embedding similarity as an extra feature, only if it pays for itself
- [ ] Freeze; fill `Documentation_template.md`; build the zip; final validator run with `--check-ids`

---

## 8. Metrics dashboard — fill in every iteration

| Iter | Blocking recall | Cands/S1 | Model | Threshold | Val F0.5 (reweighted) | Public LB | Notes |
|---|---|---|---|---|---|---|---|
| 1 | — | — | exact-match baseline | n/a | — | — | plumbing only |

**Reference floors:** all-empty = **0.056**. Any model must clear that by a wide margin.

**Log per experiment:** normalization version, blocking rules, candidate count, blocking
recall, model, features, threshold, P, R, F0.5, predicted-matches count, singleton fraction,
runtime. Do not rely on memory.

---

## 9. Debugging triage

| Symptom | Look at |
|---|---|
| Low recall | Blocking too tight; an AND rule crept in (§6.3); over-normalization (§6.4); blank addresses dropped (§6.5) |
| High blocking recall, low precision | Matcher too weak; threshold too low; shared-address / generic-name collisions -> apply §2.1 |
| Good local score, bad leaderboard | Validation not reweighted (§6.8); threshold overfit; output format; country shift |
| Too many predicted singletons | Threshold too high; blocking missing; address parser failing on an unseen country |
| Predicting ~1 match/entity | `argmax` leaked in somewhere (§1.2) — 74% of entities have 2-6 matches |
| OOM | float64 in the 35M-row matrix; pandas instead of polars; `--check-ids` (§6.14) |

---

## 10. Iteration log

### Iteration 1 — 2026-09-25, pre-code
**Status:** planning complete, no code written.
**Established:** scale (§1.1), match-set distribution (§1.2), country shift (§1.3),
the zero-reuse partition constraint (§2.1), the blank-address signal (§2.2), distractor
share (§2.3), two worked noise examples (§3.1-3.2), sampled operator frequencies (§3.3).
**Superseded:** `CHALLENGE_BRIEF.md` — predicted a multimodal image/price task; there are no
images. Its *process* advice (dumb baseline in hour 1, trust CV over public LB, experiment
log, freeze early) still stands.
**Corrected in our own analysis doc:** 20× scale error; singleton over-weighting; and it
misses both §2.1 and §2.2 entirely.
**Next:** Step 1 (scorer + baseline + validator PASS), then Step 2 (operator catalogue).

### Iteration 2 — TBD
> If Plan 1 underperforms, record here: what we measured, which assumption broke, and what
> changes. Amend the sections above in place; keep the measured numbers.

---

## 11. Open questions

- [ ] Does the zero-reuse constraint (§2.1) hold in the test split? (Unverifiable directly —
      plan a soft-feature fallback.)
- [ ] What fraction of true matches are **name-unrecoverable** (the `Dréxkor` case)? This
      caps any name-only strategy and sizes the address path's importance.
- [ ] Is the transliteration map closed/deterministic, or generated per record?
- [ ] Do the noise operators differ by country? (France was generated later — possibly by a
      different operator set.)
- [ ] Is the 60-hour vs. advertised 72-hour window discrepancy resolved? (Affects the Day-3 freeze.)
- [ ] Do match-set sizes differ by country? If India averages more matches than US, the
      macro-average shifts with the test mix.

---

## 12. Constraints checklist (non-negotiable)

- [ ] Output tab-separated, UTF-8
- [ ] Exactly one row per test S1 entity — all **1,732,544**, France included
- [ ] Empty string for singletons — not `NaN`, not `"none"`
- [ ] S2-/S3- prefixes only; no S1 self-matches; no IDs absent from the test set
- [ ] No duplicate IDs within a list; no duplicate `source1_entity_id` rows
- [ ] `matching_results.tsv` is a subset of `candidate_pairs.tsv`
- [ ] Final model MIT/Apache-2.0, <=8B parameters
- [ ] **Zero external data / APIs / geocoding / registries** — disqualifying
- [ ] Zip: `output/` (both TSVs) + `code/business_entity_resolution/{src,README.md,requirements.txt}` + filled `Documentation_template.md`
