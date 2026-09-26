# Plan 3 — Iteration 3: push past 0.976 towards 0.989

> Written 2026-09-26 12:05 IST. Deadline Sep 27, 21:00 IST. Builds on Plan_2 (causes M1–M9, fixes I1–I10).
> **Target set by the team: ≥ 0.989.** Leaderboard (screenshot 12:03): **#1 0.990556, #2 0.988842, #3 0.988273.**
> So 0.989 is achievable. Correction to my earlier estimate: I called blank-address copies of same-name businesses
> "irreducible" (≈ 0.003). The leaders prove most of that is recoverable — there is a signal we don't use yet.
> First lead: I compare names after dropping legal suffixes, but competing branches may differ exactly in the
> suffix (`… LLP` vs `… Pvt Ltd`) that a blank-address copy keeps. Being tested.

---

## 1. Where we are

| Version | Validation | Public LB | Gap |
|---|---|---|---|
| v2 | 0.9758 | 0.9684 | 0.0074 |
| v3 (features v2) | 0.9798 | 0.9738 | 0.0060 |
| **v6 (100% train data)** | **0.9810** | **0.976162** | 0.0048 |

Remaining loss on validation ≈ 0.019: recall 0.957 (true matches in the uncertain zone), precision 0.994.
**8.8% of all true matches sit in the uncertain band 0.02 < p < 0.98** — that is where the points are.

## 2. Research → what we build

| Source | Idea | Our use |
|---|---|---|
| Ditto (VLDB) and LM-based matching (EDBT 2025, arXiv 2026) | Cross-encoder: a pretrained transformer reads both records together — the accurate state of the art | **C1** Fine-tune `intfloat/multilingual-e5-small` (MIT, 118M params, 100+ languages incl. Hindi/Tamil/Telugu/Bengali/French) as a pair classifier, **only on uncertain pairs** (~4–6%) so it is fast |
| GraLMatch (EDBT 2025) | Match *groups* across sources, not isolated pairs | **C2** Group consistency: does this record agree with the other records confidently assigned to the same business? |
| Plan_2 I2 + 100% coverage | Competition between businesses for a record | **C3** Record-side score competition (unbiased now that every train business has candidates) |
| DADER / DAME (domain adaptation) | Close the gap to an unseen domain | France: self-training was weak in simulation (+0.005 of 0.05); decided by the LB A/B `v6` vs `v6_fr0.85` |

Rejected: fine-tuned LLMs for all pairs (too slow for 43M pairs); copying public repos of other teams in this
challenge (fair-play violation, audited).

## 2b. New organiser rule (2026-09-26 midday): smaller candidate sets rank higher

> "Candidate generation counts toward the final ranking … The approach that generates a smaller candidate set per
> Source 1 entity will be ranked higher in the final evaluation beyond the public/private leaderboard."
> `candidate_pairs.tsv` = exactly the set the matching model scores.

We currently send a **fixed 25 candidates per business** (43M pairs). Plan: **adaptive cut on the stage-1
blocking model's probability** — keep candidates with p1 ≥ t (at most 25). Stage 1 uses only blocking signals
(retriever ranks/scores), so this is genuinely part of candidate generation. Choose t on validation as the
smallest candidate set whose final F0.5 loss is ≤ 0.0002. Report in the documentation: candidates/S1,
recall of the candidate set, reduction ratio vs. all-pairs.

### Measured trade-off (fold 0, 735k S1s, v6 model, one-owner rule @ 0.75)

| Cut p1 ≥ | Candidates/S1 (val) | Candidates/S1 (test) | True matches in candidate set | F0.5 | Loss vs 25 fixed |
|---|---|---|---|---|---|
| none (fixed 25) | 25.00 | 25.0 | 0.9901 | 0.98098 | — |
| 0.001 | 13.64 | 15.05 | 0.9884 | 0.98092 | −0.00006 |
| **0.003** | **9.15** | **10.22** | 0.9861 | 0.98072 | **−0.00026** |
| 0.01 | 6.21 | 7.19 | 0.9817 | 0.98040 | −0.00058 |
| 0.03 | 5.09 | 6.10 | 0.9765 | 0.98005 | −0.00093 |

→ Default **p1 ≥ 0.003**: ~2.5× smaller candidate set for −0.0003. A stronger blocking model (stage 1 with cheap
string features) could shrink it further without loss — candidate for v9.

### Blank-address "ambiguity" re-checked (fold 0 missed blank-address pairs: 49,756; 38,045 with same-key competitors)
Raw name with legal suffix kept — record closer to its true S1: 16.7%, tie 59.2%, competitor closer 24.1%;
**76.8% have a competitor with an IDENTICAL raw name.** → The suffix idea is NOT the leaders' signal; these are
genuinely identical names. Remaining lever for them: copy-count priors (a business that already has its usual
number of confident copies is less likely to own another) — covered by `g_n_conf`, `g_n_conf_same_src` in v8.

## 3. Pipeline v8

```
v6 stage-2 p (out-of-fold for all 2.2M train S1s; final model on test)
   ├─ C1 cross-encoder logit   (uncertain band only; trained on fold-0 pairs)
   ├─ C2 group-consistency     (similarity to the business's other confident records)
   ├─ C3 record-side competition + S1-side context
   └─→ stage-3 LightGBM, trained/validated on folds 1–2 (where C1 is out-of-sample)
        → one-record-one-business + threshold → output/v8
```

## 4. Overfitting safeguards

1. **No leakage:** the cross-encoder never scores pairs of businesses it was trained on (trained on fold 0,
   used on folds 1–2 and test). Stage 3 uses inner cross-validation on folds 1–2.
2. **Test-like validation:** folds 1–2 = 1.47M businesses, every competitor present (100% coverage) —
   sampling noise ≈ ±0.0002.
3. **Choices by validation, not by the public LB:** a version is uploaded only if it beats the previous best on
   validation by > 0.0005. The LB is used for large moves and for the France A/B, never for fine-tuning thresholds.
4. **Final pick:** best on validation *and* consistent on the public LB; last upload = that version (in case
   the private LB uses the last submission).

## 5. Timeline

| Time (IST) | Step |
|---|---|
| 12:00–13:00 | C1 cross-encoder training (GPU) — v6 folds 1–2 train in parallel (CPU) |
| 13:00–13:30 | C1 scores test uncertain pairs |
| ~14:30 | v6 folds 1–2 done → C1 scores train folds 1–2 |
| 14:30–15:30 | C2 group features + stage-3 v8 + validation |
| ~16:00 | `output/v8` validated → upload |
| Sep 27 | France decision from LB A/B; final tuning; freeze 15:00; zip + docs |

## 6. Log

| Version | Change | Val F0.5 | Public LB | Kept? |
|---|---|---|---|---|
| v6 | 100% train | 0.9810 | **0.976162** | yes — current best |
| v6_fr0.85 | France threshold 0.85 | — | ~0.976 (same as v6) | **no** — France's gap is not a threshold/calibration issue; the model mis-ranks French pairs → needs better representation (cross-encoder) |
| v8 | + C1 cross-encoder + C2 group + C3 record competition | | | |
