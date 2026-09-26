# Maximum Accuracy Plan — Explained Simply

> Plain-language version of the doc "Amazon ML Challenge — Maximum Accuracy Strategy"
> (https://claude.ai/code/artifact/89a7deb9-4365-4c77-9d06-b79c40a30b46). Written 2026-09-25.
> Each section says **what it means**, **why it matters**, and **what we do**.

---

## 1. The short version (TL;DR)

Our Plan_1 is already good. To get the maximum score, change six things:

1. **Decide each business's answer as a whole list.** Don't use one fixed "yes if score > X" rule for all pairs.
2. **Only compare records from the same country.** Real matches always share it.
3. **Search in both directions.** Start from each clean business, and also from each messy record.
4. **Expect more fake records in the test set** than in training, and tune for that.
5. **Add a small AI text model** (fine-tuned) to catch other-language and French records.
6. **Convert Indian-script names to English letters** before comparing them.

---

## 2. The problem

- **Source 1** = a clean list of 1.73 million businesses (test set).
- **Source 2 and Source 3** = two messy lists (~10 million records together). They contain copies of those
  businesses, but misspelled, shortened, with missing addresses, written in Hindi/Tamil/etc., or fake
  businesses that match nothing.
- **Our job:** for every clean business, list all its messy copies.
- **Only 3 fields exist:** name, address, country.
- **Countries:** training has US and India only. The test also has **France (15%)**, which we've never seen.

**How we're scored (F0.5):**
- Each business gets its own score, then all the scores are averaged.
- **A wrong match hurts about 3× more than a missed match.**
  Example: the true answer is 4 copies. Finding only 3 of them costs 0.06. Adding 1 wrong one costs 0.17.
- If a business has no copies and we answer "none", we get the full 1.0 for it.

**Rules:** no internet lookups, no outside databases, and the AI model must be MIT/Apache licensed and at most
8 billion parameters.

---

## 3. What's good and what's missing in Plan_1

**Keep these (they're right):**
- Each messy record belongs to **at most one** clean business. Use this to settle conflicts.
- A blank address actually **hints at a real copy**. Never throw those records away.
- Search by name OR by address, never "name AND address".
- Never overwrite the raw text; add cleaned columns next to it.

**What's missing:**

| Problem | Simple explanation |
|---|---|
| One fixed cut-off score | Each business needs its own decision about how many matches to return |
| Country not used | Real matches are always in the same country, so there's no reason to compare across countries |
| Same-name businesses | 31% of clean businesses share their name with another one (branches such as "007 Devices" in 3 cities). The name alone can't tell them apart |
| More fakes in test | Test has more messy records per business (5.8 vs 4.7), so probably more fake records |
| France | New patterns: `R.` = rue, `AV` = avenue, region swapped for department, odd accents like `Àmicale` |
| No AI text model | Plain string comparison hits a ceiling on other-language text |

---

## 4. What we learned from the data and from research

**Checks on our own data:**

| Fact | What it means for us |
|---|---|
| 100% of real matches share the same country | Compare within a country only. We lose nothing and it's 3× faster |
| 31% of clean names are repeated | Use the address to pick the right branch |
| Only 5% of fake records copy a real name | Most fakes are easy to reject. The hard cases are sibling branches |
| An exact address match is almost never a fake (0.2%) | An address match is very strong evidence |
| US and India have the same group sizes (avg 3.46) | France probably does too |

**From research and past competitions:**
- **Kaggle Foursquare matching (a very similar problem):** top teams combined several quick search methods,
  then scored pairs with LightGBM plus a BERT-type model, then cleaned up the result using links between records.
- **Ditto / SC-Block / Sudowoodo (research papers):** fine-tuned transformer models find and match records
  better than plain string rules.
- **F-score papers:** when a score is calculated per item, choosing the answer that gives the best
  *expected* score beats using a fixed cut-off.

---

## 5. The recommended pipeline (step by step)

```
Clean text -> Split by country -> Find look-alikes -> Quick model keeps top 12
-> Smart model scores them -> Settle conflicts -> Pick the best list per business
```

1. **Clean the text.** Lowercase it, remove junk like `<<` and `--`, and make "Ltd/Limited" and
   "Rd/Road/R./Rue" comparable. Do this for the clean list too, because it also has junk.
2. **Split by country.** Only compare US with US, India with India, France with France.
3. **Find look-alikes (shortlist).** Use several methods and combine their results:
   - similar spelling of the name
   - similar address
   - same house number or PIN code
   - AI-model similarity (catches other scripts and heavy abbreviations)
   - name-only search (for copies with a blank address)
   - **reverse search:** for each messy record, which clean businesses look like it?

   Target: the shortlist should contain ≥ 98.5% of the real copies.
4. **Quick model** (LightGBM) scores the shortlist and keeps the **top 12** per business (the biggest group
   is 11). This top-12 list is our `candidate_pairs.tsv`.
5. **Smart model** gives each pair a probability that it's the same business. It uses:
   - spelling and address similarity scores
   - an AI "is this the same business?" score (fine-tuned model)
   - **competition:** does another clean business look like a better owner for this record?
   - **group fit:** does this record also resemble the business's other copies? Real copies look alike; fakes don't.
6. **Settle conflicts.** A record can belong to only one business. If two businesses claim it, the stronger
   claim wins, and the probabilities are split fairly between them.
7. **Pick the best list per business.** For each business, try "return 0, 1, 2, … top matches" and keep
   the option with the best *expected* score. Returning an empty list is allowed, which is how true
   singletons get their full 1.0.

---

## 6. Indian languages (Hindi, Tamil and others)

**The size of the problem:** about **24% of Indian names** in Source 2 are written in an Indian script,
across **9 scripts**: Hindi/Marathi (Devanagari, the most common), Telugu, Kannada, Tamil, Gujarati, Bengali,
Malayalam, Odia and Punjabi. Some names mix English and Indian letters, e.g. `Raj Investments எல்எல்பி`.

**Key idea:** the data **writes English words in Indian letters (transliteration)**. It does not translate
them. `ராஜ் இன்வெஸ்ட்மெண்ட்ஸ்` is just "Raj Investments" spelled in Tamil letters. So we convert the letters
back to English. We don't translate the meaning.

**Can we use translation models?**

| Tool | Allowed? | Use it? |
|---|---|---|
| Our own word table, learned from training pairs | Yes (safest) | **Yes, first choice** |
| IndicXlit (converts Indian letters to English letters; MIT, tiny, 21 languages) | Yes (MIT model) | **Yes, for words the table hasn't seen** |
| IndicTrans2 (real translation; MIT) | Yes | Only as an experiment. It may translate names into ordinary words |
| NLLB (Meta translation) | **No** (non-commercial licence) | No |

**Steps:**
1. Detect which script each word is in.
2. From training pairs (English name ↔ Indian-script copy), build a word-by-word table,
   e.g. `இன்வெஸ்ட்மெண்ட்ஸ்` → `investments`, `தமிழ்நாடு` → `tamil nadu`.
3. Convert each Indian word with the table, and use IndicXlit for unknown words. Convert word by word, so mixed
   names work too.
4. Save the result as a new column (`name_roman`); keep the original.
5. Search and score on both the original and the converted text.
6. Check the score for Indian-script records separately, to be sure they aren't a weak spot.

**Do we need "semantic" embeddings (AI meaning vectors)?**
Yes, but only as a **helper**, and only after training them on our pairs.
- The mistakes in this data are mostly **spelling** changes, not meaning changes.
- An untrained embedding thinks "ABC Pharmacy" ≈ "XYZ Pharmacy" (both pharmacies), which causes the
  wrong matches we must avoid.
- After fine-tuning (multilingual-e5, MIT licence), it's good for **finding** other-script and abbreviated
  copies, and as **one extra score** for the smart model.
- Nothing can recover a completely renamed business (like `Dréxkor`); only its address can.

---

## 7. Checking our work (validation)

Our test copy must look like the real test, or our numbers will be too optimistic.

| Check set | What it is | What it tells us |
|---|---|---|
| Main | 10% of training businesses held out, weighted to test's country mix (US 38% / India 47%) | Our main score |
| Extra-fakes | Same, with extra fake records added (5.8 per business, like test) | Whether our cut-offs survive more fakes |
| Unseen-country | Train on US only, test on India | Roughly how much we'll drop on France |
| Full search | Shortlist against *all* messy records, not just the held-out ones | Honest shortlist quality |

**Habits:**
- Report scores separately for singletons, small groups (1–3) and big groups (4+).
- Our test predictions should average about 3.5 matches per business. If not, something has drifted.
- Trust our own check sets more than the public leaderboard (the final ranking uses a hidden part of the test set).

---

## 8. Timeline (about 60 hours)

| Hours | Do this | Done when |
|---|---|---|
| 0–6 | Load data, build the scorer, make a simple first submission | First score on the leaderboard |
| 6–16 | Cleaning, country split, look-alike search | Shortlist has ≥ 97% of the real copies |
| 16–26 | Quick + smart models, conflict settling, best-list picking | Beats the fixed cut-off locally |
| 26–40 | Train the AI models, Indian-script conversion, group-fit features | Shortlist ≥ 98.5%; score goes up |
| 40–50 | Combine models, per-country tuning, France extras | Stable on all check sets |
| 50–60 | Freeze, rerun everything from scratch, write the docs, zip | Rerun output = uploaded file |

**Computer:** RTX 4090 laptop (16 GB). Run the heavy AI model only on the top 12 per business, in half
precision, overnight.

---

## 9. Checklist

- [ ] Scorer reproduces the example score (0.714)
- [ ] Country split added, and we verified it loses no real matches
- [ ] Reverse search added
- [ ] Conflict settling beats simple "first come, first served"
- [ ] Best-list picking beats one fixed cut-off
- [ ] Extra-fakes check set built
- [ ] Unseen-country (US → India) score recorded
- [ ] Indian-script conversion built and scored separately
- [ ] Asked organisers whether we may use unlabelled test records (for France)
- [ ] Every learned table documented (where its data came from)
- [ ] `candidate_pairs.tsv` = exactly what the final model scored

---

## 10. Sources

- Ye et al., *Optimizing F-measures* (2012) — https://arxiv.org/abs/1206.4625
- Lipton et al., *Thresholding Classifiers to Maximize F1* (2014) — https://arxiv.org/abs/1402.1892
- Ditto (2021) — https://arxiv.org/abs/2004.00584
- SC-Block (2023) — https://arxiv.org/abs/2303.03132
- Sudowoodo (2022) — https://arxiv.org/abs/2207.04122
- TransClean (2025) — https://arxiv.org/abs/2506.04006
- Foursquare Location Matching, 7th place write-up — https://future-architect.github.io/articles/20220720a/
- IndicXlit — https://github.com/AI4Bharat/IndicXlit
- IndicTrans2 — https://github.com/AI4Bharat/IndicTrans2
- multilingual-e5-base — https://huggingface.co/intfloat/multilingual-e5-base
