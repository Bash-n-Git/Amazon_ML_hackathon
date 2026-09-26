# Amazon ML Challenge 2026 — Prep Brief

> Compiled 2026-09-24, the evening before the sprint.
> **Caveat:** Unstop's page is JavaScript-rendered and could not be read directly.
> Timeline below comes from the Internshala mirror + AWS Builder Center posts.
> **Verify exact start/end times on your Unstop dashboard tonight.**

---

## 1. The Challenge

| Item | Detail |
|---|---|
| Hackathon window | **Sep 25, 9:00 AM IST → Sep 27, 9:00 PM IST** |
| Results | Oct 2, 2026 |
| Grand Finale | Oct 7, 10 AM–3 PM IST (virtual, present to Amazon Scientists) |
| Teams | 3–4 members, one leader, cross-college allowed |
| Eligibility | B.E./B.Tech/M.E./M.Tech/MS/PhD, **2027 or 2028** batch |
| Prizes | ₹1,00,000 / ₹75,000 / ₹50,000 + certificate + merch |
| The real prize | **Top 50 teams → PPI for Applied Scientist Intern** |
| Extra | Top 10 teams and top 10 women-only teams: certificates + merch |

**Discrepancy to resolve:** it's advertised as a "72-hour" hackathon, but the listed
dates span **60 hours**. Confirm which is correct — it changes the Day 3 plan materially.

### Submission requirements
- Predictions file
- Code / script / notebook, **zipped**
- **1–2 page document describing your approach**

Live **public leaderboard** during the sprint; **private leaderboard** on the full
test set revealed afterward. Budget explicit time for the write-up — it is not optional.

---

## 2. What the Problem Will Probably Be

The statement drops at **9:00 AM IST on Sep 25**. History of past editions:

| Year | Task | Data | Metric |
|---|---|---|---|
| 2021 | Browse-node classification | ~3M products; title, description, bullets | — |
| 2023 | Product length regression | Tabular catalog metadata | `max(0, 100*(1-MAPE))` |
| 2024 | Entity value extraction from **images** (weight, volume, voltage, dimensions) | Image links + metadata | — |
| 2025 | **Price prediction** | 75k train / 75k test, `catalog_content` + `image_link` | **SMAPE** |

2025 explicitly **banned external price lookups**. Expect a similar "no external data" clause.

### Working bet for 2026
Three of the last four were Amazon product-catalog tasks; the last two were
**multimodal (text + image)**. Most likely shape:

- ~50k–150k rows
- A text field (title / description / bullets) **plus** an image URL column
- Scored on a percentage-error metric (SMAPE/MAPE) or F1

**Prepare for that shape.** If it turns out pure-tabular, gradient boosting gets you
there fast anyway.

---

## 3. The Two Direct Questions

### Is the Sep 21 virtual session necessary?
**No.** It's a "Best Practices" session covering:
- AWS Builder Center setup & benefits
- AWS Free Tier & compute credits
- AWS core services walkthrough
- Live demo: train, deploy & test an ML model

Nothing in it affects your leaderboard score — you are ranked purely on predictions.
Its one concrete value is the **AWS credits**. Find the recording, skim at 2x tonight,
complete whatever credit-claim step it describes. **Do not spend Day 1 learning SageMaker.**

### Should you learn AWS Builder?
**No** — not as a skill, not this week. AWS Builder Center is a community/profile
portal, not a tool you build in. Create the account, claim the credits, move on (~20 min).

### ⚠ The one non-obvious AWS trap
A **fresh AWS account has a GPU vCPU service quota of zero.** Requesting an increase
for `G` instances (g5.xlarge / g6.xlarge) takes **hours to several days**. If you intend
to train on AWS at all, **file that quota request TODAY** — otherwise the credits are
useless during the sprint.

For a 60–72h window, the lower-risk stack is **Kaggle (30 free GPU-hrs/week, T4×2 or
P100) + Colab**, with AWS as overflow capacity.

---

## 4. Setup Checklist — Do Today

### Team & access
- [ ] All 3–4 members visible on the Unstop team page
- [ ] Leader logs into the dashboard and **locates the submission page before 9 AM**
- [ ] Shared GitHub repo created
- [ ] Shared drive for data + a single chat channel
- [ ] Roles assigned: **data/IO engineer · text modeler · vision modeler · ensembler + writer**

### Compute (priority order)
- [ ] Kaggle accounts **phone-verified** (unlocks GPU + internet); weekly quota unspent
- [ ] Colab access confirmed
- [ ] Local GPU inventory (who has what VRAM)
- [ ] AWS account + credits claimed
- [ ] **AWS G-instance quota increase requested** (do this first, it has the longest lead time)

### Environment — build tonight, not tomorrow
```
python 3.11
torch (+cuda), transformers, sentence-transformers, accelerate, peft
timm, open_clip_torch, pillow
aiohttp / requests  (parallel image downloads)
polars, pandas, numpy, scikit-learn
lightgbm, xgboost, catboost
faiss-cpu, tqdm, matplotlib
```

### Pre-download model weights ← highest-leverage prep
HuggingFace downloads mid-sprint are pure dead time. Cache locally **now**:
- [ ] A CLIP or SigLIP checkpoint
- [ ] A small VLM (Qwen2.5-VL 3B or 7B)
- [ ] A text embedder (ModernBERT, or e5 / bge)
- [ ] A gradient-boosting baseline needs nothing — good, that's your hour-1 submission

### Pre-write a template repo
- [ ] Async image downloader — retries, timeouts, **resize-on-save**
- [ ] Train / predict skeleton
- [ ] Submission-file writer
- [ ] Validation split that **mirrors the leaderboard**
- [ ] Experiment log (CSV or wandb) — so you can tell what actually helped

### Read tonight
- [ ] 2024 winner — https://github.com/KhadgaA/Amazon-ML-Challenge
- [ ] 2025 top-0.3% — https://github.com/NeelDevenShah/Amazon-ML-Challenge-2025
- [ ] Prep guide — https://github.com/KushalVijay/Amazon-ML-Challenge-Guide

---

## 5. Sprint Rhythm

**Hour 0–3** — EDA + submit a *deliberately stupid* baseline (global median / majority
class) to prove the submission pipeline end-to-end.
> Teams lose this competition to a malformed CSV, not to a bad model.

**Day 1** — Text-only model. Establish a solid CV split you trust.

**Day 2** — Add images. Ensemble **diverse model families**, not five variants of one.

**Day 3** — Freeze **6 hours before** the deadline. Write the 1–2 page doc.
Trust your CV over the public leaderboard — the private leaderboard is what counts.

### Lessons from last year's top-0.3% team (rank 80 / ~23,000)
1. *Data understanding before modeling* drove **80% of the gains**
2. Rigorous validation strategy was critical at 75k test rows
3. **Team synergy > individual skill** during the sprint
4. Model **diversity** beat any single strong model
5. Preprocessing and feature engineering beat model sophistication

---

## 6. Sources

- [Unstop listing](https://unstop.com/hackathons/amazon-ml-challenge-2026-amazon-1743604)
- [Internshala mirror](https://internshala.com/competitions/amazon-ml-challenge-2026-win-%E2%82%B9225000/)
- [AWS Builder Center post](https://builder.aws.com/post/3JBxLF0Pm7NZGZZaf24EnivwNcf_p/amazon-ml-challenge-2026-is-live-on-unstop)
- [2024 winning solution](https://github.com/KhadgaA/Amazon-ML-Challenge)
- [2025 top solution](https://github.com/NeelDevenShah/Amazon-ML-Challenge-2025)
- [Community prep guide](https://github.com/KushalVijay/Amazon-ML-Challenge-Guide)
