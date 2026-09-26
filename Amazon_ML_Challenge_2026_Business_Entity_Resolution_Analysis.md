# Amazon ML Challenge 2026 — Business Entity Resolution

## 0. Executive summary

The challenge video defines a **business entity resolution / record linkage** problem.

The task is to determine which records from **three independent data sources** refer to the **same real-world business**, even though the sources have **no shared identifier**.

The records shown in the video contain:

- Business name
- Business address

The video explicitly says that phone and email may exist in reality, but **this challenge uses name and address only**.

The core pipeline is therefore:

```text
Raw records from Source 1, Source 2, Source 3
                    |
                    v
          text/address normalization
                    |
                    v
              BLOCKING
      cheap high-recall candidate generation
                    |
                    v
       candidate pairs S1 <-> S2/S3
                    |
                    v
       pairwise similarity / ML model
                    |
                    v
        high-precision match decision
                    |
                    v
      aggregate matches per Source-1 ID
                    |
                    v
          matching_results.tsv
```

The most important strategic conclusion is:

> **Do not build this as an LLM chatbot problem.**

A large generative VLM such as Qwen-VL is not the natural core model here. The strongest practical approach is likely to combine deterministic normalization, character/token/address similarity features, blocking, a tabular classifier/ranker, and optionally a small pretrained text encoder as an additional feature source.

Because the leaderboard uses **macro F0.5**, precision is weighted more strongly than recall. The video explicitly states that a false merge costs about twice as much as a miss and advises not merging when uncertain.

---

# 1. What the video says the problem is

## 1.1 Title and objective

The video is titled:

> **Amazon ML Challenge 2026 — Business Entity Resolution**

Its central definition is:

> Determine which business records, arriving from three independent sources with no shared identifier, describe the same real-world business.

This is the standard **entity resolution / record linkage / entity matching** family of problems.

The important idea is that a business does not have to appear identically in every source.

For example, the video shows the same entity approximately as:

```text
Source 1:
Acme Robotics Inc
500 Market St, San Jose

Source 2:
Acme Robotics Inc
500 Market Street, San Jose

Source 3:
Acme Robotics
Nr City Hall, San Jose
```

A human can recognize these as related despite spelling and address-format differences. The model must do this algorithmically.

---

# 2. Why this is an entity-resolution problem rather than ordinary classification

Ordinary supervised classification usually looks like:

```text
row -> class label
```

Here the hidden label is effectively relational:

```text
Source-1 entity -> zero, one, or multiple matching records in Source 2 and Source 3
```

The model must answer questions such as:

```text
Does S1-A match S2-X?
Does S1-A match S2-Y?
Does S1-A match S3-P?
Does S1-A match S3-Q?
```

Then these pairwise decisions are converted into a **match set** for each Source-1 entity.

This distinction matters because the final output is not simply:

```text
S1-A -> class 7
```

It is something like:

```text
S1-A -> S2-X,S3-P
```

or, when there is no match:

```text
S1-B -> <empty list>
```

---

# 3. The three-source structure

The video demonstrates three separate source systems:

```text
SOURCE 1        SOURCE 2        SOURCE 3
---------       ---------       ---------
S1-...          S2-...          S3-...
S1-...          S2-...          S3-...
S1-...          S2-...          S3-...
...
```

Each source has its own IDs.

There is **no common business identifier** across the three datasets.

Therefore:

```text
S1-732914
```

cannot simply be joined using an ID to:

```text
S2-118820
S3-905477
```

Instead, the algorithm must infer those relationships from the available fields.

---

# 4. The fields actually available

The video makes an important restriction explicit.

A real business record could have:

- name
- address
- phone
- email
- tax ID
- website
- registration ID
- etc.

But this challenge uses **name and address only**.

This is crucial.

Do not design a solution around phone numbers, email addresses, geocoding, Google Maps, business directories, or external company databases.

The challenge explicitly prohibits external lookup.

The earlier challenge brief also states:

> External databases, APIs, and geocoding services are strictly prohibited. Use only the provided data.

Therefore the solution must learn the matching structure **inside the supplied datasets**.

---

# 5. Blocking: the first major stage

The video introduces **blocking** before matching.

This is probably the most important architectural concept in the entire challenge.

## 5.1 Why blocking is necessary

Suppose there are:

```text
100,000 Source-1 records
100,000 Source-2 records
100,000 Source-3 records
```

A naive all-pairs comparison would require roughly:

```text
S1 x S2 = 10^10 pairs
S1 x S3 = 10^10 pairs
```

That is tens of billions of comparisons.

Most of those comparisons are obviously impossible matches.

Instead, blocking first creates a smaller candidate set.

Conceptually:

```text
100,000 records
      |
      v
cheap blocking key
      |
      v
100,000 records -> perhaps 10-100 plausible candidates per S1 record
```

The expensive matching model is then run only on these candidates.

---

# 6. Why blocking recall is a hard ceiling

This is explicitly emphasized in the video:

> Blocking sets your recall ceiling. You cannot match a record you never consider.

Mathematically, if a true pair is excluded by blocking:

```text
true pair -> not in candidate set -> impossible to recover later
```

Therefore:

```text
Recall_final <= Recall_blocking
```

This means a brilliant classifier cannot compensate for an overly aggressive blocking rule.

### Example

Suppose the true matches are:

```text
1000 true pairs
```

Your blocking procedure keeps only:

```text
950 true pairs
```

Then the maximum achievable final recall is:

```text
950 / 1000 = 95%
```

No downstream neural network can recover the 50 missing pairs because they were never presented to it.

This is why the correct sequence is:

```text
1. Build high-recall blocking
2. Measure blocking recall
3. Then optimize pairwise precision
```

---

# 7. The video’s blocking example

The video shows a block built using **name + address**.

Example records include:

```text
Source 1
S1-732914 | Acme Robotics Inc | 500 Market St, San Jose

Source 2
S2-118820 | Acme Robotics Inc | 500 Market Street, San Jose
S2-540221 | Acme Robotix      | 12 Elm Rd, San Jose

Source 3
S3-905477 | Acme Robotics     | Nr City Hall, San Jose
S3-063118 | Acme Bakery        | 500 Market St, San Jose
```

The resulting candidate block can contain both true and false candidates.

For example:

```text
S1-732914 -> S2-118820
S1-732914 -> S2-540221
S1-732914 -> S3-905477
S1-732914 -> S3-063118
```

This is intentional.

Blocking is not supposed to perfectly solve matching. It is supposed to produce a manageable **shortlist** while preserving true matches.

The matching model then filters the shortlist.

---

# 8. Candidate generation should probably use multiple blocking rules

A single blocking key is dangerous.

For example:

```text
normalized_name_prefix + city
```

can miss records when the business name is substantially altered.

A better architecture is the union of several cheap blocking rules.

## Candidate rule A — normalized name

Create a normalized form:

```text
Acme Robotics Inc.
        ->
ACME ROBOTICS
```

Then use exact/prefix/token-based buckets.

## Candidate rule B — address components

Extract useful address structure:

```text
house number
street tokens
city
state/region
postal code, if present
```

Then form keys such as:

```text
house_number + city
street_token + city
postal_code
```

## Candidate rule C — character n-gram similarity

Use TF-IDF over character n-grams, for example:

```text
"robotics"
```

produces pieces such as:

```text
rob
obo
bot
oti
tic
ics
```

This is robust to:

```text
Robotix
Robotics
Robotic
```

and many spelling/formatting changes.

## Candidate rule D — token overlap

For example:

```text
Acme Robotics International
```

and

```text
Acme Robotics Intl
```

share many informative tokens.

## Candidate rule E — semantic embeddings

A pretrained text encoder can be used to retrieve semantically similar records.

This should be an **additional blocking source**, not the only one.

For example:

```text
record -> embedding vector
                 |
                 v
        nearest-neighbour search
                 |
                 v
             candidates
```

The union of several candidate generators is safer:

```text
Candidates =
    exact-rule candidates
  U address-rule candidates
  U character-similarity candidates
  U embedding candidates
```

This is one of the best ways to protect blocking recall.

---

# 9. Normalization is likely to be extremely important

Before using a model, convert the raw strings into several normalized representations.

Do NOT throw away the original strings.

Keep both:

```text
raw value
normalized value
structured components
```

## 9.1 Name normalization

Potential operations:

- lowercase
- Unicode normalization
- punctuation removal
- whitespace normalization
- common abbreviation normalization
- legal suffix normalization
- token sorting when appropriate
- repeated-character handling when justified

Example:

```text
"Acme Robotics, Inc."
```

could produce:

```text
acme robotics inc
acme robotics
```

The suffix removal should be conservative. `Inc`, `Ltd`, `LLC`, `Corp`, etc. often carry little identity information, but sometimes legal suffix patterns matter by region.

## 9.2 Address normalization

Potential operations:

```text
Street -> St
Road -> Rd
Avenue -> Ave
Number formatting
Punctuation normalization
Whitespace normalization
```

Also extract structured pieces where possible:

```text
house number
street name
street type
city
region/state
postal code
```

Do not assume that every address has a clean postal code or that every source follows the same ordering.

---

# 10. Preserve multiple address representations

An important engineering mistake would be:

```text
raw address -> one normalized string -> discard raw data
```

Instead create multiple views:

```text
address_raw
address_normalized
address_tokens
address_house_number
address_street
address_city
address_region
address_postal
```

This lets the model distinguish:

```text
500 Market St, San Jose
```

from

```text
500 Market Street, San Jose
```

while still detecting that they are essentially equivalent.

---

# 11. Pairwise feature engineering

For each candidate pair:

```text
(Source1 record, Source2/Source3 record)
```

create a feature vector.

A useful conceptual feature vector is:

```text
[name_exact]
[name_normalized_exact]
[name_token_jaccard]
[name_char_similarity]
[name_edit_similarity]

[address_exact]
[address_normalized_exact]
[address_token_jaccard]
[address_char_similarity]
[address_edit_similarity]

[house_number_equal]
[city_equal]
[region_equal]
[postal_equal]

[name_length_difference]
[address_length_difference]

[blocking_rule_that_generated_pair]
```

These features can be extremely effective because business matching is fundamentally a structured similarity problem.

---

# 12. String similarity measures worth trying

## Exact match

```text
name_a == name_b
```

This is a very strong feature when normalization has already been applied.

## Levenshtein / edit similarity

Useful for typos such as:

```text
Robotics
Robotix
```

## Jaro-Winkler

Often useful for short names and small spelling perturbations.

## Jaccard token similarity

For token sets:

```text
A = {acme, robotics}
B = {acme, robotics, intl}
```

Jaccard similarity is:

```text
|A ∩ B| / |A ∪ B|
```

## Character n-gram cosine similarity

Very useful because entity-resolution errors are often character-level:

```text
Ltd
Limited
Ldt
```

etc.

---

# 13. Address similarity should be decomposed

A single address similarity score is not enough.

Consider:

```text
12 Elm Road, San Jose
```

versus

```text
12 Elm Rd., San Jose
```

This should be very high similarity.

But:

```text
12 Elm Road, San Jose
```

versus

```text
14 Elm Road, San Jose
```

is substantially different, despite strong token overlap.

Therefore separate:

```text
house number match
street match
city match
region match
postal match
```

A house-number mismatch may deserve a substantial penalty.

The exact weighting should be learned/validated, not guessed permanently.

---

# 14. Region-specific patterns

The video explicitly tells participants to:

> Account for region-specific patterns in both names and addresses.

This means the data may contain source/region conventions such as:

```text
Street / St / Str
Road / Rd
Building abbreviations
Regional address ordering
Legal entity suffixes
```

and potentially different conventions across geographies.

Do not apply one blindly universal normalization rule.

Instead, consider source-aware or region-aware features where the data supports them.

For example:

```text
source = S1
region = inferred from address tokens
```

can influence how the address parser interprets the text.

The key principle is:

> Normalize aggressively enough to remove formatting noise, but not so aggressively that distinct businesses collapse into the same representation.

---

# 15. What model should actually do the matching?

For this challenge, a tabular pairwise model is a very strong candidate.

A practical first model is:

```text
CatBoost / LightGBM / XGBoost
```

Input:

```text
pairwise similarity features
```

Output:

```text
P(pair is a true match)
```

Why this is attractive:

- Fast to train
- Fast to iterate
- Handles nonlinear combinations
- Works well with mixed numerical/categorical features
- Easy to analyze feature importance
- Much easier to tune during a short hackathon than a large generative LLM

---

# 16. Where a pretrained language model can help

A pretrained encoder can still be useful.

But use it as a **feature extractor / similarity model**, not as a generative answer engine.

A useful architecture is:

```text
business name + address
          |
          v
 pretrained encoder
          |
          v
      embedding
          |
          +---- cosine similarity -----+
                                      |
string/address features -------------+----> CatBoost/LightGBM
                                      |
source/region features --------------+
```

This gives the boosting model a semantic feature in addition to deterministic string features.

---

# 17. Suggested pretrained models for this problem

## Primary lightweight option: MiniLM

A small Sentence-Transformers model such as:

```text
sentence-transformers/all-MiniLM-L6-v2
```

is a sensible baseline for semantic embeddings.

It is small, fast, and easy to run locally.

## Stronger general embedding option

A BGE-family embedding model can be used as a second experiment.

For example:

```text
BAAI/bge-base-en-v1.5
```

or a smaller BGE model when latency matters.

## Multilingual option

If the data contains multiple languages/scripts, a multilingual sentence-transformer becomes more relevant.

## Important warning

Do **not** assume embedding similarity will beat exact/character/address engineering.

For business entity resolution, two records may refer to the same entity even when their semantics are trivial:

```text
Acme Robotics
Acme Robotics Ltd
```

The important difference is often spelling, abbreviation, token correspondence, and address structure rather than high-level language semantics.

Therefore embeddings should be an additional signal.

---

# 18. Do we need a large LLM or VLM?

Probably not.

The video gives no image field and no generative-language requirement.

The problem is:

```text
record linkage over short structured text
```

not:

```text
image understanding
```

Therefore models such as:

```text
Qwen-VL
LLaMA
large instruction-tuned chat models
```

should not be the center of the solution.

A huge model would introduce:

- slower inference
- high VRAM usage
- more complicated batching
- harder reproducibility
- little obvious benefit over strong pairwise features

A small encoder plus a strong tabular matcher is much more aligned with the task.

---

# 19. Candidate scoring versus final match decision

Do not confuse these two stages.

## Stage A: candidate generation

Question:

> Which records are worth considering?

Goal:

```text
high recall
```

False positives here are acceptable because the next stage filters them.

## Stage B: matching

Question:

> Is this candidate pair actually the same business?

Goal:

```text
high precision
```

because the final metric heavily rewards precision.

Therefore:

```text
Blocking threshold -> relatively permissive
Matching threshold -> relatively conservative
```

---

# 20. The final output is a set-valued prediction

For every Source-1 entity, the model must output its complete matching set.

Example candidate set:

```text
S1-732914
    |
    +-- S2-118820
    +-- S2-540221
    +-- S3-905477
    +-- S3-063118
```

The final model may decide:

```text
TRUE:
S2-118820
S3-905477

FALSE:
S2-540221
S3-063118
```

Therefore the submission becomes:

```text
S1-732914    S2-118820,S3-905477
```

The video describes this as:

```text
one row per Source 1 entity
```

---

# 21. Singleton handling is unusually important

The video gives a very strong rule:

> Singletons are Source 1 entities with no match in Source 2 or 3.

For a singleton:

```text
predict empty list
```

and you receive a full score for that entity.

But if you invent a match:

```text
predict any match
```

and you receive zero for that entity.

This dramatically changes the optimal decision behavior.

When evidence is weak:

```text
empty list
```

may be much safer than a speculative merge.

---

# 22. The evaluation metric: macro F0.5

The video says the competition is scored on:

```text
macro F0.5
```

with:

```text
F0.5 = (1.25 * Precision * Recall)
       / (0.25 * Precision + Recall)
```

This is the standard F-beta form for beta = 0.5.

---

# 23. Why F0.5 prefers precision

For ordinary F1:

```text
beta = 1
```

precision and recall receive symmetric weighting.

For F0.5:

```text
beta = 0.5
```

precision receives more importance than recall.

That matches the video’s plain-language statement:

> A false merge costs about twice as much as a miss, so when unsure, do not merge.

This is one of the most important competition-specific details.

---

# 24. Threshold selection must be optimized for the actual metric

Suppose the classifier outputs:

```text
P(match)
```

A generic ML practitioner may choose:

```text
threshold = 0.5
```

That is arbitrary and potentially bad.

Instead test a range:

```text
0.10
0.15
0.20
...
0.90
0.95
```

and select the threshold according to the **validation version of the competition metric**.

Because the metric favors precision, the best threshold may be relatively conservative.

But do not assume this in advance. Measure it.

---

# 25. Match-level versus entity-level evaluation

The video describes the ground truth as one row per Source-1 entity containing its entire match set.

Therefore evaluation is not simply ordinary independent binary classification.

Conceptually:

```text
Ground truth:
S1-A -> {S2-1, S3-8}

Prediction:
S1-A -> {S2-1, S3-8, S3-9}
```

The extra S3-9 creates a false merge and hurts precision.

Another prediction:

```text
S1-A -> {S2-1}
```

misses S3-8 and hurts recall.

Since F0.5 emphasizes precision, these are not equally costly.

---

# 26. A better final architecture: multi-signal pair scorer

Recommended design:

```text
                  SOURCE 1 RECORD
                        |
                +-------+-------+
                |               |
              NAME           ADDRESS
                |               |
                v               v
         normalization    normalization
                |               |
                +-------+-------+
                        |
                        v
               structured features
                        |
          +-------------+-------------+
          |             |             |
       exact/        string        embedding
       token         similarity     similarity
      features        features        |
          |             |             |
          +-------------+-------------+
                        |
                        v
                 CatBoost/LightGBM
                        |
                        v
                   P(match)
                        |
              conservative threshold
                        |
                        v
                    final edges
```

Then aggregate edges by Source-1 ID.

---

# 27. Add source-pair features

The model should know whether a candidate is:

```text
S1 <-> S2
```

or:

```text
S1 <-> S3
```

because different data sources may have different noise patterns.

Useful features:

```text
source_pair
name_length_distribution_by_source
address formatting style by source
abbreviation frequency by source
```

The model can therefore learn different decision boundaries:

```text
P(match | S1,S2,features)
```

versus:

```text
P(match | S1,S3,features)
```

---

# 28. Exploit the three-source structure carefully

A potentially powerful additional signal is **tri-source consistency**.

Suppose:

```text
S1-A ~ S2-B
S1-A ~ S3-C
```

and independently:

```text
S2-B ~ S3-C
```

then the three records form a coherent cluster.

This can increase confidence.

However, do not blindly use transitivity:

```text
A matches B
B matches C
therefore A matches C
```

because noisy entity-resolution systems can create false chains.

A safer approach is:

1. score pairwise edges
2. keep only high-confidence edges
3. use cross-source consistency as an additional feature/constraint
4. audit connected components

The safest initial implementation is still pairwise S1-S2 and S1-S3 matching.

---

# 29. Important distinction: blocking can contain intentional false positives

Do not try to make blocking itself perfect.

This is a common mistake.

Bad idea:

```text
blocking rule = exact normalized name + exact normalized address
```

This may have excellent precision but terrible recall.

Better:

```text
blocking = broad union of cheap rules
matching = strict learned decision
```

Think of blocking as a **recall-preserving funnel**.

---

# 30. Validation strategy

Validation is likely to be one of the hardest parts of the challenge because the leaderboard metric is based on complete match sets.

Build a local evaluator that exactly reproduces the output semantics.

For every Source-1 entity:

```text
truth_set = true matching IDs
pred_set  = predicted matching IDs
```

then compute the same precision/recall/F0.5 logic used by the challenge.

Also report:

```text
precision
recall
F0.5
singleton accuracy / handling
number of predicted matches
candidate recall
```

Never optimize only ordinary pairwise accuracy.

A model can have very high pairwise accuracy simply because most candidate pairs are non-matches.

---

# 31. Pairwise class imbalance

Entity matching usually has a huge number of non-match pairs.

Example:

```text
100,000 candidate pairs

true matches = 2,000
non-matches = 98,000
```

A classifier that predicts everything as non-match gets:

```text
98% accuracy
```

but is useless.

Therefore use:

- precision/recall
- F0.5
- confusion matrix
- PR curve
- threshold sweep

rather than accuracy.

---

# 32. Negative-pair construction

If the official training data provides pair-level labels indirectly through the ground-truth match sets, construct negatives from the blocked candidate pairs.

A good training dataset is:

```text
positive candidates = known true matches
negative candidates = blocked pairs known not to match
```

Do not use arbitrary global random negatives only.

The model needs **hard negatives** that look similar to the positives.

Examples:

```text
Acme Robotics
Acme Robotix
Acme Robotics LLC
Acme Robotics Holdings
```

and:

```text
500 Market St, San Jose
500 Market Street, San Jose
500 Market Ave, San Jose
500 Market St, San Francisco
```

These are much more useful negatives than completely unrelated businesses.

---

# 33. Hard-negative mining

After training an initial model:

```text
model v1
   |
   v
find high-scoring false positives
   |
   v
add them as hard negatives
   |
   v
retrain model
```

This can directly improve precision.

For this challenge, this is potentially more valuable than replacing a small model with a much larger neural network.

---

# 34. Model ensemble strategy

A reasonable ensemble is not five copies of the same algorithm.

Use diverse signals:

```text
Model A: deterministic similarity score
Model B: CatBoost on engineered features
Model C: embedding cosine similarity
Model D: character TF-IDF retrieval score
```

Then either:

```text
combine features -> one final classifier
```

or:

```text
combine calibrated scores
```

The strongest architecture should be selected empirically using local validation.

---

# 35. Deterministic baseline that should be built first

Before any ML model, create a rules baseline.

Example:

```text
match if normalized_name_exact
AND normalized_address_exact
```

Then a slightly more flexible baseline:

```text
high name similarity
AND high address similarity
AND house number equal
```

This baseline gives you:

- sanity check
- debugging reference
- minimum viable submission
- understanding of data quality

If the rule baseline performs surprisingly well, do not discard it.

It may provide excellent high-precision matches that can be combined with ML.

---

# 36. Suggested local model stack

## Core packages

```text
Python 3.11
pandas / polars
numpy
scikit-learn
rapidfuzz
catboost
lightgbm
xgboost
sentence-transformers
torch
faiss-cpu or faiss-gpu if available
```

## Pretrained encoder

Start with a **small sentence embedding model** rather than a large LLM.

Suggested first experiment:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Then test a stronger BGE model if time permits.

The encoder should be used to produce embeddings for:

```text
name
address
name + address
```

Do not necessarily concatenate the entire raw record immediately. Keep separate embeddings because they answer different questions.

---

# 37. Why character models may beat an LLM here

Consider:

```text
"Acme Robotics Inc"
"ACME ROBOTICS, INC."
"Acme Robotix"
```

These are primarily **orthographic variations**.

A character 3-gram/4-gram model directly observes these local similarities.

An LLM's main strength is language understanding and generation. That is not the bottleneck here.

Similarly, for addresses:

```text
500 Market St
500 Market Street
```

simple normalization almost solves the problem.

The difficult cases are more likely to be:

```text
partial addresses
abbreviations
misspellings
reordered tokens
shared buildings
branch locations
regional formatting
```

These are exactly the cases where engineered similarity features are useful.

---

# 38. The strongest likely candidate-ranking approach

One useful architecture is:

```text
1. Block
2. Compute candidate features
3. CatBoost score
4. Sort candidates per Source-1 entity
5. Apply calibrated threshold
6. Keep only high-confidence matches
7. Build final match list
```

For each S1 record:

```text
candidate    probability
--------     -----------
S2-A          0.998
S3-X          0.991
S2-B          0.42
S3-Y          0.08
```

Then a conservative threshold might yield:

```text
S1 -> S2-A,S3-X
```

The threshold must be tuned against local F0.5.

---

# 39. Do not automatically force one match per Source-1 entity

The video explicitly allows:

```text
zero matches
one match
multiple matches
```

Therefore the model must not use:

```text
argmax candidate
```

by default.

Forcing exactly one match is especially bad because singletons exist.

Instead:

```text
keep all candidates above the final decision threshold
```

with optional consistency constraints.

---

# 40. But also do not accept every candidate above a weak threshold

A different failure mode is:

```text
S1-A -> five S2 records
```

just because all have moderately similar names.

The F0.5 objective strongly discourages this type of false merging.

Thus threshold selection is central.

---

# 41. Submission structure shown in the video

## During the challenge

The video specifies:

```text
matching_results.tsv
```

This is the file scored on the leaderboard.

It contains:

```text
one row per Source 1 entity
```

An empty list means no match.

## Final ZIP package

The video shows the final package containing:

```text
output/matching_results.tsv
output/candidate_pairs.tsv
code/...
Documentation_template.md
```

The candidate-pairs file is the **audited blocking set**.

The runnable code must reproduce the pipeline.

A utility is also shown:

```text
utils/validate_submission.py
```

Run this before submission to catch formatting errors.

---

# 42. Why candidate_pairs.tsv matters

This is more than an intermediate file.

It records the candidate set generated by blocking.

This makes the pipeline auditable:

```text
raw data
   |
   v
blocking
   |
   v
candidate_pairs.tsv
   |
   v
matching model
   |
   v
matching_results.tsv
```

This means your blocking stage should be deterministic, reproducible, and saved explicitly.

---

# 43. Recommended repository structure

```text
amazon-ml-2026/
│
├── data/
│   ├── train/
│   └── test/
│
├── src/
│   ├── io.py
│   ├── normalize.py
│   ├── parse_address.py
│   ├── blocking.py
│   ├── pair_features.py
│   ├── embeddings.py
│   ├── train_matcher.py
│   ├── predict.py
│   ├── postprocess.py
│   └── evaluate.py
│
├── models/
│
├── output/
│   ├── candidate_pairs.tsv
│   └── matching_results.tsv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_blocking.ipynb
│   └── 03_model.ipynb
│
├── utils/
│   └── validate_submission.py
│
├── requirements.txt
├── README.md
└── Documentation_template.md
```

---

# 44. Recommended experiment order

Do not start with a transformer.

Use this sequence:

## Experiment 0 — inspect data

Determine:

```text
row counts per source
missing names
missing addresses
name length distributions
address length distributions
duplicate rates
source-specific formatting
```

## Experiment 1 — exact normalization

Measure:

```text
exact normalized name
exact normalized address
exact both
```

## Experiment 2 — blocking

Add:

```text
name blocks
address blocks
character similarity retrieval
```

Measure candidate recall.

## Experiment 3 — pairwise engineered features

Train:

```text
CatBoost
```

## Experiment 4 — hard negatives

Mine false positives and retrain.

## Experiment 5 — embeddings

Add sentence embeddings as features.

## Experiment 6 — threshold tuning

Optimize directly for F0.5.

## Experiment 7 — cross-source consistency

Add carefully validated consistency features.

---

# 45. EDA questions that can reveal the hidden structure

The first serious analysis should answer:

### Names

```text
How often are names exactly duplicated?
How often do legal suffixes appear?
How many tokens are typical?
What kinds of spelling errors occur?
Do sources use abbreviations systematically?
```

### Addresses

```text
How often is an address missing?
How often are city/region/postal tokens present?
Do sources abbreviate Street/Road/Avenue?
Are addresses reordered between sources?
Are there shared addresses containing multiple businesses?
```

### Sources

```text
Does S1 have a different formatting convention from S2?
Does S3 use abbreviations more aggressively?
Do some source pairs have much higher noise?
```

These observations should drive feature engineering.

---

# 46. Shared addresses are a trap

An important issue is that multiple businesses may legitimately share an address.

For example:

```text
500 Market St, San Jose
```

may correspond to multiple businesses.

Therefore:

```text
address exact match
```

must not automatically imply:

```text
same business
```

The name must remain a major signal.

Similarly, the same business may appear at branch locations over time, but the challenge's hidden data determines what counts as a match. Do not invent business-history assumptions from outside data.

---

# 47. The inverse trap: name collisions

Generic names can also collide.

For example:

```text
ABC Trading
ABC Trading Ltd
ABC Trading Company
```

may represent different entities.

Therefore:

```text
name similarity high
```

is not sufficient by itself.

Name and address should be treated jointly.

---

# 48. Useful feature interactions

Examples:

```text
high name similarity + same house number + same city
```

is strong.

But:

```text
high name similarity + different city + different house number
```

should be much weaker.

This is precisely where tree-based models are useful because they can learn nonlinear interactions.

---

# 49. Calibrate confidence by candidate rank

Another useful analysis is to examine:

```text
best candidate score
second-best candidate score
score margin
```

Example:

```text
best = 0.997
second = 0.21
margin = 0.787
```

versus:

```text
best = 0.82
second = 0.80
margin = 0.02
```

The second case is ambiguous.

Because false merges are expensive, the score margin can be a useful feature or conservative decision criterion.

---

# 50. Possible final decision logic

A robust system could conceptually use:

```text
if pair_score >= T_high:
    accept
elif pair_score >= T_low and strong_structural_evidence:
    accept
else:
    reject
```

But avoid hand-building dozens of arbitrary rules after ML unless validation demonstrates that they help.

A clean first version is better:

```text
probability >= T
```

with threshold optimized on validation.

---

# 51. What the earlier challenge brief got wrong

The earlier prep brief was based on uncertainty about the challenge statement and predicted a likely product-catalog task involving text + images and possibly price prediction.

That assumption is now superseded by the supplied challenge video.

The actual video clearly shows:

```text
Business Entity Resolution
```

and the fields are name + address.

Therefore the earlier plan involving:

```text
CLIP / SigLIP
Qwen-VL
image downloading
image embeddings
product price regression
```

should **not** be the primary preparation for this actual task.

The infrastructure advice in the brief about reproducibility, validation, submission checking, and fast iteration remains useful, but the modeling target has changed.

---

# 52. Practical model recommendation for your local GPU

Your GPU is useful, but this problem may not require heavy GPU compute.

Use the GPU primarily for:

```text
sentence embeddings
ANN retrieval if needed
larger transformer experiments
```

The CPU can handle a lot of:

```text
normalization
RapidFuzz/string matching
TF-IDF
feature construction
CatBoost/LightGBM on moderate datasets
```

Therefore do not optimize the project around "making the GPU run at 100%".

Optimize around:

```text
candidate recall
precision
validation
iteration speed
```

---

# 53. Recommended pretrained-model setup

Install a small encoder first.

A practical initial choice:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Then optionally cache one stronger embedding model for experimentation.

You do **not** need Qwen-VL for the problem shown in the video.

If you later discover that the actual dataset contains unexpected free-form multilingual text, add a multilingual encoder only when the data justifies it.

---

# 54. Minimal end-to-end pseudo-code

```python
# 1. Load source tables
s1 = load_source1()
s2 = load_source2()
s3 = load_source3()

# 2. Normalize text
for df in [s1, s2, s3]:
    df["name_norm"] = normalize_name(df["name"])
    df["address_norm"] = normalize_address(df["address"])
    df = add_address_components(df)

# 3. Generate candidates
cand12 = block(s1, s2)
cand13 = block(s1, s3)

# 4. Save audited candidate set
save_candidate_pairs(cand12, cand13)

# 5. Build pair features
X12 = make_pair_features(cand12)
X13 = make_pair_features(cand13)

# 6. Train matcher on known training relationships
model = CatBoostClassifier(...)
model.fit(X_train, y_train)

# 7. Score test candidates
p12 = model.predict_proba(X12)[:, 1]
p13 = model.predict_proba(X13)[:, 1]

# 8. Conservative threshold tuned on validation
m12 = cand12[p12 >= threshold]
m13 = cand13[p13 >= threshold]

# 9. Aggregate by Source-1 ID
results = aggregate_matches(m12, m13)

# 10. Ensure every Source-1 ID is represented
results = add_empty_lists_for_singletons(results, s1)

# 11. Write exact required format
write_matching_results(results)
```

---

# 55. Three levels of solution sophistication

## Level 1 — strong baseline

```text
Normalization
+ exact matching
+ RapidFuzz
+ blocking
+ conservative rules
```

This should be built first.

## Level 2 — competition-grade

```text
Level 1
+ character TF-IDF
+ structured address features
+ CatBoost
+ hard-negative mining
+ threshold optimization
```

This is likely the main target.

## Level 3 — advanced

```text
Level 2
+ pretrained embeddings
+ ANN retrieval
+ score ensemble
+ source-specific models
+ tri-source consistency features
+ calibrated uncertainty
```

Only pursue this after Level 2 is working.

---

# 56. What I would NOT spend hackathon time on initially

Avoid these as first moves:

```text
huge LLM
VLM
fine-tuning a multi-billion parameter model
external business APIs
Google Maps/geocoding
web search for each business
complex graph neural networks
```

The challenge explicitly prohibits external lookups, and the data shown is fundamentally a structured text-matching problem.

---

# 57. The key optimization hierarchy

For this specific challenge, prioritize approximately in this order:

```text
1. Correctly understand the data
2. Build reliable normalization
3. Build high-recall blocking
4. Build a precise pairwise matcher
5. Validate the exact competition metric
6. Tune the decision threshold
7. Mine hard negatives
8. Add embeddings
9. Add cross-source consistency
10. Only then consider more complex models
```

The exact ranking of later items must be determined experimentally, but the first several stages are structural rather than optional.

---

# 58. Recommended first-day workflow

## Hour 0–1: data inspection

Produce:

```text
row counts
schema
missingness
duplicate analysis
sample records
source differences
```

## Hour 1–2: deterministic baseline

Implement:

```text
normalization
exact matching
simple fuzzy matching
```

## Hour 2–4: blocking

Implement multiple blocks and measure:

```text
candidate count
candidate recall
average candidates per S1
max candidates per S1
```

## Hour 4–8: pairwise model

Build:

```text
feature generator
CatBoost
validation evaluator
threshold sweep
```

This should produce the first serious submission.

---

# 59. What to log for every experiment

Use a simple CSV/JSON log.

Record:

```text
experiment_id
normalization_version
blocking_rules
candidate_count
blocking_recall
model
features
threshold
precision
recall
F0.5
number_of_predicted_matches
singleton_fraction
runtime
```

Without this, a hackathon quickly turns into:

```text
"I think experiment 7 was better than experiment 5..."
```

Do not rely on memory.

---

# 60. Most important debugging questions

When performance is bad, identify which layer failed.

### Low recall

Probably:

```text
blocking failure
normalization failure
candidate generation too aggressive
```

### High candidate recall but low precision

Probably:

```text
matching model too weak
features insufficient
threshold too low
shared-address/name collisions
```

### Good local score but bad leaderboard score

Investigate:

```text
validation mismatch
leakage
source-distribution mismatch
output formatting
threshold overfitting
```

### Many false singleton predictions

Investigate:

```text
blocking misses
threshold too high
poor address parsing
```

---

# 61. The most important conceptual picture

Think of the challenge as a graph.

```text
                 same real-world entity
                         |
          +--------------+--------------+
          |                             |
       Source 1                      Source 2
       S1-A ------------------------- S2-X
          |                             |
          |                             |
          +--------------------------- S3-P
                                      Source 3
```

Your algorithm has to recover the hidden edges:

```text
S1-A <-> S2-X
S1-A <-> S3-P
```

without being given the edge labels at test time.

Blocking proposes possible edges.

The matching model decides which proposed edges are real.

The final submission lists all accepted edges for every Source-1 node.

That is the cleanest mental model for the entire competition.

---

# 62. Final recommendation

For the actual problem shown in the video, I would prepare this stack:

```text
Python 3.11

Data / feature layer:
pandas or polars
numpy
rapidfuzz
scikit-learn

Matching layer:
CatBoost
LightGBM

Candidate retrieval:
character TF-IDF
RapidFuzz
FAISS if embedding retrieval becomes useful

Optional neural representation:
sentence-transformers/all-MiniLM-L6-v2
BGE-family encoder as a second experiment

Infrastructure:
tqdm
joblib
matplotlib
```

The core model should be:

```text
BLOCKING
   -> pairwise features
   -> CatBoost/LightGBM
   -> threshold tuned for F0.5
   -> singleton-aware aggregation
   -> matching_results.tsv
```

The LLM/VLM should be treated as optional, not foundational.

---

# 63. Final checklist before coding

```text
[ ] Understand Source 1 / 2 / 3 layout
[ ] Inspect exact column names
[ ] Inspect train_ground_truth.tsv
[ ] Inspect candidate_pairs expectations
[ ] Implement name normalization
[ ] Implement address normalization
[ ] Preserve raw + normalized + parsed forms
[ ] Build multiple blocking rules
[ ] Measure blocking recall
[ ] Build pairwise feature matrix
[ ] Train first CatBoost/LightGBM matcher
[ ] Build exact local F0.5 evaluator
[ ] Tune threshold
[ ] Handle empty match lists
[ ] Mine hard negatives
[ ] Add character TF-IDF
[ ] Add embedding similarity
[ ] Test source-specific behavior
[ ] Save candidate_pairs.tsv
[ ] Save matching_results.tsv
[ ] Run validate_submission.py
[ ] Package reproducible code
```

---

# 64. Bottom line

This is fundamentally a **high-precision entity matching problem**.

The winning mindset is not:

> "Which biggest pretrained LLM can I run?"

It is:

> "How do I avoid losing true matches during blocking, and how do I reject deceptive false matches without sacrificing too much recall?"

The competition metric and the singleton rule make that distinction especially important.

A well-engineered combination of:

```text
normalization
+ blocking
+ character/token/address similarity
+ CatBoost/LightGBM
+ conservative thresholding
+ hard-negative mining
+ optional embeddings
```

is much more directly aligned with the problem than a large generative LLM.

