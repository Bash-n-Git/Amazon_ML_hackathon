# Pipeline Flowchart — Business Entity Resolution

> Full flow from raw data to the submission zip. Diagrams use Mermaid, which renders on GitHub, in VS Code
> (Markdown Preview Mermaid extension), in Obsidian and in Typora. A plain-text version is at the bottom.

---

## 1. Big picture

```mermaid
flowchart TD
    A[Raw TSV files<br/>S1, S2, S3 + ground truth] --> B[EDA<br/>understand the data]
    B --> C[Preprocessing<br/>clean + normalise text]
    C --> D[Transliteration<br/>Indian scripts → English letters]
    D --> E[Split by country<br/>US / India / France]
    E --> F[Candidate generation<br/>find look-alikes]
    F --> G[Stage-1 quick model<br/>keep top 12 per S1]
    G --> H[Feature extraction<br/>pair → numbers]
    H --> I[Stage-2 smart model<br/>match probability]
    I --> J[Settle conflicts<br/>one record → one S1]
    J --> K[Pick best list per S1<br/>expected F0.5]
    K --> L[Write output TSVs<br/>+ validate]
    L --> M[Submit + zip]

    V[Validation sets<br/>+ exact F0.5 scorer] -.checks.-> F
    V -.checks.-> I
    V -.tunes.-> K
```

---

## 2. EDA (Exploratory Data Analysis)

```mermaid
flowchart LR
    A[Load data<br/>polars, sep=tab] --> B[Save as parquet]
    B --> C[Group sizes<br/>avg 3.46, max 11]
    B --> D[Partition check<br/>record → max 1 S1]
    B --> E[Blank address<br/>15× match signal]
    B --> F[Country check<br/>matches 100% same country]
    B --> G[Script mix<br/>9 Indian scripts]
    B --> H[Noise catalogue<br/>typos, drops, abbreviations]
    C & D & E & F & G & H --> I[Design decisions]
```

---

## 3. Preprocessing + transliteration

```mermaid
flowchart TD
    A[Raw name + address] --> B[Keep raw copy<br/>never overwrite]
    A --> C[Lowercase, fix spaces,<br/>remove junk: << -- ()]
    C --> D[Normalise suffixes<br/>Ltd/Limited, Pvt, SARL, SAS]
    C --> E[Normalise address words<br/>Rd→road, St→street, R.→rue]
    C --> F[Space-stripped name<br/>for domain names like abc.com]
    C --> G{Indian script?}
    G -- yes --> H[Word table<br/>learned from train pairs]
    H -- unknown word --> I[IndicXlit<br/>offline model]
    H --> J[name_roman /<br/>address_roman]
    I --> J
    G -- no --> J
    E --> K[Parse address<br/>house no, street, city, state, PIN]
    D & F & J & K --> L[Clean table<br/>raw + normalised columns]
```

---

## 4. Candidate generation (blocking)

```mermaid
flowchart TD
    A[Clean table] --> B[Split by country]
    B --> C1[Name spelling search<br/>char n-gram TF-IDF, top 20]
    B --> C2[Address search<br/>char n-gram TF-IDF, top 20]
    B --> C3[House no / PIN + city<br/>index, top 10]
    B --> C4[AI embedding search<br/>name + address, top 20]
    B --> C5[Name-only search<br/>for blank addresses, top 10]
    B --> C6[Reverse search<br/>each S2/S3 → top 3 S1]
    C1 & C2 & C3 & C4 & C5 & C6 --> D[UNION of all<br/>OR, never AND]
    D --> E{Recall ≥ 98.5%<br/>on validation?}
    E -- no --> F[Improve search<br/>before modelling]
    F --> B
    E -- yes --> G[Shortlist<br/>~30-60 per S1]
```

---

## 5. Features + models

```mermaid
flowchart TD
    A[Shortlist] --> B[Stage-1 LightGBM<br/>cheap string features]
    B --> C[Top 12 per S1<br/>= candidate_pairs.tsv]
    C --> D1[Name features<br/>similarity, overlap, containment]
    C --> D2[Address features<br/>house no, street, city, state]
    C --> D3[Record flags<br/>blank addr, script, source]
    C --> D4[Competition features<br/>rank, margin, rival S1s]
    C --> D5[Group-fit features<br/>similar to other copies?]
    C --> D6[AI cross-encoder score<br/>fine-tuned, MIT/Apache]
    D1 & D2 & D3 & D4 & D5 & D6 --> E[Feature table<br/>float32]
    E --> F1[LightGBM]
    E --> F2[CatBoost]
    E --> F3[LightGBM ranker]
    F1 & F2 & F3 --> G[Blend + calibrate<br/>isotonic]
    G --> H[Match probability<br/>per pair]
```

---

## 6. Decision layer

```mermaid
flowchart TD
    A[Match probability per pair] --> B[Group by S2/S3 record]
    B --> C[Column normalise<br/>probabilities across S1s ≤ 1]
    C --> D[Drop weak rival claims<br/>below the tuned margin]
    D --> E[Group by S1]
    E --> F[Sort candidates<br/>high → low]
    F --> G[Try k = 0,1,2,…,12<br/>compute expected F0.5]
    G --> H{Best k}
    H -- k = 0 --> I[Empty list<br/>singleton]
    H -- k ≥ 1 --> J[Top k records]
    I & J --> K[matching_results.tsv]
```

---

## 7. Validation loop

```mermaid
flowchart LR
    A[Train data] --> B[Hold out 10% of S1 groups]
    B --> V1[Main set<br/>reweighted to test country mix]
    B --> V2[Extra-fakes set<br/>5.8 records per S1]
    A --> V3[US-only train →<br/>India test<br/>France proxy]
    V1 & V2 & V3 --> S[Exact macro F0.5 scorer]
    S --> R[Log: recall, cands/S1,<br/>P, R, F0.5, runtime]
    R --> T{Improved?}
    T -- yes --> K[Keep change]
    T -- no --> X[Revert + note<br/>in iteration log]
```

---

## 8. Output + submission

```mermaid
flowchart TD
    A[matching_results.tsv] --> C[Every test S1 has 1 row<br/>1,732,544 rows]
    B[candidate_pairs.tsv] --> C
    C --> D[validate_submission.py]
    D --> E{PASS?}
    E -- no --> F[Fix format:<br/>tabs, empty lists, duplicates]
    F --> D
    E -- yes --> G[Upload to portal]
    G --> H[Final: rerun from src/<br/>--check-ids]
    H --> I[Zip: output/ + code/ +<br/>Documentation_template.md]
```

---

## Plain-text version

```
RAW DATA (S1, S2, S3, ground truth)
   |
   v
EDA ............. group sizes, partition, blank address, country, scripts, noise types
   |
   v
PREPROCESS ...... keep raw | lowercase, junk strip | suffix + address-word normalise
   |              | space-stripped name | parse address
   v
TRANSLITERATE ... Indian script -> word table (from train) -> IndicXlit for unknown words
   |
   v
SPLIT BY COUNTRY  US | India | France
   |
   v
CANDIDATES (OR-union) ... name TF-IDF | address TF-IDF | house no/PIN | AI embedding
   |                       | name-only | reverse (record -> S1)
   |   gate: recall >= 98.5% on validation, else fix here
   v
STAGE-1 LightGBM ........ keep top 12 per S1  ==> candidate_pairs.tsv
   |
   v
FEATURES ....... name | address | record flags | competition | group-fit | AI cross-encoder
   |
   v
STAGE-2 MODELS . LightGBM + CatBoost + ranker -> blend -> calibrate -> probability
   |
   v
SETTLE CONFLICTS  one record -> one S1 (column normalise + margin)
   |
   v
PICK BEST LIST .. per S1, try k = 0..12, keep best expected F0.5 (k = 0 -> empty)
   |
   v
OUTPUT ......... matching_results.tsv -> validate_submission.py -> upload -> zip

VALIDATION (runs alongside): main set | extra-fakes set | US->India set -> exact F0.5 scorer
```
