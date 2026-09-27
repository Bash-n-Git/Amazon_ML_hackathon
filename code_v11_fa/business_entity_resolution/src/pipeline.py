"""Stage-1 pruning -> pair features -> stage-2 model -> decision layer -> submission files.

Usage:
  python pipeline.py stage1      # train stage-1 (OOF on train), prune train + test to top N_KEEP per S1
  python pipeline.py features    # pair features for pruned train + test
  python pipeline.py stage2      # stage-2 OOF on train, decision-layer comparison, fit final models
  python pipeline.py predict     # score test, decide, write output/*.tsv
Requires work/cands/<split>_<country>.parquet from build_candidates.py.
"""
import config  # noqa: F401  (first: points temp/cache dirs at the project drive)
import json
from pathlib import Path
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from blocking import K, union_wide
from config import OUT, SEED, WORK
from decide import column_normalize, select_expected_f, select_greedy_exclusive, select_threshold
from features import pair_features
from io_utils import truth_pairs, write_lists
from metric import report
from normalize import prepared

N_KEEP = 25
N_FOLDS = 3
FEAT_CHUNK_ROWS = 500_000  # pairs per pair_features call; peak ~7.6 KB/pair (parallel list exprs) => ~4 GB
TEST_MIX = {"US": 0.383, "India": 0.467, "France": 0.150}
BLK_COLS = [f"blk_{r}_{m}" for r in K for m in ("rank", "score")] + ["blk_n_hits"]
S1_FEATS = BLK_COLS + ["s1_n_cands", "s1_rank_best", "s1_gap_best"]
P = WORK / "pipe"
P.mkdir(exist_ok=True)
# Versioned runs (Plan_2): BER_RUN=v3 keeps stage-2 artefacts in work/pipe/v3 and outputs in output/v3,
# adds the feat2/ feature group and down-samples easy negatives. Unset = the original v2 run, untouched.
RUN = os.environ.get("BER_RUN", "")
PR = P / RUN if RUN else P
OUTR = OUT / RUN if RUN else OUT
USE_FEAT2 = bool(RUN)
PR.mkdir(exist_ok=True)
OUTR.mkdir(exist_ok=True)
EASY_P1 = 1e-3      # negatives with stage-1 p below this are "easy" ...
EASY_KEEP = 10      # ... keep 1 in 10 of them for training, with weight 10 (only when USE_FEAT2)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def countries(split):
    return sorted(p.stem.split("_", 1)[1] for p in (WORK / "cands").glob(f"{split}_*.parquet"))


def truth_gids() -> pl.DataFrame:
    ids = prepared("train", ["gid", "entity_id"])
    return (truth_pairs().join(ids.rename({"entity_id": "s1", "gid": "g1"}), on="s1")
            .join(ids.rename({"entity_id": "rid", "gid": "g2"}), on="rid")
            .select(pl.col("g1").alias("s1"), pl.col("g2").alias("rid"), pl.lit(1, pl.UInt8).alias("label")))


def wide(split, country, chunk: int = 0, n_chunks: int = 1) -> pl.DataFrame:
    """One row per (s1, rid). chunk/n_chunks: only S1s in that hash slice (all of an S1's candidates stay
    together, so per-S1 features are exact) — keeps the pivot's RAM down on the big test countries."""
    c = pl.scan_parquet(WORK / "cands" / f"{split}_{country}.parquet")
    if n_chunks > 1:
        c = c.filter((pl.col("s1").hash(SEED) % n_chunks) == chunk)
    w = union_wide(c.collect())
    for col in BLK_COLS:  # same columns on every split/country
        if col not in w.columns:
            w = w.with_columns(pl.lit(None, pl.Float32).alias(col))
    best = pl.max_horizontal([pl.col(f"blk_{r}_score") for r in K])
    w = w.with_columns(best.alias("_best")).with_columns(
        pl.len().over("s1").cast(pl.UInt16).alias("s1_n_cands"),
        pl.col("_best").rank("ordinal", descending=True).over("s1").cast(pl.UInt16).alias("s1_rank_best"),
        (pl.col("_best") - pl.col("_best").max().over("s1")).alias("s1_gap_best"),
    ).drop("_best")
    return w.select(["s1", "rid"] + S1_FEATS).with_columns(pl.lit(country).alias("country"))


def fold_of(s1: pl.Expr, k=N_FOLDS) -> pl.Expr:
    return (s1.hash(SEED) % k).cast(pl.UInt8)


LGB1 = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.9,
            bagging_fraction=0.5, bagging_freq=1, verbose=-1, seed=SEED, num_threads=24)
LGB2 = dict(objective="binary", learning_rate=0.1, num_leaves=127, min_data_in_leaf=50, feature_fraction=0.8,
            bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, metric="binary_logloss", verbose=-1, seed=SEED,
            num_threads=24)


# Stage-2 learner. "lgb" (CPU) is the default: measured 0.22 s/round vs 0.6 s/round for XGBoost-cuda on 3M rows.
GBM = os.environ.get("BER_GBM", "lgb")
XGB2 = dict(objective="binary:logistic", eval_metric="logloss", device="cuda", tree_method="hist",
            max_depth=0, grow_policy="lossguide", max_leaves=127, learning_rate=0.1, min_child_weight=5,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, max_bin=256, seed=SEED)
MAX_ROUNDS = 5000


def fit_fold(Xt, yt, Xv, yv, wt=None):
    """Train with early stopping on the validation fold. Returns (valid predictions, best iteration, logloss).
    wt: optional training weights (easy-negative down-sampling)."""
    if GBM == "xgb":
        import xgboost as xgb
        dt = xgb.QuantileDMatrix(Xt, yt, weight=wt)
        dv = xgb.QuantileDMatrix(Xv, yv, ref=dt)
        res = {}
        b = xgb.train(XGB2, dt, MAX_ROUNDS, evals=[(dv, "v")], early_stopping_rounds=100, evals_result=res,
                      verbose_eval=False)
        it = b.best_iteration + 1
        return b.predict(dv, iteration_range=(0, it)), it, res["v"]["logloss"][b.best_iteration]
    bst = lgb.train(LGB2, lgb.Dataset(Xt, yt, weight=wt), MAX_ROUNDS, valid_sets=[lgb.Dataset(Xv, yv)],
                    callbacks=[lgb.early_stopping(100, verbose=False)])
    return (bst.predict(Xv, num_iteration=bst.best_iteration), bst.best_iteration,
            bst.best_score["valid_0"]["binary_logloss"])


def fit_final(X, y, rounds, w=None):
    if GBM == "xgb":
        import xgboost as xgb
        xgb.train(XGB2, xgb.QuantileDMatrix(X, y, weight=w), rounds).save_model(str(PR / "stage2.ubj"))
    else:
        lgb.train(LGB2, lgb.Dataset(X, y, weight=w), rounds).save_model(str(PR / "stage2.txt"))
    (PR / "stage2_backend.txt").write_text(GBM)


def load_final():
    """Returns a function X -> P(match), for whichever backend trained the saved model."""
    backend = (PR / "stage2_backend.txt").read_text().strip() if (PR / "stage2_backend.txt").exists() else "lgb"
    if backend == "xgb":
        import xgboost as xgb
        b = xgb.Booster()
        b.load_model(str(PR / "stage2.ubj"))
        b.set_param({"device": "cuda"})
        return lambda X: b.inplace_predict(X)
    bst = lgb.Booster(model_file=str(PR / "stage2.txt"))
    return bst.predict


def _X(df, feats):
    return df.select(pl.col(feats).cast(pl.Float32)).to_numpy()


def _xy(df, feats):
    return _X(df, feats), df["label"].to_numpy()


# ---------------------------------------------------------------- stage 1
S1_TRAIN_FRAC = 0.25  # stage-1 models see a sample of S1s; 13 features need far fewer than ~70M rows


TEST_WIDE_CHUNKS = 8


def stage1_prune(split, bst):
    """Prune each country of `split` to N_KEEP/S1, in S1 slices; resumable per country.
    Train-type splits (e.g. trainrest) also get the ground-truth label."""
    tg = truth_gids() if split.startswith("train") else None
    for c in countries(split):
        out = P / f"{split}_s1_{c}.parquet"
        if out.exists():
            log(f"stage1 {split} {c}: skip (exists)")
            continue
        parts = []
        for i in range(TEST_WIDE_CHUNKS):
            w = wide(split, c, i, TEST_WIDE_CHUNKS)
            w = w.with_columns(pl.Series("p1", bst.predict(_X(w, S1_FEATS))))
            w = w.filter(pl.col("p1").rank("ordinal", descending=True).over("s1") <= N_KEEP)
            if tg is not None:
                w = w.join(tg, on=["s1", "rid"], how="left").with_columns(pl.col("label").fill_null(0).cast(pl.UInt8))
            parts.append(w)
            del w
        pl.concat(parts).write_parquet(out)
        log(f"stage1 {split} {c}: pruned to {N_KEEP}/S1")
        del parts


def stage1_test(bst):
    stage1_prune("test", bst)


def stage1_rest():
    """The 70% of train S1s outside the original sample: pruned with the saved stage-1 model (which never saw
    them), like test."""
    stage1_prune("trainrest", lgb.Booster(model_file=str(P / "stage1.txt")))


def stage1():
    """Built one country at a time: the full train union does not fit in RAM alongside a feature matrix."""
    if (P / "train_s1.parquet").exists() and (P / "stage1.txt").exists():
        log("stage1 train: skip (train_s1.parquet + stage1.txt exist)")
        return stage1_test(lgb.Booster(model_file=str(P / "stage1.txt")))
    tg = truth_gids()
    files = []
    n_true_all = n_hit = 0
    for c in countries("train"):
        w = wide("train", c).join(tg, on=["s1", "rid"], how="left").with_columns(
            pl.col("label").fill_null(0).cast(pl.UInt8), fold_of(pl.col("s1"), 3).alias("fold"))
        n_true_all += tg.join(w.select("s1").unique(), on="s1").height
        n_hit += int(w["label"].sum())
        files.append(P / f"train_wide_{c}.parquet")
        w.write_parquet(files[-1])
        log(f"stage1 train {c}: {w.height:,} union rows")
        del w
    log(f"stage1 blocking recall {n_hit / n_true_all:.4f}")

    def sample(fold_expr):
        s = (pl.col("s1").hash(SEED + 1) % 1000) < int(S1_TRAIN_FRAC * 1000)
        return pl.concat([pl.scan_parquet(f).filter(fold_expr & s).collect() for f in files])

    models = []
    for f in range(3):
        X, y = _xy(sample(pl.col("fold") != f), S1_FEATS)
        models.append(lgb.train(LGB1, lgb.Dataset(X, y), 300))
        del X, y
    kept = []
    n_keep_true = 0
    for path in files:
        w = pl.read_parquet(path)
        oof = np.zeros(w.height, np.float32)
        for f in range(3):
            m = (w["fold"] == f).to_numpy()
            oof[m] = models[f].predict(_X(w.filter(pl.col("fold") == f), S1_FEATS))
        k = w.with_columns(pl.Series("p1", oof)).filter(
            pl.col("p1").rank("ordinal", descending=True).over("s1") <= N_KEEP).drop("fold")
        n_keep_true += int(k["label"].sum())
        kept.append(k)
        del w
    keep = pl.concat(kept)
    log(f"stage1 kept {keep.height / keep['s1'].n_unique():.1f}/S1, recall after prune {n_keep_true / n_true_all:.4f}")
    keep.write_parquet(P / "train_s1.parquet")
    del keep, kept
    X, y = _xy(sample(pl.lit(True)), S1_FEATS)
    bst = lgb.train(LGB1, lgb.Dataset(X, y), 300)
    bst.save_model(str(P / "stage1.txt"))
    del X, y
    stage1_test(bst)


# ---------------------------------------------------------------- features
FEAT = P / "feat"  # one parquet per (split, country, chunk): nothing split-sized is ever held in RAM


def feat_files(split):
    return sorted(FEAT.glob(f"{split}_*.parquet"))


def data_split(split):
    """Which records a pipeline split draws from: train, trainrest -> train; test -> test."""
    return "train" if split.startswith("train") else "test"


def s1_files(split):
    return [P / "train_s1.parquet"] if split == "train" else sorted(P.glob(f"{split}_s1_*.parquet"))


def features(*splits):
    """Resumable: a chunk whose file exists is skipped. Default splits: train, test."""
    from features import REC_COLS
    FEAT.mkdir(exist_ok=True)
    for split in splits or ("train", "test"):
        prepared(data_split(split), ["gid"])  # make sure the cache exists; records are then read per country
        files = s1_files(split)
        for f in files:
            base = pl.read_parquet(f)
            for c in base["country"].unique().sort().to_list():
                b = base.filter(pl.col("country") == c)
                rc = (pl.scan_parquet(WORK / f"{data_split(split)}_prep.parquet").filter(pl.col("country") == c)
                        .select(REC_COLS).collect())
                t = time.time()
                # chunk by S1 (competition features are per-S1) so the string lists fit in RAM
                n_chunks = max(1, -(-b.height // FEAT_CHUNK_ROWS))
                b = b.with_columns((pl.col("s1").hash(SEED) % n_chunks).alias("_chunk"))
                for i in range(n_chunks):
                    out = FEAT / f"{split}_{c}_{i:03d}.parquet"
                    if out.exists():
                        continue
                    pair_features(b.filter(pl.col("_chunk") == i).drop("_chunk"), rc).write_parquet(out)
                    log(f"  features {split} {c} chunk {i + 1}/{n_chunks}: {time.time() - t:.0f}s so far")
                log(f"features {split} {c}: {b.height:,} rows in {n_chunks} chunks, {time.time() - t:.0f}s")
                del rc, b
            del base


FEAT2 = P / "feat2"  # v2 feature group, same chunk names as feat/


def features2(*splits):
    """Plan_2 feature group (features_v2.py) for every existing feat/ chunk. Resumable per chunk."""
    from features_v2 import pair_features2, prep2
    FEAT2.mkdir(exist_ok=True)
    for split in splits or ("train", "test"):
        t = time.time()
        p2 = prep2(data_split(split))
        log(f"features2 {split}: record columns ready ({p2.height:,} records, {time.time() - t:.0f}s)")
        files = feat_files(split)
        for i, f in enumerate(files):
            out = FEAT2 / f.name
            if out.exists():
                continue
            chunk = pl.read_parquet(f)
            ids = pl.concat([chunk["s1"], chunk["rid"]]).unique().implode()
            pair_features2(chunk, p2.filter(pl.col("gid").is_in(ids))).write_parquet(out)
            if (i + 1) % 10 == 0 or i + 1 == len(files):
                log(f"  features2 {split} chunk {i + 1}/{len(files)}: {time.time() - t:.0f}s so far")
        log(f"features2 {split}: done, {time.time() - t:.0f}s")
        del p2


def load_feat(f, columns=None):
    """One feature chunk: feat/ joined with feat2/ when USE_FEAT2."""
    d = pl.read_parquet(f, columns=columns) if columns is None else pl.read_parquet(f, columns=[c for c in columns if not c.startswith("v2_")])
    if USE_FEAT2:
        cols2 = None if columns is None else ["s1", "rid"] + [c for c in columns if c.startswith("v2_")]
        d = d.join(pl.read_parquet(FEAT2 / f.name, columns=cols2), on=["s1", "rid"], how="left")
    return d.with_columns(pl.col(pl.Float64).cast(pl.Float32))


def feat_cols(df):
    drop = {"s1", "rid", "label", "country", "fold", "p2", "p", "_keep", "_w"}
    return [c for c in df.columns if c not in drop]


# ---------------------------------------------------------------- stage 2 + decision
def evaluate(pred, name, truth, meta):
    r = report(pred, truth, meta, TEST_MIX)
    log(f"  {name:38s} F0.5 {r['macro_f05']:.4f}  test-mix {r['test_mix_f05']:.4f}  P {r['pair_precision']:.3f}  R {r['pair_recall']:.3f}  pred/S1 {r['pred_per_s1']:.2f}  buckets {r['by_bucket']}")
    return r


def stage2():
    df = (pl.concat([load_feat(f) for f in feat_files("train")], how="diagonal_relaxed")  # f32: halves RAM
            .with_columns(fold_of(pl.col("s1")).alias("fold")))
    feats = feat_cols(df)
    log(f"stage2 [{RUN or 'v2'}] rows {df.height:,}, {len(feats)} features, positives {df['label'].sum():,}")
    # training weights: keep 1 in EASY_KEEP easy negatives with weight EASY_KEEP (validation rows are never dropped)
    easy = (pl.col("label") == 0) & (pl.col("p1") < EASY_P1)
    keep = ~easy | ((pl.struct("s1", "rid").hash(SEED) % EASY_KEEP) == 0)
    df = df.with_columns(pl.when(pl.lit(not USE_FEAT2)).then(True).otherwise(keep).alias("_keep"),
                         pl.when(pl.lit(USE_FEAT2) & easy).then(float(EASY_KEEP)).otherwise(1.0).cast(pl.Float32).alias("_w"))
    log(f"  training rows after easy-negative down-sampling: {df['_keep'].sum():,}")
    oof = np.zeros(df.height, np.float32)
    iters = []
    log(f"stage2 backend: {GBM}")
    for f in range(N_FOLDS):
        t = time.time()
        trn, val = df.filter((pl.col("fold") != f) & pl.col("_keep")), df.filter(pl.col("fold") == f)
        Xt, yt = _xy(trn, feats); Xv, yv = _xy(val, feats)
        wt = trn["_w"].to_numpy()
        del trn, val
        pv, it, loss = fit_fold(Xt, yt, Xv, yv, wt)
        oof[(df["fold"] == f).to_numpy()] = pv
        iters.append(it)
        log(f"  fold {f}: best_iter {it}, logloss {loss:.5f}, {time.time() - t:.0f}s")
        del Xt, yt, Xv, yv
    df = df.with_columns(pl.Series("p", oof))
    df.select("s1", "rid", "label", "country", "p").write_parquet(PR / "train_oof.parquet")

    # evaluation universe: every sampled S1 (all S1 that reached stage 1)
    s1_meta = pl.read_parquet(P / "train_s1.parquet", columns=["s1", "country"]).unique("s1")
    truth = truth_gids().join(s1_meta.select("s1"), on="s1").select("s1", "rid")
    res = {}
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        res[f"thr{t}"] = evaluate(select_threshold(df, t), f"threshold {t}", truth, s1_meta)
    for t in (0.5, 0.6, 0.65, 0.7, 0.75, 0.8):
        res[f"greedy{t}"] = evaluate(select_greedy_exclusive(df, t), f"greedy exclusive + threshold {t}", truth, s1_meta)
    cn = column_normalize(df)
    res["expF"] = evaluate(select_expected_f(cn), "colnorm + expected F0.5", truth, s1_meta)
    for m in (0.1, 0.3, 0.5):
        res[f"expF_m{m}"] = evaluate(select_expected_f(column_normalize(df, margin=m)), f"colnorm(margin {m}) + expected F0.5", truth, s1_meta)
    (PR / "stage2_eval.json").write_text(json.dumps(res, indent=1))

    df = df.filter(pl.col("_keep"))
    X, y = _xy(df, feats)
    w = df["_w"].to_numpy()
    del df
    fit_final(X, y, int(np.mean(iters) * 1.1), w)
    (PR / "feats.json").write_text(json.dumps(feats))
    log("stage2 final model saved")


# ---------------------------------------------------------------- stage 2 on ALL train S1s (Plan_2 I7)
def stage2full():
    """Stage 2 on train + trainrest (100% of train S1s), memory-lean for ~55M pairs:
    - training matrix built chunk by chunk from kept rows only, turned into one LightGBM Dataset, raw freed;
      folds are Dataset.subset() views (no copies)
    - validation = fold 0 (1/3 of all train S1s): early stopping + rule comparison; its rows are scored
      chunk by chunk (all rows, incl. easy negatives)
    - final model on every kept row. Artefacts in PR (use BER_RUN=v6)."""
    files = feat_files("train") + feat_files("trainrest")
    feats = feat_cols(load_feat(files[0]).head(1))
    easy = (pl.col("label") == 0) & (pl.col("p1") < EASY_P1)
    keep = ~easy | ((pl.struct("s1", "rid").hash(SEED) % EASY_KEEP) == 0)
    # pass 1: count kept rows per chunk (cheap columns), so the matrix is allocated once (no vstack copy)
    counts = [pl.read_parquet(f, columns=["s1", "rid", "label", "p1"]).filter(keep).height for f in files]
    n = sum(counts)
    X = np.empty((n, len(feats)), np.float32)
    y, w, fold = np.empty(n, np.float32), np.empty(n, np.float32), np.empty(n, np.uint8)
    t = time.time()
    o = 0
    for i, (f, k) in enumerate(zip(files, counts)):
        d = load_feat(f).with_columns(fold_of(pl.col("s1")).alias("fold")).filter(keep)
        assert d.height == k
        X[o:o + k] = _X(d, feats)
        y[o:o + k] = d["label"].to_numpy()
        w[o:o + k] = d.select(pl.when(easy).then(float(EASY_KEEP)).otherwise(1.0)).to_series().to_numpy()
        fold[o:o + k] = d["fold"].to_numpy()
        o += k
        if (i + 1) % 40 == 0:
            log(f"  stage2full loading {i + 1}/{len(files)} chunks, {o:,}/{n:,} kept rows, {time.time() - t:.0f}s")
        del d
    log(f"stage2full [{RUN}] {len(y):,} kept rows ({int(y.sum()):,} positives), {len(feats)} features")
    ds = lgb.Dataset(X, y, weight=w, free_raw_data=True, params={"max_bin": 255}).construct()
    del X
    tr_idx, va_idx = np.flatnonzero(fold != 0), np.flatnonzero(fold == 0)
    t = time.time()
    bst = lgb.train(LGB2, ds.subset(tr_idx), MAX_ROUNDS, valid_sets=[ds.subset(va_idx)],
                    callbacks=[lgb.early_stopping(100, verbose=False)])
    it = bst.best_iteration
    log(f"  fold 0 (validation): best_iter {it}, weighted logloss {bst.best_score['valid_0']['binary_logloss']:.5f}, {time.time() - t:.0f}s")
    # score every fold-0 row (easy negatives included), chunk by chunk
    parts = []
    for f in files:
        d = load_feat(f).with_columns(fold_of(pl.col("s1")).alias("fold")).filter(pl.col("fold") == 0)
        parts.append(d.select("s1", "rid", "label", "country").with_columns(
            pl.Series("p", bst.predict(_X(d, feats), num_iteration=it).astype(np.float32))))
    oof = pl.concat(parts)
    del parts
    oof.write_parquet(PR / "train_oof.parquet")
    s1_meta = (pl.concat([pl.read_parquet(f, columns=["s1", "country"]) for f in s1_files("train") + s1_files("trainrest")])
                 .unique("s1").filter(fold_of(pl.col("s1")) == 0))
    truth = truth_gids().join(s1_meta.select("s1"), on="s1").select("s1", "rid")
    log(f"  validation: {s1_meta.height:,} S1s (fold 0 of all train)")
    res = {}
    for thr in (0.5, 0.6, 0.65, 0.7, 0.75, 0.8):
        res[f"greedy{thr}"] = evaluate(select_greedy_exclusive(oof, thr), f"greedy exclusive + threshold {thr}", truth, s1_meta)
    (PR / "stage2_eval.json").write_text(json.dumps(res, indent=1))
    del oof
    t = time.time()
    final = lgb.train(LGB2, ds, int(it * 1.1))
    final.save_model(str(PR / "stage2.txt"))
    (PR / "stage2_backend.txt").write_text("lgb")
    (PR / "feats.json").write_text(json.dumps(feats))
    log(f"stage2full final model saved ({int(it * 1.1)} rounds, {time.time() - t:.0f}s)")


def stage2full_oof(folds="1,2"):
    """Out-of-fold scores for the remaining folds of the stage2full run (same params, fixed rounds = fold-0 best
    iteration), so every train pair gets an honest p — needed for record-side competition features.
    Writes PR/train_oof_f<k>.parquet per fold; resumable per fold."""
    files = feat_files("train") + feat_files("trainrest")
    feats = json.loads((PR / "feats.json").read_text())
    rounds = int(int(lgb.Booster(model_file=str(PR / "stage2.txt")).num_trees()) / 1.1)
    easy = (pl.col("label") == 0) & (pl.col("p1") < EASY_P1)
    keep = ~easy | ((pl.struct("s1", "rid").hash(SEED) % EASY_KEEP) == 0)
    for k in [int(x) for x in folds.split(",")]:
        out = PR / f"train_oof_f{k}.parquet"
        if out.exists():
            log(f"stage2full_oof fold {k}: skip (exists)")
            continue
        counts = [pl.read_parquet(f, columns=["s1", "rid", "label", "p1"]).filter(keep & (fold_of(pl.col("s1")) != k)).height
                  for f in files]
        n = sum(counts)
        X = np.empty((n, len(feats)), np.float32)
        y, w = np.empty(n, np.float32), np.empty(n, np.float32)
        o = 0
        for f, c in zip(files, counts):
            d = load_feat(f).filter(keep & (fold_of(pl.col("s1")) != k))
            X[o:o + c] = _X(d, feats); y[o:o + c] = d["label"].to_numpy()
            w[o:o + c] = d.select(pl.when(easy).then(float(EASY_KEEP)).otherwise(1.0)).to_series().to_numpy()
            o += c
            del d
        t = time.time()
        ds = lgb.Dataset(X, y, weight=w, free_raw_data=True).construct()
        del X
        bst = lgb.train(LGB2, ds, rounds)
        del ds
        parts = []
        for f in files:
            d = load_feat(f).filter(fold_of(pl.col("s1")) == k)
            parts.append(d.select("s1", "rid", "label", "country").with_columns(
                pl.Series("p", bst.predict(_X(d, feats)).astype(np.float32))))
        pl.concat(parts).write_parquet(out)
        log(f"stage2full_oof fold {k}: {n:,} training rows, {rounds} rounds, {time.time() - t:.0f}s")


# ---------------------------------------------------------------- self-training for unseen countries (Plan_2)
PSEUDO_HI, PSEUDO_LO = 0.97, 0.03


def pseudo(rounds="1"):
    """Final stage-2 model trained on labelled train pairs + confident pseudo-labels for test countries.
    Base scores from work/pipe/<BER_PSEUDO_FROM>/test_scores.parquet (default v3); countries from
    BER_PSEUDO_COUNTRIES (default France). Pseudo-positive = one-owner winner with p >= 0.97; pseudo-negative =
    p <= 0.03; everything in between is left out. Uses test inputs only — no labels, no external data.
    Writes the model to PR like stage2(), so `predict` then scores the test set with it."""
    src = P / os.environ.get("BER_PSEUDO_FROM", "v3")
    ctry = os.environ.get("BER_PSEUDO_COUNTRIES", "France").split(",")
    feats = json.loads((src / "feats.json").read_text())
    tr = pl.concat([load_feat(f) for f in feat_files("train")], how="diagonal_relaxed")
    easy = (pl.col("label") == 0) & (pl.col("p1") < EASY_P1)
    tr = tr.filter(~easy | ((pl.struct("s1", "rid").hash(SEED) % EASY_KEEP) == 0)).with_columns(
        pl.when(easy).then(float(EASY_KEEP)).otherwise(1.0).cast(pl.Float32).alias("_w"))
    Xtr, ytr = _xy(tr, feats)
    wtr = tr["_w"].to_numpy()
    del tr
    tids = prepared("test", ["gid", "country"]).filter(pl.col("country").is_in(ctry)).select(pl.col("gid").alias("s1"))
    files = [f for f in feat_files("test") if f.name.split("_")[1] in ctry]
    te = pl.concat([load_feat(f, ["s1", "rid"] + feats) for f in files], how="diagonal_relaxed")
    Xte = _X(te, feats)
    d = pl.read_parquet(src / "test_scores.parquet").join(tids, on="s1")
    d = te.select("s1", "rid").join(d, on=["s1", "rid"], how="left")  # align with Xte row order
    rounds = int(rounds)
    for rnd in range(1, rounds + 1):
        win = apply_rule(d, f"greedy{PSEUDO_HI}").with_columns(pl.lit(1).alias("pl"))
        lab = d.join(win, on=["s1", "rid"], how="left").with_columns(
            pl.when(pl.col("pl") == 1).then(1).when(pl.col("p") <= PSEUDO_LO).then(0).otherwise(None).alias("pl"))
        m = lab["pl"].is_not_null().to_numpy()
        yp = lab["pl"].to_numpy()[m].astype(float)
        log(f"pseudo round {rnd}: {m.sum():,} of {len(m):,} {ctry} test pairs labelled ({yp.mean():.1%} positive)")
        rounds_gbm = 1350  # ~mean best iteration of the v3 folds x 1.1
        bst = lgb.train(LGB2, lgb.Dataset(np.vstack([Xtr, Xte[m]]), np.concatenate([ytr, yp]),
                                          weight=np.concatenate([wtr, np.ones(m.sum(), np.float32)])), rounds_gbm)
        d = d.select("s1", "rid").with_columns(pl.Series("p", bst.predict(Xte).astype(np.float32)))
    bst.save_model(str(PR / "stage2.txt"))
    (PR / "stage2_backend.txt").write_text("lgb")
    (PR / "feats.json").write_text(json.dumps(feats))
    log(f"pseudo: model saved to {PR}")


# ---------------------------------------------------------------- stage 3 (Plan_2 I2): score-based competition
STACK_BASE = ["blk_comb_rev_rank", "blk_comb_rev_score", "rec_addr_empty", "rec_src", "name_key_eq", "name_tset",
              "addr_tset", "v2_rec_vocab_share", "v2_num_first_eq", "v2_words_tsort", "p1"]


def stack_feats(df: pl.DataFrame) -> pl.DataFrame:
    """Per-S1 context of the stage-2 probability p. Only S1-side statistics: every candidate of an evaluated S1
    is present in train and test alike, so these are unbiased (record-side ones are not — 30% S1 sample)."""
    lp = (pl.col("p").clip(1e-6, 1 - 1e-6) / (1 - pl.col("p").clip(1e-6, 1 - 1e-6))).log()
    second = df.group_by("s1").agg(pl.col("p").top_k(2).min().alias("s3_second"))
    return df.join(second, on="s1", how="left").with_columns(
        lp.alias("s3_logit"),
        pl.col("p").rank("min", descending=True).over("s1").alias("s3_rank"),
        (pl.col("p").max().over("s1") - pl.col("p")).alias("s3_gap_best"),
        pl.col("p").sum().over("s1").alias("s3_sum"),
        (pl.col("p") >= 0.5).sum().over("s1").alias("s3_n_05"),
        (pl.col("p") >= 0.9).sum().over("s1").alias("s3_n_09"),
    )


S3_FEATS = ["p", "s3_logit", "s3_rank", "s3_gap_best", "s3_sum", "s3_n_05", "s3_n_09", "s3_second"] + STACK_BASE
LGB3 = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.9,
            bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=SEED, num_threads=24)


def stage3():
    """Stacked model on stage-2 OOF p + per-S1 competition. OOF on the same folds; writes PR/stage3.txt,
    PR/train_oof3.parquet and adds s3-rule results to PR/stage2_eval.json."""
    base = pl.read_parquet(PR / "train_oof.parquet")
    extra = pl.concat([load_feat(f, ["s1", "rid"] + STACK_BASE) for f in feat_files("train")], how="diagonal_relaxed")
    df = stack_feats(base.join(extra, on=["s1", "rid"], how="left")).with_columns(fold_of(pl.col("s1")).alias("fold"))
    del extra
    log(f"stage3 [{RUN or 'v2'}] rows {df.height:,}, {len(S3_FEATS)} features")
    oof = np.zeros(df.height, np.float32)
    iters = []
    for f in range(N_FOLDS):
        t = time.time()
        trn, val = df.filter(pl.col("fold") != f), df.filter(pl.col("fold") == f)
        Xt, yt = _xy(trn, S3_FEATS); Xv, yv = _xy(val, S3_FEATS)
        bst = lgb.train(LGB3, lgb.Dataset(Xt, yt), 3000, valid_sets=[lgb.Dataset(Xv, yv)],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[(df["fold"] == f).to_numpy()] = bst.predict(Xv, num_iteration=bst.best_iteration)
        iters.append(bst.best_iteration)
        log(f"  stage3 fold {f}: best_iter {bst.best_iteration}, logloss {bst.best_score['valid_0']['binary_logloss']:.5f}, {time.time() - t:.0f}s")
        del trn, val, Xt, Xv
    d3 = df.select("s1", "rid", "label", "country").with_columns(pl.Series("p", oof))
    d3.write_parquet(PR / "train_oof3.parquet")
    s1_meta = pl.read_parquet(P / "train_s1.parquet", columns=["s1", "country"]).unique("s1")
    truth = truth_gids().join(s1_meta.select("s1"), on="s1").select("s1", "rid")
    res = json.loads((PR / "stage2_eval.json").read_text())
    for t in (0.5, 0.6, 0.65, 0.7, 0.75, 0.8):
        res[f"s3greedy{t}"] = evaluate(select_greedy_exclusive(d3, t), f"stage3 greedy + threshold {t}", truth, s1_meta)
    (PR / "stage2_eval.json").write_text(json.dumps(res, indent=1))
    X, y = _xy(df, S3_FEATS)
    del df
    lgb.train(LGB3, lgb.Dataset(X, y), int(np.mean(iters) * 1.1)).save_model(str(PR / "stage3.txt"))
    log("stage3 final model saved")


def rec_feats(df: pl.DataFrame) -> pl.DataFrame:
    """Record-side context of p: how strongly do OTHER S1s claim this record? Unbiased only when every S1 of the
    split has candidates (100% train coverage, and test)."""
    top2 = df.group_by("rid").agg(pl.col("p").top_k(2).alias("_t2"), pl.len().alias("r_n_claim"),
                                  pl.col("p").sum().alias("r_sum"))
    df = df.join(top2, on="rid", how="left").with_columns(
        pl.col("_t2").list.get(0).alias("_b1"), pl.col("_t2").list.get(1, null_on_oob=True).fill_null(0.0).alias("_b2"))
    other = pl.when(pl.col("p") >= pl.col("_b1")).then(pl.col("_b2")).otherwise(pl.col("_b1"))
    return df.with_columns(other.alias("r_max_other"), (pl.col("p") - other).alias("r_gap_other"),
                           pl.col("p").rank("min", descending=True).over("rid").alias("r_rank")).drop("_t2", "_b1", "_b2")


S3F_FEATS = S3_FEATS + ["r_n_claim", "r_sum", "r_max_other", "r_gap_other", "r_rank"]


def stage3full():
    """v7: stacked model on the full out-of-fold p of the stage2full run (all train S1s) with S1-side AND
    record-side competition. Evaluated on all train S1s — the first test-like validation (every competitor present)."""
    oof = pl.concat([pl.read_parquet(PR / "train_oof.parquet")] +
                    [pl.read_parquet(PR / f"train_oof_f{k}.parquet") for k in (1, 2)])
    files = feat_files("train") + feat_files("trainrest")
    extra = pl.concat([load_feat(f, ["s1", "rid"] + STACK_BASE) for f in files], how="diagonal_relaxed")
    df = rec_feats(stack_feats(oof.join(extra, on=["s1", "rid"], how="left"))).with_columns(fold_of(pl.col("s1")).alias("fold"))
    del extra
    s1_meta = pl.concat([pl.read_parquet(f, columns=["s1", "country"]) for f in s1_files("train") + s1_files("trainrest")]).unique("s1")
    truth = truth_gids().join(s1_meta.select("s1"), on="s1").select("s1", "rid")
    log(f"stage3full [{RUN}] rows {df.height:,}, S1s {s1_meta.height:,}, {len(S3F_FEATS)} features")
    res = {}
    for thr in (0.6, 0.65, 0.7, 0.75, 0.8):  # base stage-2 p, all competitors present
        res[f"greedy{thr}"] = evaluate(select_greedy_exclusive(df, thr), f"[full] stage2 greedy + threshold {thr}", truth, s1_meta)
    easy = (pl.col("label") == 0) & (pl.col("p") < 1e-3)
    df = df.with_columns((~easy | ((pl.struct("s1", "rid").hash(SEED) % EASY_KEEP) == 0)).alias("_keep"),
                         pl.when(easy).then(float(EASY_KEEP)).otherwise(1.0).cast(pl.Float32).alias("_w"))
    oof3 = np.zeros(df.height, np.float32)
    iters = []
    for f in range(N_FOLDS):
        t = time.time()
        trn = df.filter((pl.col("fold") != f) & pl.col("_keep"))
        val = df.filter(pl.col("fold") == f)
        Xt, yt = _xy(trn, S3F_FEATS); Xv, yv = _xy(val, S3F_FEATS)
        bst = lgb.train(LGB3, lgb.Dataset(Xt, yt, weight=trn["_w"].to_numpy()), 3000, valid_sets=[lgb.Dataset(Xv, yv)],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
        oof3[(df["fold"] == f).to_numpy()] = bst.predict(Xv, num_iteration=bst.best_iteration)
        iters.append(bst.best_iteration)
        log(f"  stage3full fold {f}: best_iter {bst.best_iteration}, logloss {bst.best_score['valid_0']['binary_logloss']:.5f}, {time.time() - t:.0f}s")
        del trn, val, Xt, Xv
    d3 = df.select("s1", "rid", "label", "country").with_columns(pl.Series("p", oof3))
    d3.write_parquet(PR / "train_oof3full.parquet")
    for thr in (0.5, 0.6, 0.65, 0.7, 0.75, 0.8):
        res[f"s3greedy{thr}"] = evaluate(select_greedy_exclusive(d3, thr), f"[full] stage3 greedy + threshold {thr}", truth, s1_meta)
    (PR / "stage3full_eval.json").write_text(json.dumps(res, indent=1))
    df = df.filter(pl.col("_keep"))
    X, y = _xy(df, S3F_FEATS)
    w = df["_w"].to_numpy()
    del df
    lgb.train(LGB3, lgb.Dataset(X, y, weight=w), int(np.mean(iters) * 1.1)).save_model(str(PR / "stage3.txt"))
    (PR / "stage3_feats.json").write_text(json.dumps(S3F_FEATS))
    log("stage3full final model saved")


# ---------------------------------------------------------------- v8 (Plan_3): cross-encoder + group consistency
BAND = (0.01, 0.99)


def group_feats(df: pl.DataFrame, split: str) -> pl.DataFrame:
    """C2 (GraLMatch idea): for uncertain pairs, how similar is the record to the OTHER records already confidently
    assigned (p >= 0.5) to the same S1? df: s1, rid, p. Returns s1, rid, g_* (only for band pairs)."""
    from rapidfuzz import fuzz
    from rapidfuzz.process import cpdist
    tgt = df.filter(pl.col("p").is_between(*BAND, closed="none")).select("s1", "rid")
    conf = df.filter(pl.col("p") >= 0.5).select("s1", pl.col("rid").alias("rc"))
    x = tgt.join(conf, on="s1").filter(pl.col("rid") != pl.col("rc"))
    ids = pl.concat([x["rid"], x["rc"]]).unique()
    r = (pl.scan_parquet(WORK / f"{data_split(split)}_prep.parquet").select("gid", "src", "name_roman", "addr_roman")
           .filter(pl.col("gid").is_in(ids.implode())).collect())
    x = (x.join(r.rename({"gid": "rid", "src": "st", "name_roman": "nt", "addr_roman": "at"}), on="rid")
          .join(r.rename({"gid": "rc", "src": "sc", "name_roman": "nc", "addr_roman": "ac"}), on="rc"))
    ns = cpdist(x["nt"].to_list(), x["nc"].to_list(), scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
    ad = cpdist(x["at"].to_list(), x["ac"].to_list(), scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
    x = x.with_columns(pl.Series("ns", ns), pl.when((pl.col("at") == "") | (pl.col("ac") == "")).then(None)
                       .otherwise(pl.Series("ad", ad)).alias("ad"))
    g = x.group_by("s1", "rid").agg(pl.col("ns").max().alias("g_name_max"), pl.col("ns").mean().alias("g_name_mean"),
                                    pl.col("ad").max().alias("g_addr_max"), pl.len().alias("g_n_conf"),
                                    (pl.col("st") == pl.col("sc")).sum().alias("g_n_conf_same_src"))
    return tgt.join(g, on=["s1", "rid"], how="left").with_columns(pl.col("g_n_conf").fill_null(0),
                                                                    pl.col("g_n_conf_same_src").fill_null(0))


G_FEATS = ["g_name_max", "g_name_mean", "g_addr_max", "g_n_conf", "g_n_conf_same_src"]

IDF_BAND = (0.001, 0.999)


def token_idf(split: str) -> pl.DataFrame:
    """Document frequency -> IDF of name and address tokens over all records of the split (cached)."""
    path = WORK / f"{data_split(split)}_tokidf.parquet"
    if path.exists():
        return pl.read_parquet(path)
    r = pl.scan_parquet(WORK / f"{data_split(split)}_prep.parquet").select("gid", "name_roman", "addr_roman")
    n = r.select(pl.len()).collect().item()
    out = []
    for col, kind in (("name_roman", "n"), ("addr_roman", "a")):
        df = (r.select("gid", pl.col(col).str.split(" ").list.unique().alias("t")).explode("t")
               .filter(pl.col("t").is_not_null() & (pl.col("t").str.len_chars() >= 2))
               .group_by("t").agg(pl.len().alias("df")).collect())
        out.append(df.select(pl.lit(kind).alias("kind"), "t", (np.log(n) - (pl.col("df") + 1).log()).cast(pl.Float32).alias("idf")))
    res = pl.concat(out)
    res.write_parquet(path)
    return res


def idf_feats(df: pl.DataFrame, split: str) -> pl.DataFrame:
    """L6 (pipeline_optimization_guide): rare-token evidence and PIN conflict for uncertain pairs.
    df: s1, rid, p. Returns s1, rid, i_* (null outside IDF_BAND)."""
    tgt = df.filter(pl.col("p").is_between(*IDF_BAND, closed="none")).select("s1", "rid")
    ids = pl.concat([tgt["s1"], tgt["rid"]]).unique()
    r = (pl.scan_parquet(WORK / f"{data_split(split)}_prep.parquet").select("gid", "name_roman", "addr_roman")
           .filter(pl.col("gid").is_in(ids.implode())).collect())
    idf = token_idf(split)
    feats = []
    for col, kind in (("name_roman", "n"), ("addr_roman", "a")):
        tok = (r.select("gid", pl.col(col).str.split(" ").list.unique().alias("t")).explode("t")
                .join(idf.filter(pl.col("kind") == kind).select("t", "idf"), on="t", how="inner"))
        a = tgt.join(tok.rename({"gid": "s1"}), on="s1")
        b = tgt.join(tok.rename({"gid": "rid"}), on="rid")
        inter = a.join(b.select("s1", "rid", "t"), on=["s1", "rid", "t"]).group_by("s1", "rid").agg(
            pl.col("idf").sum().alias("isum"), pl.col("idf").max().alias("imax"), (pl.col("idf") > 8).sum().alias("nrare"))
        sa = a.group_by("s1", "rid").agg(pl.col("idf").sum().alias("asum"))
        sb = b.group_by("s1", "rid").agg(pl.col("idf").sum().alias("bsum"))
        f = (tgt.join(inter, on=["s1", "rid"], how="left").join(sa, on=["s1", "rid"], how="left").join(sb, on=["s1", "rid"], how="left")
                .with_columns(pl.col("isum", "nrare").fill_null(0), pl.col("asum", "bsum").fill_null(0.0)))
        union = pl.col("asum") + pl.col("bsum") - pl.col("isum")
        feats.append(f.select("s1", "rid",
                              pl.when(union > 0).then(pl.col("isum") / union).otherwise(None).alias(f"i_{kind}_wjac"),
                              pl.when(pl.col("asum") > 0).then(pl.col("isum") / pl.col("asum")).otherwise(None).alias(f"i_{kind}_wcov"),
                              pl.col("imax").alias(f"i_{kind}_rarest"), pl.col("nrare").alias(f"i_{kind}_nrare")))
    pin = r.select("gid", pl.col("addr_roman").str.extract(r"\b([1-9]\d{5})\b", 1).alias("pin"))
    pf = (tgt.join(pin.rename({"gid": "s1", "pin": "pa"}), on="s1", how="left").join(pin.rename({"gid": "rid", "pin": "pb"}), on="rid", how="left")
             .select("s1", "rid", pl.when(pl.col("pa").is_null() | pl.col("pb").is_null()).then(None)
                     .otherwise((pl.col("pa") != pl.col("pb")).cast(pl.Int8)).alias("i_pin_conflict")))
    out = tgt
    for f in feats + [pf]:
        out = out.join(f, on=["s1", "rid"], how="left")
    return out


I_FEATS = ["i_n_wjac", "i_n_wcov", "i_n_rarest", "i_n_nrare", "i_a_wjac", "i_a_wcov", "i_a_rarest", "i_a_nrare", "i_pin_conflict"]
S8_FEATS = S3F_FEATS + ["ce"] + G_FEATS


def v8_frame(base: pl.DataFrame, split: str, ce_path) -> pl.DataFrame:
    """base: s1, rid, p (+label) for a whole split (all competitors). Adds S1-side, record-side, CE, group feats."""
    d = rec_feats(stack_feats(base))
    d = d.join(pl.read_parquet(ce_path), on=["s1", "rid"], how="left")
    return d.join(group_feats(base, split), on=["s1", "rid"], how="left")


def stage3v8():
    """Stage 3 with C1+C2+C3, trained and validated on folds 1-2 of all train S1s (the cross-encoder was trained on
    fold 0, so its scores are out-of-sample here). Validation applies the one-owner rule over ALL folds' p (every
    competitor present) and scores folds 1-2 only."""
    full = pl.concat([pl.read_parquet(PR / "train_oof.parquet")] +
                     [pl.read_parquet(PR / f"train_oof_f{k}.parquet") for k in (1, 2)]).select("s1", "rid", "label", "country", "p")
    s1_meta = pl.concat([pl.read_parquet(f, columns=["s1", "country"]) for f in s1_files("train") + s1_files("trainrest")]).unique("s1")
    ev_meta = s1_meta.filter(fold_of(pl.col("s1")) != 0)
    truth = truth_gids().join(ev_meta.select("s1"), on="s1").select("s1", "rid")
    t = time.time()
    d = v8_frame(full.select("s1", "rid", "p"), "train", CE_DIR / "ce_train.parquet")
    d = d.join(full.select("s1", "rid", "label", "country"), on=["s1", "rid"]).filter(fold_of(pl.col("s1")) != 0)
    files = feat_files("train") + feat_files("trainrest")
    extra = pl.concat([load_feat(f, ["s1", "rid"] + STACK_BASE).filter(fold_of(pl.col("s1")) != 0) for f in files], how="diagonal_relaxed")
    d = d.join(extra, on=["s1", "rid"], how="left")
    del extra
    log(f"stage3v8 [{RUN}] rows {d.height:,} (folds 1-2), {len(S8_FEATS)} features, CE coverage {d['ce'].is_not_null().mean():.1%}, {time.time() - t:.0f}s")

    def test_like(p_f12: pl.DataFrame, thr):  # one-owner rule over all folds, then score folds 1-2
        allp = pl.concat([full.filter(fold_of(pl.col("s1")) == 0).select("s1", "rid", "p"), p_f12.select("s1", "rid", "p")])
        return select_greedy_exclusive(allp, thr).filter(fold_of(pl.col("s1")) != 0)

    res = {}
    for thr in (0.65, 0.7, 0.75, 0.8):
        res[f"greedy{thr}"] = evaluate(test_like(d, thr), f"[v8 eval] v6 stage2 greedy {thr}", truth, ev_meta)
    easy = (pl.col("label") == 0) & (pl.col("p") < 1e-3)
    d = d.with_columns((~easy | ((pl.struct("s1", "rid").hash(SEED) % EASY_KEEP) == 0)).alias("_keep"),
                       pl.when(easy).then(float(EASY_KEEP)).otherwise(1.0).cast(pl.Float32).alias("_w"),
                       (pl.col("s1").hash(SEED + 11) % 3).alias("_ifold"))
    oof3 = np.zeros(d.height, np.float32)
    iters = []
    for f in range(3):
        t = time.time()
        trn = d.filter((pl.col("_ifold") != f) & pl.col("_keep"))
        val = d.filter(pl.col("_ifold") == f)
        bst = lgb.train(LGB3, lgb.Dataset(_X(trn, S8_FEATS), trn["label"].to_numpy(), weight=trn["_w"].to_numpy()), 3000,
                        valid_sets=[lgb.Dataset(_X(val, S8_FEATS), val["label"].to_numpy())],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
        oof3[(d["_ifold"] == f).to_numpy()] = bst.predict(_X(val, S8_FEATS), num_iteration=bst.best_iteration)
        iters.append(bst.best_iteration)
        log(f"  stage3v8 inner fold {f}: best_iter {bst.best_iteration}, logloss {bst.best_score['valid_0']['binary_logloss']:.5f}, {time.time() - t:.0f}s")
        if f == 0:
            imp = sorted(zip(S8_FEATS, bst.feature_importance("gain")), key=lambda z: -z[1])
            tot = sum(g for _, g in imp)
            log("  top features: " + ", ".join(f"{n} {g / tot:.1%}" for n, g in imp[:10]))
        del trn, val
    d3 = d.select("s1", "rid").with_columns(pl.Series("p", oof3))
    for thr in (0.6, 0.65, 0.7, 0.75, 0.8, 0.85):
        res[f"s3greedy{thr}"] = evaluate(test_like(d3, thr), f"[v8 eval] stage3 v8 greedy {thr}", truth, ev_meta)
    (PR / "stage3v8_eval.json").write_text(json.dumps(res, indent=1))
    k = d.filter(pl.col("_keep"))
    lgb.train(LGB3, lgb.Dataset(_X(k, S8_FEATS), k["label"].to_numpy(), weight=k["_w"].to_numpy()),
              int(np.mean(iters) * 1.1)).save_model(str(PR / "stage3.txt"))
    (PR / "stage3_feats.json").write_text(json.dumps(S8_FEATS))
    log("stage3v8 final model saved")


CE_DIR = WORK / "ce"
S9_FEATS = S8_FEATS + ["p_s2"]


LGB3_BIG = dict(LGB3, num_leaves=127, min_data_in_leaf=100, learning_rate=0.05, feature_fraction=0.8)


class FoldAvg:
    """Average of the inner-CV fold models (each at its own best iteration): the exact models validation scored."""
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        return np.mean([m.predict(X, num_iteration=m.best_iteration or None) for m in self.models], axis=0)


def _cv(d, feats, label="label", save=None):
    """Inner 3-fold CV on d (uses _ifold, _keep, _w). Returns (oof predictions, mean best iteration, fold models).
    save: path prefix -> fold models written to <save>_f<k>.txt (reused if all exist, with the saved oof)."""
    if save is not None and all(Path(f"{save}_f{f}.txt").exists() for f in range(3)) and Path(f"{save}_oof.npy").exists():
        models = [lgb.Booster(model_file=f"{save}_f{f}.txt") for f in range(3)]
        oof = np.load(f"{save}_oof.npy")
        if oof.shape[0] == d.height:
            log(f"    inner CV reused from {Path(save).name}_f*.txt")
            return oof, int(np.mean([m.best_iteration for m in models])), models
    big = os.environ.get("BER_S3BIG") == "1"
    nseed = int(os.environ.get("BER_SEEDS", "1"))
    oof = np.zeros(d.height, np.float32)
    its, models = [], []
    for f in range(3):
        trn = d.filter((pl.col("_ifold") != f) & pl.col("_keep"))
        val = d.filter(pl.col("_ifold") == f)
        Xt, yt, wt = _X(trn, feats), trn[label].to_numpy(), trn["_w"].to_numpy()
        Xv, yv = _X(val, feats), val[label].to_numpy()
        pf = np.zeros(val.height, np.float32)
        for sd in range(nseed):  # seed-averaged within the fold; each seed model is kept for the test average
            prm = dict(LGB3_BIG if big else LGB3)
            if sd > 0:  # seed 0 = exactly the original settings; extra seeds only for the averaged copies
                prm.update(seed=SEED + sd, bagging_seed=SEED + sd, feature_fraction_seed=SEED + sd)
            b = lgb.train(prm, lgb.Dataset(Xt, yt, weight=wt), 5000 if big else 3000, valid_sets=[lgb.Dataset(Xv, yv)],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            pf += b.predict(Xv, num_iteration=b.best_iteration) / nseed
            its.append(b.best_iteration)
            models.append(b)
            if save is not None:
                b.save_model(f"{save}_f{f}{'' if sd == 0 else f's{sd}'}.txt", num_iteration=b.best_iteration)
            log(f"    inner fold {f} seed {sd}: best_iter {b.best_iteration}, logloss {b.best_score['valid_0']['binary_logloss']:.5f}")
        oof[(d["_ifold"] == f).to_numpy()] = pf
        del trn, val, Xt, Xv
    if save is not None:
        np.save(f"{save}_oof.npy", oof)
    return oof, int(np.mean(its)), models


def stage4():
    """v9 (Plan_3): second stacking round. Stage-3 OOF p (v8 features) -> recompute S1-side, record-side and group
    features from the sharper p -> stage 4 (+ the original stage-2 p as p_s2). Same inner folds as stage 3.
    Test: stage-3 test scores (cached by `predict s3...`) -> same features -> stage 4 -> output/v9."""
    ALL = os.environ.get("BER_ALLFOLDS") == "1"  # v10: CE available for every fold -> use all 2.2M S1s
    ce_tr, ce_te, oname = (("ce_train_all.parquet", "ce_test_avg.parquet", "v10") if ALL
                           else ("ce_train.parquet", "ce_test.parquet", "v9"))
    if os.environ.get("BER_CE3") == "1":  # v12: three ~2M-pair cross-encoders, each fold scored out-of-fold, test = mean
        ce_tr, ce_te = "ce_train_all3.parquet", "ce_test_avg3.parquet"
        oname = "v12"
    elif os.environ.get("BER_CE_WIDE") == "1":  # cross-encoder scores for the wider band (0.001-0.999)
        ce_tr, ce_te = (("ce_train_all_w.parquet", "ce_test_avg_w.parquet") if ALL      # v11: A+B mix, test = mean
                        else ("ce_train_w_a.parquet", "ce_test_w_a.parquet"))          # v9x: model A everywhere
    inset = pl.lit(True) if ALL else (fold_of(pl.col("s1")) != 0)
    EXTRA = os.environ.get("BER_EXTRA") == "1"  # v11: + L6 rare-token/IDF, PIN conflict, model-vs-CE disagreement
    CEB = os.environ.get("BER_CEB") == "1"  # v13: + large cross-encoder score as an extra feature
    XF = ((I_FEATS + ["dis_ce"]) if EXTRA else []) + (["ce_base"] if CEB else [])
    F3, F4 = S8_FEATS + XF, S9_FEATS + XF
    if EXTRA:
        oname = "v12" if os.environ.get("BER_CE3") == "1" else ("v11" if ALL else "v9x")
    dis = (pl.col("p") - 1 / (1 + (-pl.col("ce")).exp())).abs().alias("dis_ce")
    # stage 3 must be refit (and test re-scored by the refit) whenever its inputs differ from the cached v8 model
    CUT = float(os.environ.get("BER_CUT", "0"))  # adaptive candidate set: keep p1 >= CUT (blocking stage)
    REFIT = ALL or EXTRA or os.environ.get("BER_CE_WIDE") == "1" or CUT > 0
    if CUT > 0:
        oname = f"{oname}_cut{CUT:g}"
    FOLDAVG = os.environ.get("BER_FOLDAVG") == "1"  # test = average of the validated fold models (no refit)
    if CEB:
        oname = "v13"
    if os.environ.get("BER_S3BIG") == "1":
        oname = f"{oname}_big{os.environ.get('BER_SEEDS', '1')}"
    if FOLDAVG:
        oname = f"{oname}_fa"
        REFIT = True  # stage-3 test scores must come from these fold models, never from a cached refit
    full = pl.concat([pl.read_parquet(PR / "train_oof.parquet")] +
                     [pl.read_parquet(PR / f"train_oof_f{k}.parquet") for k in (1, 2)]).select("s1", "rid", "label", "country", "p")
    s1_meta = pl.concat([pl.read_parquet(f, columns=["s1", "country"]) for f in s1_files("train") + s1_files("trainrest")]).unique("s1")
    ev_meta = s1_meta.filter(inset)  # every S1 is evaluated, also those left with no candidate after the cut
    if CUT > 0:
        p1tr = pl.concat([pl.read_parquet(f, columns=["s1", "rid", "p1"]).with_columns(pl.col("p1").cast(pl.Float32))
                          for f in s1_files("train") + s1_files("trainrest")])
        n0 = full.height
        full = full.join(p1tr, on=["s1", "rid"]).filter(pl.col("p1") >= CUT).drop("p1")
        del p1tr
        log(f"stage4: candidate cut p1 >= {CUT}: train pairs {n0:,} -> {full.height:,} ({full.height / max(1, full['s1'].n_unique()):.1f}/S1)")
    truth = truth_gids().join(ev_meta.select("s1"), on="s1").select("s1", "rid")
    f0 = full.filter(~inset).select("s1", "rid", "p")
    files = feat_files("train") + feat_files("trainrest")
    extra = (pl.concat([load_feat(f, ["s1", "rid"] + STACK_BASE).filter(inset) for f in files], how="diagonal_relaxed")
               .with_columns(pl.col(pl.Float64).cast(pl.Float32)))
    easy_flags = lambda d: d.with_columns(
        (~((pl.col("label") == 0) & (pl.col("p_s2") < 1e-3)) | ((pl.struct("s1", "rid").hash(SEED) % EASY_KEEP) == 0)).alias("_keep"),
        pl.when((pl.col("label") == 0) & (pl.col("p_s2") < 1e-3)).then(float(EASY_KEEP)).otherwise(1.0).cast(pl.Float32).alias("_w"),
        (pl.col("s1").hash(SEED + 11) % 3).alias("_ifold"))

    def evaluate_rule(d3, tag):
        best = None
        for thr in (0.6, 0.65, 0.7, 0.75, 0.8):
            r = evaluate(select_greedy_exclusive(pl.concat([f0, d3.select("s1", "rid", "p")]), thr).filter(inset),
                         f"[{tag}] greedy {thr}", truth, ev_meta)
            if best is None or r["test_mix_f05"] > best[1]["test_mix_f05"]:
                best = (thr, r)
        return best

    # ---- stage 3 again (same as stage3v8), keeping its out-of-fold p this time
    t = time.time()
    d = v8_frame(full.select("s1", "rid", "p"), "train", CE_DIR / ce_tr)
    d = (d.join(full.select("s1", "rid", "label", "country", pl.col("p").alias("p_s2")), on=["s1", "rid"])
          .filter(inset).join(extra, on=["s1", "rid"], how="left"))
    if EXTRA:
        d = d.join(idf_feats(full.select("s1", "rid", "p"), "train"), on=["s1", "rid"], how="left").with_columns(dis)
    if CEB:
        d = d.join(pl.read_parquet(CE_DIR / "cb_train_all.parquet"), on=["s1", "rid"], how="left")
    d = easy_flags(d).with_columns(pl.col(pl.Float64).cast(pl.Float32))
    log(f"stage4: stage-3 frame {d.height:,} rows, {time.time() - t:.0f}s")
    ck3 = PR / f"stage3_oof_{oname}.npz"
    if not FOLDAVG and ck3.exists() and np.load(ck3)["p3"].shape[0] == d.height:
        z = np.load(ck3); p3, it3 = z["p3"], int(z["it3"])
        log(f"stage4: stage-3 out-of-fold scores reused from {ck3.name}")
    else:
        p3, it3, m3s = _cv(d, F3, save=str(PR / f"s3cv_{oname}") if FOLDAVG else None)
        np.savez(ck3, p3=p3, it3=it3)
    if FOLDAVG:
        b3m = FoldAvg(m3s)
    elif REFIT:  # refit stage 3 on the training rows; its test scores are recomputed below
        m3 = PR / f"stage3_{oname}.txt"
        if m3.exists() and m3.stat().st_mtime > ck3.stat().st_mtime:
            b3m = lgb.Booster(model_file=str(m3))
        else:
            if ALL:  # 55M rows: zero-weight refit (no filtered copy) to fit in RAM
                X3 = _X(d, F3)
                w3 = np.where(d["_keep"].to_numpy(), d["_w"].to_numpy(), 0.0).astype(np.float32)
                ds3 = lgb.Dataset(X3, d["label"].to_numpy(), weight=w3, free_raw_data=True).construct()
                del X3, w3
            else:    # v8/v9's original refit on the kept rows (LB-proven)
                k3 = d.filter(pl.col("_keep"))
                ds3 = lgb.Dataset(_X(k3, F3), k3["label"].to_numpy(), weight=k3["_w"].to_numpy(), free_raw_data=True).construct()
                del k3
            b3m = lgb.train(LGB3, ds3, int(it3 * 1.1))
            del ds3
            b3m.save_model(str(m3))
    s3 = d.select("s1", "rid", "label", "country", "ce", "p_s2", "_keep", "_w", "_ifold", *[c for c in I_FEATS if EXTRA],
                  *(["ce_base"] if CEB else [])).with_columns(pl.Series("p", p3))
    del d
    b3 = evaluate_rule(s3, "stage3 (v8)")
    # ---- stage 4: features recomputed from stage-3 p (fold-0 pairs keep their stage-2 p as competitors)
    base = pl.concat([f0, s3.select("s1", "rid", "p")])
    d4 = rec_feats(stack_feats(base)).join(group_feats(base, "train"), on=["s1", "rid"], how="left").filter(inset)
    d4 = (d4.join(s3.drop("p"), on=["s1", "rid"]).join(extra, on=["s1", "rid"], how="left")
            .with_columns(pl.col(pl.Float64).cast(pl.Float32)))
    if EXTRA:
        d4 = d4.with_columns(dis)
    del extra, base
    log(f"stage4: frame {d4.height:,} rows, {len(S9_FEATS)} features")
    p4, it4, m4s = _cv(d4, F4, save=str(PR / f"s4cv_{oname}") if FOLDAVG else None)
    b4 = evaluate_rule(d4.select("s1", "rid").with_columns(pl.Series("p", p4)), "stage4 (v9)")
    # keep the out-of-fold scores (decision-rule search, error analysis) — stage 3 and stage 4
    (d4.select("s1", "rid", "label", "country").with_columns(pl.Series("p4", p4))
       .join(s3.select("s1", "rid", pl.col("p").alias("p3"), "p_s2", "ce"), on=["s1", "rid"], how="left")
       .write_parquet(PR / f"train_oof34_{oname}.parquet"))
    res = {"stage3": {"thr": b3[0], **b3[1]}, "stage4": {"thr": b4[0], **b4[1]}}
    (PR / f"stage4_eval_{oname}.json").write_text(json.dumps(res, indent=1))
    if FOLDAVG:
        ds4 = None
    elif ALL:
        X4 = _X(d4, F4)
        w4 = np.where(d4["_keep"].to_numpy(), d4["_w"].to_numpy(), 0.0).astype(np.float32)
        ds4 = lgb.Dataset(X4, d4["label"].to_numpy(), weight=w4, free_raw_data=True).construct()
        del X4, w4
    else:  # original (v9) refit on kept rows
        k4 = d4.filter(pl.col("_keep"))
        ds4 = lgb.Dataset(_X(k4, F4), k4["label"].to_numpy(), weight=k4["_w"].to_numpy(), free_raw_data=True).construct()
        del k4
    if FOLDAVG:
        b = FoldAvg(m4s)
    else:
        b = lgb.train(LGB3, ds4, int(it4 * 1.1))
        del ds4
    if not FOLDAVG:
        b.save_model(str(PR / f"stage4_{oname}.txt"))
    del d4
    log(f"stage4 validation: stage3 {b3[1]['macro_f05']:.4f} (thr {b3[0]}) -> stage4 {b4[1]['macro_f05']:.4f} (thr {b4[0]})")
    # ---- test
    if os.environ.get("BER_SKIP_TEST") == "1":  # score the test set with `predict4 <oname>` (per country, low RAM)
        log(f"stage4 {oname}: models saved; test scoring skipped (run predict4 {oname})")
        return
    t2 = pl.read_parquet(PR / "test_scores.parquet")
    if CUT > 0:
        tp1 = pl.concat([pl.read_parquet(f, columns=["s1", "rid", "p1"]) for f in sorted(P.glob("test_s1_*.parquet"))])
        n0 = t2.height
        t2 = t2.join(tp1.with_columns(pl.col("p1").cast(pl.Float32)), on=["s1", "rid"]).filter(pl.col("p1") >= CUT).drop("p1")
        del tp1
        log(f"stage4: candidate cut p1 >= {CUT}: test pairs {n0:,} -> {t2.height:,} ({t2.height / t2['s1'].n_unique():.1f}/S1 with candidates)")
    textra = pl.concat([load_feat(f, ["s1", "rid"] + STACK_BASE) for f in feat_files("test")], how="diagonal_relaxed")
    tidf = idf_feats(t2, "test") if EXTRA else None
    if REFIT:  # stage-3 test scores from the refitted stage-3 model (same inputs as in training)
        f3 = v8_frame(t2, "test", CE_DIR / ce_te).join(textra, on=["s1", "rid"], how="left")
        if EXTRA:
            f3 = f3.join(tidf, on=["s1", "rid"], how="left").with_columns(dis)
        t3 = f3.select("s1", "rid").with_columns(pl.Series("p", b3m.predict(_X(f3, F3)).astype(np.float32)))
        del f3
    else:
        t3 = pl.read_parquet(PR / "test_scores3.parquet")
    tt = (v8_frame(t3, "test", CE_DIR / ce_te).join(t2.rename({"p": "p_s2"}), on=["s1", "rid"], how="left")
            .join(textra, on=["s1", "rid"], how="left"))
    if EXTRA:
        tt = tt.join(tidf, on=["s1", "rid"], how="left").with_columns(dis)
    del textra
    out = tt.select("s1", "rid").with_columns(pl.Series("p", b.predict(_X(tt, F4)).astype(np.float32)))
    out.write_parquet(PR / f"test_scores4_{oname}.parquet")
    del tt
    sel = select_greedy_exclusive(out, b4[0])
    ids = prepared("test", ["gid", "entity_id", "src"])
    to_id = lambda x: (x.join(ids.select(pl.col("gid").alias("s1"), pl.col("entity_id").alias("s1_id")), on="s1")
                        .join(ids.select(pl.col("gid").alias("rid"), pl.col("entity_id").alias("rid_id")), on="rid")
                        .select(pl.col("s1_id").alias("s1"), pl.col("rid_id").alias("rid")))
    s1_ids = ids.filter(pl.col("src") == 1)["entity_id"]
    od = OUT / oname
    od.mkdir(exist_ok=True)
    write_lists(to_id(sel), s1_ids, od / "matching_results.tsv", "matched_entity_ids")
    write_lists(to_id(out.select("s1", "rid")), s1_ids, od / "candidate_pairs.tsv", "candidate_entity_ids")
    log(f"stage4: wrote output/{oname} ({sel.height:,} matches, {sel.height / s1_ids.len():.2f}/S1, threshold {b4[0]})")


def predict4(oname="v9x", thr="0.7"):
    """Test scoring only, from the saved models of a finished stage4() run (no retraining), one country at a time
    (exact: features are per-S1 or per-record, and records never cross countries — keeps RAM low).
    Models: fold-averaged s3cv_/s4cv_<oname>_f*.txt when present (BER_FOLDAVG runs), else stage3_/stage4_<oname>.txt.
    Setting inferred from oname: v11* = all folds + averaged A/B CE; v9x* = model-A CE; both with IDF features."""
    thr = float(thr)
    XF = I_FEATS + ["dis_ce"] + (["ce_base"] if oname.startswith("v13") else [])
    F3, F4 = S8_FEATS + XF, S9_FEATS + XF
    ceb = pl.read_parquet(CE_DIR / "cb_test_avg.parquet") if oname.startswith("v13") else None
    ce_te = CE_DIR / ("ce_test_avg3.parquet" if oname.startswith(("v12", "v13", "v14")) else
                      "ce_test_avg_w.parquet" if oname.startswith("v11") else "ce_test_w_a.parquet")
    if (PR / f"s3cv_{oname}_f0.txt").exists():
        load = lambda st: FoldAvg([lgb.Booster(model_file=str(f)) for f in sorted(PR.glob(f"{st}cv_{oname}_f*.txt"))])
        b3m, b = load("s3"), load("s4")
        log(f"predict4 {oname}: {len(b3m.models)} stage-3 and {len(b.models)} stage-4 models")
        log(f"predict4 {oname}: fold-averaged models")
    else:
        b3m = lgb.Booster(model_file=str(PR / f"stage3_{oname}.txt"))
        b = lgb.Booster(model_file=str(PR / f"stage4_{oname}.txt"))
    dis = (pl.col("p") - 1 / (1 + (-pl.col("ce")).exp())).abs().alias("dis_ce")
    ctry = prepared("test", ["gid", "country"]).select(pl.col("gid").alias("s1"), "country")
    t2all = pl.read_parquet(PR / "test_scores.parquet").join(ctry, on="s1")
    outs = []
    for c in t2all["country"].unique().sort().to_list():
        t2 = t2all.filter(pl.col("country") == c).drop("country")
        textra = (pl.concat([load_feat(f, ["s1", "rid"] + STACK_BASE) for f in feat_files("test") if f.name.split("_")[1] == c],
                            how="diagonal_relaxed").with_columns(pl.col(pl.Float64).cast(pl.Float32)))
        tidf = idf_feats(t2, "test")
        f3 = v8_frame(t2, "test", ce_te).join(textra, on=["s1", "rid"], how="left").join(tidf, on=["s1", "rid"], how="left").with_columns(dis)
        if ceb is not None:
            f3 = f3.join(ceb, on=["s1", "rid"], how="left")
        t3 = f3.select("s1", "rid").with_columns(pl.Series("p", b3m.predict(_X(f3, F3)).astype(np.float32)))
        del f3
        tt = (v8_frame(t3, "test", ce_te).join(t2.rename({"p": "p_s2"}), on=["s1", "rid"], how="left")
                .join(textra, on=["s1", "rid"], how="left").join(tidf, on=["s1", "rid"], how="left").with_columns(dis))
        if ceb is not None:
            tt = tt.join(ceb, on=["s1", "rid"], how="left")
        outs.append(tt.select("s1", "rid").with_columns(pl.Series("p", b.predict(_X(tt, F4)).astype(np.float32))))
        del tt, textra, tidf, t3
        log(f"predict4 {oname}: {c} scored ({outs[-1].height:,} pairs)")
    out = pl.concat(outs)
    del outs, t2all
    out.write_parquet(PR / f"test_scores4_{oname}.parquet")
    sel = select_greedy_exclusive(out, thr)
    ids = prepared("test", ["gid", "entity_id", "src"])
    to_id = lambda x: (x.join(ids.select(pl.col("gid").alias("s1"), pl.col("entity_id").alias("s1_id")), on="s1")
                        .join(ids.select(pl.col("gid").alias("rid"), pl.col("entity_id").alias("rid_id")), on="rid")
                        .select(pl.col("s1_id").alias("s1"), pl.col("rid_id").alias("rid")))
    s1_ids = ids.filter(pl.col("src") == 1)["entity_id"]
    od = OUT / oname
    od.mkdir(exist_ok=True)
    write_lists(to_id(sel), s1_ids, od / "matching_results.tsv", "matched_entity_ids")
    write_lists(to_id(out.select("s1", "rid")), s1_ids, od / "candidate_pairs.tsv", "candidate_entity_ids")
    log(f"predict4: wrote output/{oname} ({sel.height:,} matches, {sel.height / s1_ids.len():.2f}/S1, threshold {thr})")


def apply_rule(df, rule):
    """Decision rules by the keys stage2() evaluates: thr<t>, greedy<t>, expF, expF_m<margin>.
    Per-country thresholds (test only): 'greedy0.7|France:0.85' = one-owner rule, then 0.85 for France S1s,
    0.7 elsewhere (an unseen country needs a stricter cut-off; Plan_2 cold-country test)."""
    if "|" in rule:
        base, *over = rule.split("|")
        t0 = float(base.removeprefix("greedy"))
        ctry = prepared("test", ["gid", "country"]).select(pl.col("gid").alias("s1"), "country")
        tmap = {c: float(t) for c, t in (o.split(":") for o in over)}
        best = df.filter(pl.col("p") == pl.col("p").max().over("rid")).join(ctry, on="s1", how="left")
        thr = pl.col("country").replace_strict(tmap, default=t0, return_dtype=pl.Float64)
        return best.filter(pl.col("p") >= thr).select("s1", "rid")
    if rule.startswith("thr"):
        return select_threshold(df, float(rule[3:]))
    if rule.startswith("greedy"):
        return select_greedy_exclusive(df, float(rule[6:]))
    if rule.startswith("expF_m"):
        return select_expected_f(column_normalize(df, margin=float(rule[6:])))
    return select_expected_f(column_normalize(df))


def best_rule():
    res = json.loads((PR / "stage2_eval.json").read_text())
    return max(res, key=lambda k: res[k]["test_mix_f05"])


def predict(rule=None):
    rule = rule or best_rule()
    out_dir = OUTR if "|" not in rule else OUT / f"{RUN or 'v2'}_fr{rule.split(':')[-1]}"
    feats = json.loads((PR / "feats.json").read_text())
    cache = PR / "test_scores.parquet"  # delete it after retraining stage 2
    if cache.exists() and cache.stat().st_mtime > (PR / "stage2_backend.txt").stat().st_mtime:
        df = pl.read_parquet(cache)
        log(f"loaded cached scores for {df.height:,} test pairs")
    else:
        score = load_final()
        scored = []  # score one chunk file at a time; keep only (s1, rid, p)
        for f in feat_files("test"):
            d = load_feat(f, ["s1", "rid"] + feats)
            scored.append(d.select("s1", "rid").with_columns(pl.Series("p", score(_X(d, feats)).astype(np.float32))))
            del d
        df = pl.concat(scored)
        del scored
        df.write_parquet(cache)
        log(f"scored {df.height:,} test pairs")
    if rule.startswith("s3"):  # stacked stage-3 probabilities (cached like stage 2)
        cache3 = PR / "test_scores3.parquet"
        if cache3.exists() and cache3.stat().st_mtime > (PR / "stage3.txt").stat().st_mtime:
            df = pl.read_parquet(cache3)
        else:
            extra = pl.concat([load_feat(f, ["s1", "rid"] + STACK_BASE) for f in feat_files("test")], how="diagonal_relaxed")
            d = stack_feats(df.join(extra, on=["s1", "rid"], how="left"))
            del extra
            f3 = PR / "stage3_feats.json"
            feats3 = json.loads(f3.read_text()) if f3.exists() else S3_FEATS
            if "r_rank" in feats3:
                d = rec_feats(d)
            if "ce" in feats3:  # v8: cross-encoder logit + group consistency
                d = d.join(pl.read_parquet(CE_DIR / "ce_test.parquet"), on=["s1", "rid"], how="left")
                d = d.join(group_feats(df.select("s1", "rid", "p"), "test"), on=["s1", "rid"], how="left")
            b3 = lgb.Booster(model_file=str(PR / "stage3.txt"))
            df = d.select("s1", "rid").with_columns(pl.Series("p", b3.predict(_X(d, feats3)).astype(np.float32)))
            del d
            df.write_parquet(cache3)
        log(f"stage-3 scores for {df.height:,} test pairs")
        rule, out_dir = rule[2:], OUT / os.environ.get("BER_OUT_NAME", f"{RUN or 'v2'}_s3")
    log(f"decision rule: {rule}")
    sel = apply_rule(df, rule)
    ids = prepared("test", ["gid", "entity_id", "src"])
    to_id = lambda d: (d.join(ids.select(pl.col("gid").alias("s1"), pl.col("entity_id").alias("s1_id")), on="s1")
                        .join(ids.select(pl.col("gid").alias("rid"), pl.col("entity_id").alias("rid_id")), on="rid")
                        .select(pl.col("s1_id").alias("s1"), pl.col("rid_id").alias("rid")))
    s1_ids = ids.filter(pl.col("src") == 1)["entity_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    write_lists(to_id(sel), s1_ids, out_dir / "matching_results.tsv", "matched_entity_ids")
    write_lists(to_id(df.select("s1", "rid")), s1_ids, out_dir / "candidate_pairs.tsv", "candidate_entity_ids")
    log(f"wrote submission: {sel.height:,} matches, {sel.height / s1_ids.len():.2f}/S1")


if __name__ == "__main__":
    fn = {"stage1": stage1, "features": features, "features2": features2, "stage2": stage2, "stage3": stage3,
          "pseudo": pseudo, "stage1rest": stage1_rest, "stage2full": stage2full, "stage2full_oof": stage2full_oof,
          "stage3full": stage3full, "stage3v8": stage3v8, "stage4": stage4, "predict4": predict4,
          "predict": predict}[sys.argv[1]]
    fn(*sys.argv[2:])  # predict optionally takes a rule key, e.g. `predict greedy0.5`
