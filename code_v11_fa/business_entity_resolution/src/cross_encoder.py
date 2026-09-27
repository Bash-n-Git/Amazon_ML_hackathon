"""Cross-encoder for uncertain pairs (Plan_3). Ditto-style: both records serialised as
'name: ... address: ...' (original scripts), read jointly by a pretrained multilingual transformer
(intfloat/multilingual-e5-small, MIT) with a 1-logit match head.

Only pairs in the stage-2 uncertainty band (BAND_LO < p < BAND_HI) are trained on and scored — ~4-6% of pairs.

Leakage-free design:
  train  : band pairs of fold-0 S1s (v6 out-of-fold p), 5% of those S1s held out for monitoring
  score  : band pairs of folds 1-2 (out-of-sample for the cross-encoder) and of the test set
Stage 3 then learns from folds 1-2 only, where the cross-encoder score is honest.

Usage (BER_RUN=v6):  python cross_encoder.py bench | train | score_test | score_train
"""
import config  # noqa: F401  (first: points temp/cache dirs at the project drive)
import math
import os
import sys
import time

import numpy as np
import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import SEED, WORK
from pipeline import PR, fold_of, log

MODEL = os.environ.get("BER_CE_MODEL", "intfloat/multilingual-e5-small")
BAND_LO, BAND_HI = 0.01, 0.99
MAX_LEN = int(os.environ.get("BER_CE_MAXLEN", "96"))
BS_TRAIN, BS_EVAL = 128, 512
LR, EPOCHS, WARMUP = 5e-5, 1, 0.05
CE = WORK / "ce"
CE.mkdir(exist_ok=True)
DEV = "cuda"


def texts(pairs: pl.DataFrame, split: str):
    """(text_a, text_b) lists for (s1, rid) pairs, from the raw name/address of the prepared records."""
    ids = pl.concat([pairs["s1"], pairs["rid"]]).unique()
    r = (pl.scan_parquet(WORK / f"{split}_prep.parquet").select("gid", "name", "addr")
           .filter(pl.col("gid").is_in(ids.implode())).collect())
    ser = r.select("gid", ("name: " + pl.col("name") + " address: " + pl.col("addr")).alias("t"))
    d = (pairs.select("s1", "rid").join(ser.rename({"gid": "s1", "t": "a"}), on="s1", how="left")
              .join(ser.rename({"gid": "rid", "t": "b"}), on="rid", how="left"))
    return d["a"].to_list(), d["b"].to_list()


def encode(tok, a, b, chunk=200_000):
    """Tokenise pairs into a compact padded int32 matrix (N, MAX_LEN) + lengths — no per-pair Python lists kept.
    Only input_ids/attention_mask are used (token_type_ids stay zero, as in the first cross-encoder)."""
    ids = np.full((len(a), MAX_LEN), tok.pad_token_id, np.int32)
    lens = np.empty(len(a), np.int16)
    for i in range(0, len(a), chunk):
        e = tok(a[i:i + chunk], b[i:i + chunk], truncation=True, max_length=MAX_LEN, padding=False)["input_ids"]
        for j, x in enumerate(e):
            ids[i + j, :len(x)] = x
            lens[i + j] = len(x)
    return ids, lens


def batches(enc, idx, bs, tok, labels=None):
    """Slice the compact matrix; trim each batch to its longest pair (dynamic padding without Python work)."""
    ids, lens = enc
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        L = int(lens[j].max())
        x = torch.from_numpy(ids[j, :L].astype(np.int64)).to(DEV, non_blocking=True)
        batch = {"input_ids": x, "attention_mask": (x != tok.pad_token_id).long()}
        yield batch, (None if labels is None else torch.tensor(labels[j], dtype=torch.float32, device=DEV))


@torch.no_grad()
def predict(model, tok, a, b):
    model.eval()
    enc = encode(tok, a, b)
    order = np.argsort(enc[1], kind="stable")  # length-sorted for speed
    out = np.empty(len(a), np.float32)
    for (batch, _), i in zip(batches(enc, order, BS_EVAL, tok), range(0, len(order), BS_EVAL)):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out[order[i:i + BS_EVAL]] = model(**batch).logits.float().squeeze(-1).cpu().numpy()
    return out


def band(df):
    return df.filter(pl.col("p").is_between(BAND_LO, BAND_HI, closed="none"))


WIDE_LO, WIDE_HI = 0.001, 0.999


def band_wide(df):
    """The full scoring band 0.001 < p < 0.999 (original + extra)."""
    return df.filter(pl.col("p").is_between(WIDE_LO, WIDE_HI, closed="none"))


def band_extra(df):
    """Pairs in the wider band but outside the original one: (0.001, 0.01] and [0.99, 0.999)."""
    return df.filter(pl.col("p").is_between(WIDE_LO, WIDE_HI, closed="none") & ~pl.col("p").is_between(BAND_LO, BAND_HI, closed="none"))


def train_oof_scores(folds):
    """Stage-2 out-of-fold p for the given folds of all train S1s."""
    src = {0: PR / "train_oof.parquet", 1: PR / "train_oof_f1.parquet", 2: PR / "train_oof_f2.parquet"}
    return pl.concat([pl.read_parquet(src[k]) for k in folds])


def train(n_limit=None, folds=(0,), name="model", cap=None):
    torch.manual_seed(SEED)
    oof = band(train_oof_scores(folds))
    if cap and oof.height > cap:  # random subsample of training pairs (by business, deterministic)
        oof = oof.filter((pl.col("s1").hash(SEED + 3) % 1000) < int(1000 * cap / oof.height))
    hold = (pl.col("s1").hash(SEED + 7) % 20) == 0
    tr, va = oof.filter(~hold), oof.filter(hold)
    if n_limit:
        tr, va = tr.head(n_limit), va.head(n_limit // 10)
    log(f"cross-encoder train: {tr.height:,} pairs ({tr['label'].mean():.1%} positive), monitor {va.height:,}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1).to(DEV)
    a, b = texts(tr, "train")
    enc = encode(tok, a, b)
    y = tr["label"].to_numpy().astype(np.float32)
    va_a, va_b = texts(va, "train")
    steps = EPOCHS * math.ceil(len(y) / BS_TRAIN)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / max(1, WARMUP * steps)) * max(0.0, (steps - s) / max(1, steps * (1 - WARMUP))))
    lossf = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(SEED)
    t = time.time()
    step = 0
    for ep in range(EPOCHS):
        model.train()
        idx = rng.permutation(len(y))
        lens = enc[1]
        blk = BS_TRAIN * 50  # sort within blocks => little padding, still random across blocks
        idx = np.concatenate([b[np.argsort(lens[b], kind="stable")] for b in np.array_split(idx, max(1, len(idx) // blk))])
        border = rng.permutation(len(idx) // BS_TRAIN + 1)
        idx = np.concatenate([idx[i * BS_TRAIN:(i + 1) * BS_TRAIN] for i in border])
        for batch, lab in batches(enc, idx, BS_TRAIN, tok, y):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(**batch).logits.float().squeeze(-1)
            loss = lossf(logit, lab)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            if step % 500 == 0 or step == steps:
                log(f"  step {step}/{steps}  loss {loss.item():.4f}  {step * BS_TRAIN / (time.time() - t):,.0f} pairs/s")
    pv = predict(model, tok, va_a, va_b)
    yv = va["label"].to_numpy()
    pp = 1 / (1 + np.exp(-pv))
    ll = float(-np.mean(yv * np.log(np.clip(pp, 1e-7, 1)) + (1 - yv) * np.log(np.clip(1 - pp, 1e-7, 1))))
    base = va["p"].to_numpy()
    llb = float(-np.mean(yv * np.log(np.clip(base, 1e-7, 1)) + (1 - yv) * np.log(np.clip(1 - base, 1e-7, 1))))
    acc, accb = float(((pp > 0.5) == yv).mean()), float(((base > 0.5) == yv).mean())
    log(f"cross-encoder monitor (band pairs, unseen S1s): logloss {ll:.4f} vs stage-2 {llb:.4f}; "
        f"accuracy {acc:.4f} vs stage-2 {accb:.4f}; {time.time() - t:.0f}s")
    if not n_limit:
        model.save_pretrained(CE / name); tok.save_pretrained(CE / name)


def score(split, name="model", folds=(1, 2), dest=None, sel=None):
    """Band pairs -> ce logit. split=test: v6 test scores. split=train: the given folds' out-of-fold scores."""
    tok = AutoTokenizer.from_pretrained(CE / name)
    model = AutoModelForSequenceClassification.from_pretrained(CE / name).to(DEV)
    sel = sel or band
    if split == "test":
        d = sel(pl.read_parquet(PR / "test_scores.parquet"))
        data = "test"
    else:
        d = sel(train_oof_scores(folds))
        data = "train"
    t = time.time()
    out = []
    step = 500_000
    name_out = dest or f"ce_{split}.parquet"
    parts = CE / "parts"
    parts.mkdir(exist_ok=True)
    d = d.sort(["s1", "rid"])  # deterministic chunking => chunk files can be reused after an interruption
    for i in range(0, d.height, step):
        ck = parts / f"{name_out}.{i // step:03d}"
        if ck.exists():
            out.append(pl.read_parquet(ck))
            log(f"  ce score {split}: chunk {i // step} reused from disk")
            continue
        part = d.slice(i, step)
        a, b = texts(part, data)
        res = part.select("s1", "rid").with_columns(pl.Series("ce", predict(model, tok, a, b)))
        res.write_parquet(ck)  # checkpoint every chunk
        out.append(res)
        log(f"  ce score {split}: {min(i + step, d.height):,}/{d.height:,} pairs, {time.time() - t:.0f}s")
    pl.concat(out).write_parquet(CE / name_out)
    log(f"cross-encoder scored {d.height:,} {split} band pairs")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "bench":
        train(n_limit=20_000)
    elif cmd == "train":
        train()
    elif cmd == "train_b":            # v10: model B on folds 1-2 (2x the data of model A)
        train(folds=(1, 2), name="model_b")
    elif cmd == "score_fold0_b":      # B scores fold 0 (out-of-sample for B)
        score("train", name="model_b", folds=(0,), dest="ce_train_f0_b.parquet")
    elif cmd == "score_test_b":
        score("test", name="model_b", dest="ce_test_b.parquet")
    elif cmd == "combine":            # all-fold train CE (fold 0 from B, folds 1-2 from A); test = mean(A, B)
        tr = pl.concat([pl.read_parquet(CE / "ce_train.parquet"), pl.read_parquet(CE / "ce_train_f0_b.parquet")])
        tr.write_parquet(CE / "ce_train_all.parquet")
        te = (pl.read_parquet(CE / "ce_test.parquet").join(pl.read_parquet(CE / "ce_test_b.parquet").rename({"ce": "ce_b"}),
              on=["s1", "rid"], how="full", coalesce=True)
              .select("s1", "rid", pl.mean_horizontal("ce", "ce_b").alias("ce")))
        te.write_parquet(CE / "ce_test_avg.parquet")
        log(f"combined: train {tr.height:,} pairs, test {te.height:,} pairs")
    elif cmd == "score_wide":         # v11: extra band pairs, out-of-fold (fold 0 by B, folds 1-2 by A), test by both
        score("train", name="model_b", folds=(0,), dest="cex_train_f0_b.parquet", sel=band_extra)
        score("train", name="model", folds=(1, 2), dest="cex_train_f12_a.parquet", sel=band_extra)
        score("test", name="model", dest="cex_test_a.parquet", sel=band_extra)
        score("test", name="model_b", dest="cex_test_b.parquet", sel=band_extra)
    elif cmd == "train_fold":         # v12: `train_fold <name> <folds>` e.g. train_fold model_c02 0,2 (batch 256)
        BS_TRAIN, LR = 256, 7e-5
        train(folds=tuple(int(x) for x in sys.argv[3].split(",")), name=sys.argv[2])
    elif cmd == "train_fold_cap":     # v13: `train_fold_cap <name> <folds> <cap>` (bigger model, capped pairs)
        BS_TRAIN, LR = 256, 5e-5
        train(folds=tuple(int(x) for x in sys.argv[3].split(",")), name=sys.argv[2], cap=int(sys.argv[4]))
    elif cmd == "score_band_fold":    # `score_band_fold <name> <fold> <dest>`: core band 0.01-0.99 only
        score("train", name=sys.argv[2], folds=(int(sys.argv[3]),), dest=sys.argv[4], sel=band)
    elif cmd == "score_band_test":
        score("test", name=sys.argv[2], dest=sys.argv[3], sel=band)
    elif cmd == "combine_base":       # v13: ce_base = large cross-encoder, out-of-fold per fold; test = mean of 3
        tr = pl.concat([pl.read_parquet(CE / f"cb_train_f{k}.parquet") for k in (0, 1, 2)]).rename({"ce": "ce_base"})
        tr.write_parquet(CE / "cb_train_all.parquet")
        te = (pl.read_parquet(CE / "cb_test_m0.parquet").rename({"ce": "c0"})
                .join(pl.read_parquet(CE / "cb_test_m1.parquet").rename({"ce": "c1"}), on=["s1", "rid"])
                .join(pl.read_parquet(CE / "cb_test_m2.parquet").rename({"ce": "c2"}), on=["s1", "rid"])
                .select("s1", "rid", pl.mean_horizontal("c0", "c1", "c2").alias("ce_base")))
        te.write_parquet(CE / "cb_test_avg.parquet")
        log(f"combined large cross-encoder: train {tr.height:,} pairs, test {te.height:,} pairs")
    elif cmd == "score_wide_fold":    # `score_wide_fold <name> <fold> <dest>`: out-of-fold scores, full 0.001-0.999 band
        score("train", name=sys.argv[2], folds=(int(sys.argv[3]),), dest=sys.argv[4], sel=band_wide)
    elif cmd == "score_wide_test":    # `score_wide_test <name> <dest>`
        score("test", name=sys.argv[2], dest=sys.argv[3], sel=band_wide)
    elif cmd == "combine3":           # v12: every fold scored by a ~2M-pair model that never saw it; test = mean of 3
        f0 = pl.concat([pl.read_parquet(CE / "ce_train_f0_b.parquet"), pl.read_parquet(CE / "cex_train_f0_b.parquet")])
        tr = pl.concat([f0, pl.read_parquet(CE / "ce3_train_f1.parquet"), pl.read_parquet(CE / "ce3_train_f2.parquet")])
        tr.write_parquet(CE / "ce_train_all3.parquet")
        tb = pl.concat([pl.read_parquet(CE / "ce_test_b.parquet"), pl.read_parquet(CE / "cex_test_b.parquet")]).rename({"ce": "c0"})
        te = (tb.join(pl.read_parquet(CE / "ce3_test_c02.parquet").rename({"ce": "c1"}), on=["s1", "rid"])
                .join(pl.read_parquet(CE / "ce3_test_c01.parquet").rename({"ce": "c2"}), on=["s1", "rid"])
                .select("s1", "rid", pl.mean_horizontal("c0", "c1", "c2").alias("ce")))
        te.write_parquet(CE / "ce_test_avg3.parquet")
        log(f"combined 3 models: train {tr.height:,} pairs, test {te.height:,} pairs")
    elif cmd == "combine_wide_a":     # v9x: model A only (consistent train/test, like v8/v9): folds 1-2 + test
        tr = pl.concat([pl.read_parquet(CE / "ce_train.parquet"), pl.read_parquet(CE / "cex_train_f12_a.parquet")])
        tr.write_parquet(CE / "ce_train_w_a.parquet")
        te = pl.concat([pl.read_parquet(CE / "ce_test.parquet"), pl.read_parquet(CE / "cex_test_a.parquet")])
        te.write_parquet(CE / "ce_test_w_a.parquet")
        log(f"combined wide (A only): train {tr.height:,} pairs, test {te.height:,} pairs")
    elif cmd == "combine_wide":
        tr = pl.concat([pl.read_parquet(CE / "ce_train_all.parquet"), pl.read_parquet(CE / "cex_train_f0_b.parquet"),
                        pl.read_parquet(CE / "cex_train_f12_a.parquet")])
        tr.write_parquet(CE / "ce_train_all_w.parquet")
        tx = (pl.read_parquet(CE / "cex_test_a.parquet").join(pl.read_parquet(CE / "cex_test_b.parquet").rename({"ce": "ce_b"}), on=["s1", "rid"])
                .select("s1", "rid", pl.mean_horizontal("ce", "ce_b").alias("ce")))
        te = pl.concat([pl.read_parquet(CE / "ce_test_avg.parquet"), tx])
        te.write_parquet(CE / "ce_test_avg_w.parquet")
        log(f"combined wide: train {tr.height:,} pairs, test {te.height:,} pairs")
    else:
        score(cmd.split("_")[1])
