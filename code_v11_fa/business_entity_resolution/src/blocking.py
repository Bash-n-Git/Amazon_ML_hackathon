"""Candidate generation: union of independent retrievers, run within each country.

Each retriever returns (s1, rid, retriever, score, rank). Union = OR over retrievers.
"""
import config  # noqa: F401  (first: points temp/cache dirs at the project drive)
import time


import numpy as np
import polars as pl
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

N_THREADS = -1
CHUNK = 20_000


FIT_SAMPLE = 1_000_000
N_JOBS = 24


def _transform_parallel(vec, texts):
    parts = np.array_split(texts, max(1, min(N_JOBS * 4, len(texts) // 20_000 + 1)))
    mats = Parallel(n_jobs=N_JOBS)(delayed(vec.transform)(p) for p in parts)
    return sp.vstack(mats).tocsr()


def tfidf_matrices(q_text, i_text, analyzer="char", ngram=(3, 3), min_df=2, max_df=0.5, seed=0):
    """Fit vocabulary+idf on a sample of both sides, transform both in parallel. L2-normalised rows."""
    rng = np.random.default_rng(seed)
    allt = np.concatenate([q_text, i_text])
    fit = allt if len(allt) <= FIT_SAMPLE else allt[rng.choice(len(allt), FIT_SAMPLE, replace=False)]
    vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=min_df, max_df=max_df,
                          sublinear_tf=True, dtype=np.float32)
    vec.fit(fit)
    return _transform_parallel(vec, q_text), _transform_parallel(vec, i_text)


GPU_Q_CHUNK = 2048        # queries per GPU pass (dense V x B block)
GPU_I_CHUNK = 125_000     # index rows per sparse block (score block = I_CHUNK x Q_CHUNK fp32 = 1 GB)


def topk(Q, I, k):
    """Exact cosine top-k of rows of Q against rows of I. GPU when available, else CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return topk_gpu(Q, I, k)
    except ImportError:
        pass
    return topk_cpu(Q, I, k)


def _grouped_topk(S, k, G=64):
    """Exact top-k along dim 0, ~2x faster than torch.topk on tall blocks: take the max of each group
    of G rows, keep the k best groups (they must contain the k best rows), then top-k inside them."""
    import torch
    R, n = S.shape
    if R <= k * G:
        v, i = torch.topk(S, min(k, R), dim=0)
        return v, i
    Rp = (R + G - 1) // G * G
    if Rp != R:
        S = torch.nn.functional.pad(S, (0, 0, 0, Rp - R), value=-1.0)
    gm = S.view(Rp // G, G, n).amax(dim=1)
    _, gi = torch.topk(gm, k, dim=0)
    rows = (gi.unsqueeze(1) * G + torch.arange(G, device=S.device).view(1, G, 1)).reshape(k * G, n)
    v, j = torch.topk(torch.gather(S, 0, rows), k, dim=0)
    return v, torch.gather(rows, 0, j)


def topk_gpu(Q, I, k):
    """Sparse(index block) @ dense(query block) on the GPU, running top-k merge over index blocks.
    fp16 arithmetic: scores match the CPU product to ~1e-3, enough for ranking."""
    import torch
    dev = "cuda"
    V = Q.shape[1]
    blocks = []
    for b in range(0, I.shape[0], GPU_I_CHUNK):
        Ib = I[b:b + GPU_I_CHUNK].tocsr()
        blocks.append((b, torch.sparse_csr_tensor(torch.from_numpy(Ib.indptr.astype(np.int32)),
                                                  torch.from_numpy(Ib.indices.astype(np.int32)),
                                                  torch.from_numpy(Ib.data.astype(np.float16)),
                                                  size=Ib.shape, device=dev)))
    qs, is_, ss = [], [], []
    Qc = Q.tocsc() if False else Q
    for a in range(0, Q.shape[0], GPU_Q_CHUNK):
        qb = Qc[a:a + GPU_Q_CHUNK]
        n = qb.shape[0]
        Qd = torch.zeros((V, n), dtype=torch.float16, device=dev)
        coo = qb.tocoo()
        Qd[torch.from_numpy(coo.col.astype(np.int64)).to(dev), torch.from_numpy(coo.row.astype(np.int64)).to(dev)] = \
            torch.from_numpy(coo.data.astype(np.float16)).to(dev)
        best_v = best_i = None
        for off, Ib in blocks:
            S = torch.sparse.mm(Ib, Qd)                       # (rows_in_block, n)
            v, i = _grouped_topk(S, k)                       # (kk, n)
            i = i + off
            if best_v is None:
                best_v, best_i = v, i
            else:
                v2 = torch.cat([best_v, v]); i2 = torch.cat([best_i, i])
                best_v, sel = torch.topk(v2, min(k, v2.shape[0]), dim=0)
                best_i = torch.gather(i2, 0, sel)
            del S
        keep = best_v > 0
        cols = torch.arange(n, device=dev).expand_as(best_v)
        qs.append((cols[keep] + a).cpu().numpy()); is_.append(best_i[keep].cpu().numpy()); ss.append(best_v[keep].float().cpu().numpy())
    del blocks
    torch.cuda.empty_cache()
    return np.concatenate(qs), np.concatenate(is_), np.concatenate(ss)


def topk_cpu(Q, I, k):
    """Cosine top-k of rows of Q against rows of I. Returns (q_idx, i_idx, score)."""
    I = I.T.tocsr()
    qs, is_, ss = [], [], []
    for a in range(0, Q.shape[0], CHUNK):
        C = sp_matmul_topn(Q[a:a + CHUNK], I, top_n=k, sort=True, n_threads=N_THREADS).tocoo()
        qs.append(C.row + a); is_.append(C.col); ss.append(C.data)
    return np.concatenate(qs), np.concatenate(is_), np.concatenate(ss)


RETRIEVERS = pl.Enum(["addr_fwd", "comb_fwd", "blank_fwd", "comb_rev", "name_rev", "name_fwd", "exact"])
EMPTY = {"s1": pl.UInt32, "rid": pl.UInt32, "retriever": RETRIEVERS, "score": pl.Float32, "rank": pl.UInt16}


def _pairs(qi, ii, sc, q_ids, i_ids, name, reverse):
    a, b = q_ids[qi], i_ids[ii]
    s1, rid = (b, a) if reverse else (a, b)
    df = pl.DataFrame({"s1": s1, "rid": rid, "score": sc.astype(np.float32), "q": qi})
    df = df.with_columns(pl.col("score").rank("ordinal", descending=True).over("q").cast(pl.UInt16).alias("rank"),
                         pl.lit(name).cast(RETRIEVERS).alias("retriever")).drop("q")
    return df.select(list(EMPTY))


NAME_TFIDF = dict(analyzer="char", ngram=(3, 3))
ADDR_TFIDF = dict(analyzer="char_wb", ngram=(3, 3))

# Retriever -> k. Forward = S1 queries records; reverse = record queries S1.
K = {
    "addr_fwd": 50,    # address char-3gram (aliases, renamed businesses)
    "comb_fwd": 50,    # name + address in one vector (separates same-name branches)
    "blank_fwd": 30,   # name, searched only among blank-address records (small, uncrowded index)
    "comb_rev": 3,     # record -> S1, combined
    # name_rev (record -> S1, name only) dropped: 0.02% unique recall for ~40% of reverse time
}


def _hstack_scaled(a, b):
    """Concatenate two L2-normalised blocks with weight sqrt(.5) each: dot = (cos_a + cos_b) / 2."""
    c = sp.hstack([a, b], format="csr")  # fast CSR path, no COO copy
    c.data *= np.float32(np.sqrt(0.5))
    return c


def exact_pairs(s1: pl.DataFrame, rest: pl.DataFrame) -> pl.DataFrame:
    """Exact name_key buckets (skipping keys shared by > 20 S1), and exact name_key + first address number."""
    big = s1.group_by("name_key").len().filter(pl.col("len") > 20).select("name_key")
    a = s1.select(pl.col("entity_id").alias("s1"), "name_key").join(big, on="name_key", how="anti")
    p1 = a.join(rest.select(pl.col("entity_id").alias("rid"), "name_key"), on="name_key").select("s1", "rid")
    fn = pl.col("addr_nums").str.split(" ").list.first()
    a2 = s1.filter(pl.col("addr_nums") != "").select(pl.col("entity_id").alias("s1"), "name_key", fn.alias("n0"))
    b2 = rest.filter(pl.col("addr_nums") != "").select(pl.col("entity_id").alias("rid"), "name_key", fn.alias("n0"))
    p2 = a2.join(b2, on=["name_key", "n0"]).select("s1", "rid")
    return (pl.concat([p1, p2]).unique()
              .with_columns(pl.lit("exact").alias("retriever"), pl.lit(1.0, pl.Float32).alias("score"), pl.lit(1, pl.UInt16).alias("rank"))
              .select(list(EMPTY)))


def _vectors(s1: pl.DataFrame, rest: pl.DataFrame, tag: str | None):
    """Name and address TF-IDF matrices for S1 and records; cached under work/tfidf/<tag>_*.npz."""
    from config import WORK
    d = WORK / "tfidf"
    files = [d / f"{tag}_{m}.npz" for m in ("Ns", "Nr", "As", "Ar")] if tag else []
    if files and all(f.exists() for f in files):
        mats = [sp.load_npz(f).tocsr() for f in files]
        if mats[0].shape[0] == s1.height and mats[1].shape[0] == rest.height:
            return mats
    Ns, Nr = tfidf_matrices(s1["name_key"].to_numpy(), rest["name_key"].to_numpy(), **NAME_TFIDF)
    As, Ar = tfidf_matrices(s1["addr_roman"].to_numpy(), rest["addr_roman"].to_numpy(), **ADDR_TFIDF)
    if files:
        d.mkdir(exist_ok=True)
        for f, m in zip(files, (Ns, Nr, As, Ar)):
            sp.save_npz(f, m, compressed=False)
    return Ns, Nr, As, Ar


def candidates(df: pl.DataFrame, s1_subset: pl.Series | None = None, rev_subset: pl.Series | None = None,
               k: dict = K, split: str | None = None, verbose=True) -> pl.DataFrame:
    """Union of retrievers, per country. df: prepared() records of one split with a `gid` row-id column;
    output s1/rid are gids (UInt32).
    s1_subset: only emit pairs for these S1 (evaluation). rev_subset: only run reverse queries for
    these record ids (evaluation; results per query do not depend on which other records are queried).
    split: when given, TF-IDF matrices are cached per split and country."""
    out = []
    for country in df["country"].unique().sort().to_list():
        d = df.filter(pl.col("country") == country).select("gid", "src", "name_key", "addr_roman")
        s1 = d.filter(pl.col("src") == 1)
        rest = d.filter(pl.col("src") != 1)
        t = time.time()
        Ns, Nr, As, Ar = _vectors(s1, rest, f"{split}_{country}" if split else None)
        Cs, Cr = _hstack_scaled(Ns, As), _hstack_scaled(Nr, Ar)
        s1_ids, rest_ids = s1["gid"].to_numpy(), rest["gid"].to_numpy()
        if verbose:
            print(f"  {country:8s} vectorised {s1.height:,} S1 / {rest.height:,} records in {time.time() - t:.0f}s", flush=True)

        fq = np.arange(s1.height) if s1_subset is None else np.flatnonzero(s1["gid"].is_in(s1_subset.implode()).to_numpy())
        rq = np.arange(rest.height) if rev_subset is None else np.flatnonzero(rest["gid"].is_in(rev_subset.implode()).to_numpy())
        blank = np.flatnonzero((rest["addr_roman"] == "").to_numpy())
        jobs = {
            "name_fwd": lambda n: _pairs(*topk(Ns[fq], Nr, n), s1_ids[fq], rest_ids, "name_fwd", False),
            "addr_fwd": lambda n: _pairs(*topk(As[fq], Ar, n), s1_ids[fq], rest_ids, "addr_fwd", False),
            "comb_fwd": lambda n: _pairs(*topk(Cs[fq], Cr, n), s1_ids[fq], rest_ids, "comb_fwd", False),
            "blank_fwd": lambda n: _pairs(*topk(Ns[fq], Nr[blank], n), s1_ids[fq], rest_ids[blank], "blank_fwd", False),
            "comb_rev": lambda n: _pairs(*topk(Cr[rq], Cs, n), rest_ids[rq], s1_ids, "comb_rev", True),
            "name_rev": lambda n: _pairs(*topk(Nr[rq], Ns, n), rest_ids[rq], s1_ids, "name_rev", True),
        }
        for name, n in k.items():
            t = time.time()
            ck = None
            if split:
                from config import WORK
                (WORK / "cands" / "parts").mkdir(parents=True, exist_ok=True)
                ck = WORK / "cands" / "parts" / f"{split}_{country}_{name}_k{n}_{len(fq)}.parquet"
            if ck is not None and ck.exists():
                r = pl.read_parquet(ck)
            else:
                r = jobs[name](n)
                if s1_subset is not None:
                    r = r.filter(pl.col("s1").is_in(s1_subset.implode()))
                if ck is not None:
                    r.write_parquet(ck)
            out.append(r)
            if verbose:
                print(f"  {country:8s} {name:10s} {r.height:>11,d} pairs  {time.time() - t:6.0f}s", flush=True)
    return pl.concat(out) if out else pl.DataFrame(schema=EMPTY)


def union_wide(c: pl.DataFrame) -> pl.DataFrame:
    """One row per (s1, rid): blk_<retriever>_rank / _score (null when that retriever missed) + blk_n_hits."""
    c = c.group_by(["s1", "rid", "retriever"]).agg(pl.col("rank").min(), pl.col("score").max())
    rk = c.pivot(on="retriever", index=["s1", "rid"], values="rank")
    sc = c.pivot(on="retriever", index=["s1", "rid"], values="score")
    names = [x for x in rk.columns if x not in ("s1", "rid")]
    rk = rk.rename({n: f"blk_{n}_rank" for n in names})
    sc = sc.rename({n: f"blk_{n}_score" for n in names})
    out = rk.join(sc, on=["s1", "rid"])
    return out.with_columns(pl.sum_horizontal([pl.col(f"blk_{n}_rank").is_not_null() for n in names]).alias("blk_n_hits"))


def recall_report(c: pl.DataFrame, truth: pl.DataFrame, s1_ids: pl.Series) -> dict:
    t = truth.filter(pl.col("s1").is_in(s1_ids.implode()))
    u = c.select("s1", "rid").unique()
    hit = t.join(u, on=["s1", "rid"]).height
    out = {"recall": round(hit / t.height, 4), "cands_per_s1": round(u.height / s1_ids.len(), 1)}
    for name in c["retriever"].unique().sort().to_list():
        ui = c.filter(pl.col("retriever") == name).select("s1", "rid").unique()
        out[f"recall_{name}"] = round(t.join(ui, on=["s1", "rid"]).height / t.height, 4)
    return out
