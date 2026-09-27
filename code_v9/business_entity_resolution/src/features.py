"""Pair features for (s1, rid) candidates. All numeric, float32.

Groups:
  name_*   string similarity on romanised names / keys
  addr_*   string + number similarity on romanised addresses
  rec_*    properties of the candidate record (noise flags, source)
  blk_*    which retriever found the pair and at what rank/score
  cmp_*    competition: how this pair ranks among the S1's candidates and among the record's S1 claimants
"""
import config  # noqa: F401  (first: points temp/cache dirs at the project drive)
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist

W = -1  # rapidfuzz workers: all cores

REC_COLS = ["gid", "name", "addr", "src", "name_roman", "name_key", "addr_roman", "addr_nums", "n_indic_unknown"]


def _sim(a, b, scorer, **kw):
    return cpdist(a, b, scorer=scorer, workers=W, dtype=np.float32, **kw)


def _tok_overlap(a: str, b: str) -> pl.Expr:
    """|A∩B|/|A|, |A∩B|/|B|, jaccard over whitespace tokens of two string columns."""
    A, B = pl.col(a).str.split(" "), pl.col(b).str.split(" ")
    inter = A.list.set_intersection(B).list.len()
    la, lb = A.list.unique().list.len(), B.list.unique().list.len()
    uni = A.list.set_union(B).list.len()
    return inter, la, lb, uni


def pair_features(pairs: pl.DataFrame, recs: pl.DataFrame) -> pl.DataFrame:
    """pairs: s1, rid gids (+ blocking columns from blocking.union_wide). recs: prepared() frame."""
    r = recs.select(REC_COLS)
    df = (pairs.join(r.rename({c: f"a_{c}" for c in REC_COLS}).rename({"a_gid": "s1"}), on="s1")
               .join(r.rename({c: f"b_{c}" for c in REC_COLS}).rename({"b_gid": "rid"}), on="rid"))

    an, bn = df["a_name_roman"].to_list(), df["b_name_roman"].to_list()
    ak, bk = df["a_name_key"].to_list(), df["b_name_key"].to_list()
    aa, ba = df["a_addr_roman"].to_list(), df["b_addr_roman"].to_list()
    f = {
        "name_ratio": _sim(an, bn, fuzz.ratio),
        "name_partial": _sim(an, bn, fuzz.partial_ratio),
        "name_tset": _sim(an, bn, fuzz.token_set_ratio),
        "name_tsort": _sim(an, bn, fuzz.token_sort_ratio),
        "name_jw": _sim(an, bn, JaroWinkler.normalized_similarity),
        "key_ratio": _sim(ak, bk, fuzz.ratio),
        "key_partial": _sim(ak, bk, fuzz.partial_ratio),
        "key_lev": _sim(ak, bk, Levenshtein.distance),
        "addr_ratio": _sim(aa, ba, fuzz.ratio),
        "addr_tset": _sim(aa, ba, fuzz.token_set_ratio),
        "addr_partial": _sim(aa, ba, fuzz.partial_ratio),
    }
    df = df.with_columns(**{k: pl.Series(v) for k, v in f.items()})

    ni, nla, nlb, nu = _tok_overlap("a_name_roman", "b_name_roman")
    ai, ala, alb, au = _tok_overlap("a_addr_roman", "b_addr_roman")
    Na, Nb = pl.col("a_addr_nums").str.split(" "), pl.col("b_addr_nums").str.split(" ")
    b_empty = pl.col("b_addr_roman") == ""
    df = df.with_columns(
        (pl.col("a_name_key") == pl.col("b_name_key")).alias("name_key_eq"),
        (ni / nla).alias("name_cont_a"), (ni / nlb).alias("name_cont_b"), (ni / nu).alias("name_jac"),
        nla.alias("name_ntok_a"), nlb.alias("name_ntok_b"),
        (pl.col("a_name_key").str.len_chars() - pl.col("b_name_key").str.len_chars()).alias("key_len_diff"),
        pl.when(b_empty).then(None).otherwise(ai / ala).alias("addr_cont_a"),
        pl.when(b_empty).then(None).otherwise(ai / alb).alias("addr_cont_b"),
        pl.when(b_empty).then(None).otherwise(ai / au).alias("addr_jac"),
        # numbers: 3-state via nulls when either side has none
        pl.when((pl.col("a_addr_nums") == "") | (pl.col("b_addr_nums") == "")).then(None)
          .otherwise(Na.list.set_intersection(Nb).list.len() / Na.list.set_union(Nb).list.len()).alias("addr_num_jac"),
        pl.when((pl.col("a_addr_nums") == "") | (pl.col("b_addr_nums") == "")).then(None)
          .otherwise(Na.list.first() == Nb.list.first()).alias("addr_num_first_eq"),
        (pl.col("b_addr_nums") != "").alias("rec_has_nums"),
        b_empty.alias("rec_addr_empty"),
        pl.col("b_src").alias("rec_src"),
        pl.col("b_n_indic_unknown").alias("rec_indic_unknown"),
        pl.col("b_name").str.contains(r"[ऀ-෿]").alias("rec_indic"),
        pl.col("b_name").str.contains(r"(?i)\.(com|in|net|org|co|fr)\b").alias("rec_domain"),
        pl.col("b_name").str.contains(r"^\s*(<<|>>|--)").alias("rec_junk"),
        pl.col("b_name").str.contains(r"[()\[\]]").alias("rec_paren"),
        pl.col("b_name").str.contains("  ").alias("rec_dblspace"),
        pl.col("b_name").str.to_lowercase().eq(pl.col("b_name")).alias("rec_lower"),
        pl.col("a_name").str.contains(r"^\s*(<<|>>|--)").alias("s1_junk"),
    )
    # blank addresses must not zero the address similarity signal
    df = df.with_columns([pl.when(b_empty).then(None).otherwise(pl.col(c)).alias(c) for c in ("addr_ratio", "addr_tset", "addr_partial")])
    df = add_competition(df)
    return df.drop([c for c in df.columns if c.startswith(("a_", "b_"))])


def add_competition(df: pl.DataFrame) -> pl.DataFrame:
    """Rank/gap of a cheap combined score within the S1's candidates.

    Record-side competition (other S1s claiming the same record) comes from the reverse retrievers'
    blk_comb_rev_* / blk_name_rev_* columns, which rank the record against ALL S1s. Counting claimants
    inside the candidate set would be biased on train, where only a sample of S1s is queried."""
    s = (pl.col("name_tset") + pl.col("key_ratio") + pl.col("addr_tset").fill_null(pl.col("name_tset"))) / 300
    df = df.with_columns(s.alias("cmp_score"))
    return df.with_columns(
        pl.len().over("s1").alias("cmp_n_cand_s1"),
        pl.col("cmp_score").rank("ordinal", descending=True).over("s1").alias("cmp_rank_in_s1"),
        (pl.col("cmp_score") - pl.col("cmp_score").max().over("s1")).alias("cmp_gap_to_best_s1"),
        (pl.col("cmp_score") - pl.col("cmp_score").mean().over("s1")).alias("cmp_gap_to_mean_s1"),
    )
