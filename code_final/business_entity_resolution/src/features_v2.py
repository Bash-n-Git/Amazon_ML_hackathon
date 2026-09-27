"""Feature group v2 (Plan_2 §5): extra pair features for the pairs already in work/pipe/feat/.

Written as work/pipe/feat2/<same chunk name>, one row per (s1, rid) of the matching feat/ chunk, and
joined on (s1, rid) by pipeline.py when BER_RUN is set. All prefixed `v2_`.

  I1  fuzzy house numbers  : leading zeros stripped, digit edit distance, containment, unmatched counts
  I3  address noise undone : saint->street, fourth->4th, filler words dropped (po box, township, unit ...),
                             repeated tokens removed; typo-tolerant comparison of the address *words*
  I4  alias / invented name: share of the record's name tokens that occur in any S1 name of the split,
                             and the frequency of its rarest token
  I5  France               : extra street words (ch, fbg, sq, crs), bis/ter dropped
  I6  Indic phonetic key   : sh/s, ee/i, aa/a, w/v, z/j, doubled letters ... (sree = shree, jai = jay)
  I9  tie-safe ranks       : rank('min') instead of row-order tie breaking
Record-level columns are cached per split in work/<split>_prep2.parquet.
"""
import config  # noqa: F401  (first: points temp/cache dirs at the project drive)
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cpdist

from blocking import K
from config import WORK
from normalize import LEGAL, prepared

ORDINALS = {w: f"{i}{'st' if i % 10 == 1 and i != 11 else 'nd' if i % 10 == 2 and i != 12 else 'rd' if i % 10 == 3 and i != 13 else 'th'}"
            for i, w in enumerate(["", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth",
                                   "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth",
                                   "seventeenth", "eighteenth", "nineteenth", "twentieth"]) if w}
ADDR2_MAP = {"saint": "street", "ch": "chemin", "fbg": "faubourg", "sq": "square", "crs": "cours", **ORDINALS}
FILLER = {"po", "box", "township", "unit", "apartment", "suite", "floor", "number", "door", "h", "house", "plot",
          "bis", "ter", "and", "the"}
# Indic romanisation variants -> one spelling (order matters: digraphs before single letters)
PHON = [("sh", "s"), ("ph", "f"), ("th", "t"), ("kh", "k"), ("gh", "g"), ("bh", "b"), ("dh", "d"), ("ch", "c"),
        ("ck", "k"), ("ee", "i"), ("ii", "i"), ("oo", "u"), ("w", "v"), ("z", "j"), ("y", "i"), ("q", "k"), ("x", "ks")]


def _addr2(e: pl.Expr) -> pl.Expr:
    toks = e.str.split(" ").list.eval(pl.element().replace(ADDR2_MAP)).list.eval(pl.element().filter(~pl.element().is_in(list(FILLER)) & (pl.element() != "")))
    return toks.list.unique(maintain_order=True).list.join(" ")


def _phon(e: pl.Expr) -> pl.Expr:
    for a, b in PHON:
        e = e.str.replace_all(a, b, literal=True)
    for c in "abcdefghijklmnopqrstuvwxyz":
        e = e.str.replace_all(c + "+", c)
    return e


def prep2(split: str) -> pl.DataFrame:
    """Record-level v2 columns for every record of a split (cached)."""
    p = WORK / f"{split}_prep2.parquet"
    if p.exists():
        return pl.read_parquet(p)
    d = prepared(split, ["gid", "src", "country", "name_roman", "name_key", "addr_roman"])
    d = d.with_columns(_addr2(pl.col("addr_roman")).alias("addr2"))
    d = d.with_columns(
        pl.col("addr2").str.extract_all(r"\d+").list.eval(pl.element().str.strip_chars_start("0").replace("", "0"))
          .list.unique(maintain_order=True).list.join(" ").alias("nums_z"),
        pl.col("addr2").str.split(" ").list.eval(pl.element().filter(~pl.element().str.contains(r"\d")))
          .list.join(" ").alias("words"),
        _phon(pl.col("name_key")).alias("name_phon"),
    )
    # alias signal: how "known" are the record's name tokens among S1 names of the same split?
    legal = set(LEGAL)
    toks = (d.select("gid", "src", pl.col("name_roman").str.split(" ").alias("t")).explode("t")
              .filter(pl.col("t").is_not_null() & (pl.col("t").str.len_chars() >= 2) & ~pl.col("t").is_in(list(legal))))
    vocab = toks.filter(pl.col("src") == 1).group_by("t").agg(pl.col("gid").n_unique().alias("f"))
    tstat = (toks.join(vocab, on="t", how="left").with_columns(pl.col("f").fill_null(0))
                 .group_by("gid").agg((pl.col("f") > 0).mean().alias("vocab_share"),
                                      pl.col("f").min().log1p().alias("min_tok_logf"),
                                      pl.len().alias("n_tok")))
    d = d.join(tstat, on="gid", how="left")
    out = d.select("gid", "addr2", "nums_z", "words", "name_phon", "vocab_share", "min_tok_logf")
    out.write_parquet(p)
    return out


def _sim(a, b, scorer, **kw):
    return cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32, **kw)


def pair_features2(chunk: pl.DataFrame, p2: pl.DataFrame) -> pl.DataFrame:
    """chunk: one feat/ chunk file (needs s1, rid, cmp_score, blk_*_score). Returns s1, rid + v2_* columns."""
    base = chunk.select("s1", "rid", "cmp_score", *[f"blk_{r}_score" for r in K])
    df = (base.join(p2.rename({c: f"a_{c}" for c in p2.columns}).rename({"a_gid": "s1"}), on="s1", how="left")
              .join(p2.rename({c: f"b_{c}" for c in p2.columns}).rename({"b_gid": "rid"}), on="rid", how="left"))
    fill = lambda c: df[c].fill_null("").to_list()
    an, bn = fill("a_nums_z"), fill("b_nums_z")
    af = [s.split(" ", 1)[0] for s in an]
    bf = [s.split(" ", 1)[0] for s in bn]
    aw, bw = fill("a_words"), fill("b_words")
    a2, b2 = fill("a_addr2"), fill("b_addr2")
    ap, bp = fill("a_name_phon"), fill("b_name_phon")
    f = {
        "v2_num_first_lev": _sim(af, bf, Levenshtein.distance),
        "v2_num_partial": _sim(an, bn, fuzz.partial_ratio),
        "v2_addr2_tset": _sim(a2, b2, fuzz.token_set_ratio),
        "v2_addr2_tsort": _sim(a2, b2, fuzz.token_sort_ratio),
        "v2_words_tsort": _sim(aw, bw, fuzz.token_sort_ratio),
        "v2_words_tset": _sim(aw, bw, fuzz.token_set_ratio),
        "v2_words_partial": _sim(aw, bw, fuzz.partial_ratio),
        "v2_phon_ratio": _sim(ap, bp, fuzz.ratio),
        "v2_phon_partial": _sim(ap, bp, fuzz.partial_ratio),
    }
    df = df.with_columns(**{k: pl.Series(v) for k, v in f.items()})
    Na, Nb = pl.col("a_nums_z").str.split(" "), pl.col("b_nums_z").str.split(" ")
    no_nums = (pl.col("a_nums_z").fill_null("") == "") | (pl.col("b_nums_z").fill_null("") == "")
    no_addr = pl.col("b_addr2").fill_null("") == ""
    no_words = (pl.col("a_words").fill_null("") == "") | (pl.col("b_words").fill_null("") == "")
    inter = Na.list.set_intersection(Nb).list.len()
    df = df.with_columns(
        pl.when(no_nums).then(None).otherwise(inter / Na.list.set_union(Nb).list.len()).alias("v2_num_jac"),
        pl.when(no_nums).then(None).otherwise(Na.list.first() == Nb.list.first()).alias("v2_num_first_eq"),
        pl.when(no_nums).then(None).otherwise(Nb.list.contains(Na.list.first())).alias("v2_num_a_first_in_b"),
        pl.when(no_nums).then(None).otherwise(Na.list.contains(Nb.list.first())).alias("v2_num_b_first_in_a"),
        pl.when(no_nums).then(None).otherwise(Na.list.len() - inter).alias("v2_num_unmatched_a"),
        pl.when(no_nums).then(None).otherwise(Nb.list.len() - inter).alias("v2_num_unmatched_b"),
        pl.when(no_nums).then(None).otherwise(pl.col("v2_num_first_lev")).alias("v2_num_first_lev"),
        pl.when(no_nums).then(None).otherwise(pl.col("v2_num_partial")).alias("v2_num_partial"),
        *[pl.when(no_addr).then(None).otherwise(pl.col(c)).alias(c) for c in ("v2_addr2_tset", "v2_addr2_tsort")],
        *[pl.when(no_words).then(None).otherwise(pl.col(c)).alias(c)
          for c in ("v2_words_tsort", "v2_words_tset", "v2_words_partial")],
        (pl.col("a_name_phon") == pl.col("b_name_phon")).alias("v2_phon_eq"),
        pl.col("b_vocab_share").alias("v2_rec_vocab_share"),
        pl.col("b_min_tok_logf").alias("v2_rec_min_tok_logf"),
        # I9: tie-safe ranks within the S1's candidate list
        pl.col("cmp_score").rank("min", descending=True).over("s1").alias("v2_cmp_rank_min"),
        pl.max_horizontal([pl.col(f"blk_{r}_score") for r in K]).rank("min", descending=True).over("s1").alias("v2_s1_rank_best_min"),
    )
    return df.select("s1", "rid", *[c for c in df.columns if c.startswith("v2_")])
