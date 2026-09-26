"""Blocking experiment: per-retriever and union recall on a train sample, plus a diagnosis of misses.

Usage: python blocking_lab.py [n_s1_per_country]
Reverse retrievers query every record of the country (exact candidate counts).
"""
import config  # noqa: F401  (first: points temp/cache dirs at the project drive)
import sys
import time

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from blocking import candidates
from config import SEED
from io_utils import truth_pairs
from normalize import prepared


def diagnose(miss: pl.DataFrame, df: pl.DataFrame) -> pl.DataFrame:
    r = df.select("gid", "name_roman", "addr_roman")
    m = (miss.join(r.rename({"gid": "s1", "name_roman": "n1", "addr_roman": "a1"}), on="s1")
             .join(r.rename({"gid": "rid", "name_roman": "n2", "addr_roman": "a2"}), on="rid"))
    ns = cpdist(m["n1"].to_list(), m["n2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)
    as_ = cpdist(m["a1"].to_list(), m["a2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)
    m = m.with_columns(pl.Series("ns", ns), pl.Series("as", as_))
    blank = pl.col("a2") == ""
    cat = (pl.when(blank & (pl.col("ns") >= 80)).then(pl.lit("blank addr, name similar"))
             .when(blank).then(pl.lit("blank addr, name changed"))
             .when((pl.col("ns") >= 80) & (pl.col("as") >= 60)).then(pl.lit("both similar (crowded out)"))
             .when((pl.col("ns") < 50) & (pl.col("as") >= 60)).then(pl.lit("renamed, addr similar"))
             .when(pl.col("ns") >= 80).then(pl.lit("name similar, addr damaged"))
             .otherwise(pl.lit("both damaged")))
    return m.with_columns(cat.alias("cat"))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    df = prepared("train", ["gid", "entity_id", "src", "country", "name_key", "name_roman", "addr_roman", "addr_nums"])
    ids = df.select("gid", "entity_id")
    truth = (truth_pairs().join(ids.rename({"entity_id": "s1", "gid": "g1"}), on="s1")
             .join(ids.rename({"entity_id": "rid", "gid": "g2"}), on="rid").select(pl.col("g1").alias("s1"), pl.col("g2").alias("rid")))
    s1_all = df.filter(pl.col("src") == 1)
    sample = pl.concat([s1_all.filter(pl.col("country") == c).sample(min(n, s1_all.filter(pl.col("country") == c).height), seed=SEED)
                        for c in s1_all["country"].unique().sort()])["gid"]
    # Reverse queries only for the true copies of the sample (recall is exact; per-query results do not
    # depend on which other records are queried) plus a random 2% of records to estimate candidate counts.
    tr_s = truth.filter(pl.col("s1").is_in(sample.implode()))
    rnd = df.filter(pl.col("src") != 1)["gid"].sample(fraction=0.02, seed=SEED)
    rev_q = pl.concat([tr_s["rid"], rnd]).unique()
    t = time.time()
    c = candidates(df, s1_subset=sample, rev_subset=rev_q, split="train")
    print(f"blocking {time.time() - t:.0f}s\n")

    tr = truth.filter(pl.col("s1").is_in(sample.implode()))
    u = c.select("s1", "rid").unique()
    hit = tr.join(u, on=["s1", "rid"])
    print(f"UNION recall {hit.height / tr.height:.4f}   cands/S1 {u.height / sample.len():.1f}")
    for name in c["retriever"].unique().sort().to_list():
        ui = c.filter(pl.col("retriever") == name).select("s1", "rid").unique()
        rec = tr.join(ui, on=["s1", "rid"]).height / tr.height
        others = c.filter(pl.col("retriever") != name).select("s1", "rid").unique()
        only = tr.join(ui, on=["s1", "rid"]).join(others, on=["s1", "rid"], how="anti").height / tr.height
        print(f"  {name:10s} recall {rec:.4f}  unique-contribution {only:.4f}  pairs/S1 {ui.height / sample.len():.1f}")
    meta = df.select(pl.col("gid").alias("s1"), "country")
    by_c = tr.join(meta, on="s1").with_columns(pl.struct("s1", "rid").is_in(hit.select(pl.struct("s1", "rid")).to_series().implode()).alias("hit"))
    print("\nrecall by country:", {r["country"]: round(r["hit"], 4) for r in by_c.group_by("country").agg(pl.col("hit").mean()).iter_rows(named=True)})

    miss = tr.join(u, on=["s1", "rid"], how="anti")
    d = diagnose(miss, df)
    print(f"\nMISSES: {miss.height} ({miss.height / tr.height:.2%} of true pairs)")
    print(d.group_by("cat").len().sort("len", descending=True).with_columns((pl.col("len") / tr.height).round(4).alias("share_of_true")))
    for cat in d["cat"].unique().to_list():
        print(f"\n-- {cat}")
        for row in d.filter(pl.col("cat") == cat).head(4).select("n1", "a1", "n2", "a2").iter_rows():
            print("  S1 :", row[0], "|", row[1][:70])
            print("  rec:", row[2], "|", row[3][:70])
