"""Step-1 baseline: exact name_key match within country, only when the key is unique among S1.

Usage: python baseline_exact.py [train|test]
"""
import config  # noqa: F401  (first: points temp/cache dirs at the project drive)
import sys
import time

import polars as pl

from config import OUT
from io_utils import truth_pairs, write_lists
from metric import report
from normalize import normalized

TEST_MIX = {"US": 0.383, "India": 0.467, "France": 0.150}


def predict(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.Series]:
    s1 = df.filter(pl.col("src") == 1)
    rest = df.filter(pl.col("src") != 1)
    uniq = s1.group_by(["country", "name_key"]).agg(pl.col("entity_id").first().alias("s1"), pl.len().alias("k")).filter(pl.col("k") == 1)
    pairs = rest.join(uniq, on=["country", "name_key"]).select("s1", pl.col("entity_id").alias("rid"))
    return pairs, s1["entity_id"]


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    t = time.time()
    df = normalized(split)
    print(f"normalized {split} in {time.time() - t:.0f}s")
    pairs, s1_ids = predict(df)
    if split == "train":
        meta = df.filter(pl.col("src") == 1).select(pl.col("entity_id").alias("s1"), "country")
        print(report(pairs, truth_pairs(), meta, TEST_MIX))
    else:
        write_lists(pairs, s1_ids, OUT / "matching_results.tsv", "matched_entity_ids")
        write_lists(pairs, s1_ids, OUT / "candidate_pairs.tsv", "candidate_entity_ids")
        print("wrote", OUT, pairs.height, "pairs")
