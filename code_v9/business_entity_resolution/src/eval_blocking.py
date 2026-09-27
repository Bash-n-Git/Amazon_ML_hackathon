"""Measure blocking recall on a random sample of train S1 against the full train pool.

Usage: python eval_blocking.py [n_sample]
"""
import config  # noqa: F401  (first: points temp/cache dirs at the project drive)
import sys
import time

import polars as pl

from blocking import candidates, recall_report
from config import SEED
from io_utils import truth_pairs
from normalize import prepared

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50_000
    df = prepared("train")
    s1_ids = df.filter(pl.col("src") == 1)["entity_id"].sample(n, seed=SEED)
    t = time.time()
    c = candidates(df, s1_ids)
    print(f"blocking {time.time() - t:.0f}s")
    print(recall_report(c, truth_pairs(), s1_ids))
