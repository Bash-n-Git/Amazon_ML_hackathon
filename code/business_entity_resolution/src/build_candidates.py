"""Run blocking for a split and save candidates per country to work/cands/<split>_<country>.parquet.

Usage:
  python build_candidates.py train [s1_fraction]   # forward queries for a random S1 sample, reverse for all records
  python build_candidates.py test                  # everything
Output columns: s1, rid (gids), retriever, score, rank.
"""
import config  # noqa: F401  (first: points temp/cache dirs at the project drive)
import sys
import time

import polars as pl

from blocking import candidates
from config import SEED, WORK
from normalize import prepared

COLS = ["gid", "src", "country", "name_key", "addr_roman"]

if __name__ == "__main__":
    split = sys.argv[1]
    # `train rest 0.3`: the S1s NOT in the original 30% sample (same seed/order => exact complement),
    # saved as cands/trainrest_<country>.parquet
    rest = len(sys.argv) > 2 and sys.argv[2] == "rest"
    frac = float(sys.argv[-1]) if len(sys.argv) > 2 else 1.0
    out = WORK / "cands"
    out.mkdir(exist_ok=True)
    df = prepared(split, COLS)
    for country in df["country"].unique().sort().to_list():
        path = out / f"{split}{'rest' if rest else ''}_{country}.parquet"
        if path.exists():
            print(f"skip {path.name} (exists)")
            continue
        d = df.filter(pl.col("country") == country)
        s1 = d.filter(pl.col("src") == 1)["gid"]
        subset = None if frac >= 1.0 else s1.sample(fraction=frac, seed=SEED)
        if rest:
            subset = s1.filter(~s1.is_in(subset.implode()))
        t = time.time()
        c = candidates(d, s1_subset=subset, split=split)
        c.write_parquet(path)
        n_s1 = s1.len() if subset is None else subset.len()
        # raw rows/S1 (before de-duplicating across retrievers): a unique() over ~100M+ rows spikes RAM
        print(f"{split} {country}: {c.height:,} rows, {c.height / n_s1:.1f} raw rows/S1, "
              f"{time.time() - t:.0f}s", flush=True)
        del c, d
