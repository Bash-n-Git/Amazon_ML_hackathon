"""Load raw TSVs once and cache them as parquet.

Records table columns: entity_id, name, addr, country, src (1/2/3).
Ground-truth pairs table columns: s1, rid (one row per true S1 -> S2/S3 link).
"""
import polars as pl

from config import pq, src_tsv, DATA


def _read_tsv(path) -> pl.DataFrame:
    # quote_char=None: names contain stray quotes; every field is tab-delimited.
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)


def records(split: str) -> pl.DataFrame:
    """All S1/S2/S3 records of a split."""
    parts = []
    for i in (1, 2, 3):
        d = _read_tsv(src_tsv(split, i))
        parts.append(d.select(
            pl.col("entity_id"),
            pl.col("business_name").fill_null("").alias("name"),
            pl.col("business_address").fill_null("").alias("addr"),
            pl.col("country").fill_null(""),
            pl.lit(i, dtype=pl.UInt8).alias("src"),
        ))
    return pl.concat(parts)


def truth_pairs() -> pl.DataFrame:
    """Train ground truth as (s1, rid) pairs (cached)."""
    p = pq("train_truth")
    if p.exists():
        return pl.read_parquet(p)
    gt = _read_tsv(DATA / "train" / "train_ground_truth.tsv")
    df = (gt.select(pl.col("source1_entity_id").alias("s1"),
                    pl.col("matched_entity_ids").fill_null("").str.split(",").alias("rid"))
            .explode("rid")
            .filter(pl.col("rid").is_not_null() & (pl.col("rid") != "")))
    df.write_parquet(p)
    return df


def write_lists(pairs: pl.DataFrame, s1_ids: pl.Series, path, col: str) -> None:
    """Write one row per S1 with a comma-joined id list (empty string when none)."""
    agg = pairs.unique(["s1", "rid"]).group_by("s1").agg(pl.col("rid").sort().str.join(",").alias(col))
    out = (pl.DataFrame({"source1_entity_id": s1_ids})
             .join(agg.rename({"s1": "source1_entity_id"}), on="source1_entity_id", how="left")
             .with_columns(pl.col(col).fill_null("")))
    assert out.height == s1_ids.len() and out["source1_entity_id"].n_unique() == out.height
    out.write_csv(path, separator="\t", quote_style="never")
