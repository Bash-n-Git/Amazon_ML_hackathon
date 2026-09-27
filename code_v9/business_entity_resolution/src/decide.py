"""Decision layer: turn pair probabilities into one match list per S1.

1. column_normalize: a record belongs to at most one S1, so its probabilities over S1 claimants
   are scaled to sum to <= 1, and claimants far below the record's best are dropped.
2. select_expected_f: per S1, sort candidates by probability and keep the top-k that maximises
   expected F0.5 = 1.25 * sum(p_1..p_k) / (k + 0.25 * N_hat), N_hat = sum(p) + m_hat.
   k = 0 (empty list) has expected value prod(1 - p) * (1 - q_hat).
"""
import polars as pl


def column_normalize(pairs: pl.DataFrame, p: str = "p", margin: float | None = None) -> pl.DataFrame:
    s = pl.col(p).sum().over("rid")
    out = pairs.with_columns((pl.col(p) / pl.max_horizontal(pl.lit(1.0), s)).alias("pn"))
    if margin is not None:
        out = out.filter(pl.col(p) >= pl.col(p).max().over("rid") - margin)
    return out


def select_expected_f(pairs: pl.DataFrame, p: str = "pn", m_hat: float = 0.0, q_hat: float = 0.0,
                      max_k: int = 12) -> pl.DataFrame:
    """Returns the selected (s1, rid) pairs."""
    df = (pairs.select("s1", "rid", p).sort(["s1", p], descending=[False, True])
               .with_columns(pl.col(p).cum_count().over("s1").alias("k"),
                             pl.col(p).cum_sum().over("s1").alias("cum"),
                             pl.col(p).sum().over("s1").alias("tot"),
                             (1 - pl.col(p)).log().sum().over("s1").exp().alias("e0")))
    df = df.filter(pl.col("k") <= max_k).with_columns(
        (1.25 * pl.col("cum") / (pl.col("k") + 0.25 * (pl.col("tot") + m_hat))).alias("ek"))
    best = df.group_by("s1").agg(pl.col("ek").max().alias("best_ek"), pl.col("e0").first())
    df = df.join(best, on="s1")
    kstar = (df.filter(pl.col("ek") == pl.col("best_ek")).group_by("s1").agg(pl.col("k").min().alias("kstar"),
                                                                           pl.col("best_ek").first(), pl.col("e0").first()))
    kstar = kstar.filter(pl.col("best_ek") > pl.col("e0") * (1 - q_hat))
    return df.join(kstar.select("s1", "kstar"), on="s1").filter(pl.col("k") <= pl.col("kstar")).select("s1", "rid")


def select_threshold(pairs: pl.DataFrame, t: float, p: str = "p") -> pl.DataFrame:
    return pairs.filter(pl.col(p) >= t).select("s1", "rid")


def select_greedy_exclusive(pairs: pl.DataFrame, t: float, p: str = "p") -> pl.DataFrame:
    """Plan_1 baseline: each record goes only to its highest-scoring S1, then threshold."""
    best = pairs.filter(pl.col(p) == pl.col(p).max().over("rid"))
    return best.filter(pl.col(p) >= t).select("s1", "rid")
