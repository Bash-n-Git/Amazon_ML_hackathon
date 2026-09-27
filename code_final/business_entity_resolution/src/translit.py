"""Indic -> Latin token table learned ONLY from train ground-truth pairs.

For each true pair (S1 Latin name/address, copy with Indic tokens) with the same token count,
tokens are aligned by position; Latin-token pairs are skipped, Indic tokens vote for their
Latin counterpart. The winning mapping is kept when it has >= MIN_COUNT votes and >= MIN_SHARE
of that token's votes. Output columns name_roman / addr_roman replace every mapped Indic token.
"""
import config  # noqa: F401  (first: points temp/cache dirs at the project drive)
import polars as pl

from config import pq

INDIC = r"[ऀ-෿]"  # Devanagari .. Sinhala blocks (covers all 9 scripts seen)
MIN_COUNT = 2
MIN_SHARE = 0.5


def _aligned(a: pl.Series, b: pl.Series) -> pl.DataFrame:
    """Two alignments, both positional:
    1. whole names with equal token counts;
    2. residuals: S1 tokens absent from the copy vs the copy's Indic tokens, when counts match
       (handles partial transliteration and dropped/inserted Latin tokens)."""
    df = pl.DataFrame({"lat": a.str.split(" "), "ind": b.str.split(" ")}).with_row_index("row")
    full = df.filter(pl.col("lat").list.len() == pl.col("ind").list.len()).select("lat", "ind")
    lat = df.select("row", "lat").explode("lat").with_row_index("pos")
    ind = df.select("row", "ind").explode("ind").with_row_index("pos")
    lat_res = lat.join(ind.rename({"ind": "lat"}).select("row", "lat"), on=["row", "lat"], how="anti")
    ind_res = ind.filter(pl.col("ind").str.contains(INDIC))
    lat_res = lat_res.sort("pos").group_by("row", maintain_order=True).agg("lat")
    ind_res = ind_res.sort("pos").group_by("row", maintain_order=True).agg("ind")
    res = (lat_res.join(ind_res, on="row")
                  .filter(pl.col("lat").list.len() == pl.col("ind").list.len()).select("lat", "ind"))
    out = pl.concat([full, res]).explode(["lat", "ind"])
    return out.filter(pl.col("ind").str.contains(INDIC) & ~pl.col("lat").str.contains(INDIC) & (pl.col("lat") != ""))


def learn_table() -> pl.DataFrame:
    p = pq("translit_table")
    if p.exists():
        return pl.read_parquet(p)
    from io_utils import truth_pairs
    from normalize import normalized
    df = normalized("train").select("entity_id", "name_norm", "addr_norm")
    t = (truth_pairs()
         .join(df.rename({"entity_id": "s1", "name_norm": "n1", "addr_norm": "a1"}), on="s1")
         .join(df.rename({"entity_id": "rid", "name_norm": "n2", "addr_norm": "a2"}), on="rid"))
    parts = []
    for c1, c2 in (("n1", "n2"), ("a1", "a2")):
        sub = t.filter(pl.col(c2).str.contains(INDIC))
        parts.append(_aligned(sub[c1], sub[c2]))
    votes = pl.concat(parts).group_by(["ind", "lat"]).len("n")
    tot = votes.group_by("ind").agg(pl.col("n").sum().alias("tot"))
    table = (votes.join(tot, on="ind").sort("n", descending=True).group_by("ind").first()
                  .filter((pl.col("n") >= MIN_COUNT) & (pl.col("n") / pl.col("tot") >= MIN_SHARE))
                  .select("ind", "lat", "n", "tot"))
    table.write_parquet(p)
    return table


def add_roman(df: pl.DataFrame, table: pl.DataFrame) -> pl.DataFrame:
    mapping = dict(zip(table["ind"].to_list(), table["lat"].to_list()))
    def rom(col):
        return (pl.when(pl.col(col).str.contains(INDIC))
                  .then(pl.col(col).str.split(" ").list.eval(pl.element().replace(mapping)).list.join(" "))
                  .otherwise(pl.col(col)))
    return df.with_columns(rom("name_norm").alias("name_roman"), rom("addr_norm").alias("addr_roman"))


if __name__ == "__main__":
    tab = learn_table()
    print(tab.height, "tokens learned")
    print(tab.sort("n", descending=True).head(15))
