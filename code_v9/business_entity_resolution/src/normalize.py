"""Text normalisation (v1). Adds columns; never overwrites raw `name` / `addr`.

name_norm  : casefolded, junk/punctuation stripped, whitespace collapsed
name_key   : name_norm with legal suffixes removed and all spaces dropped (domain-safe)
addr_norm  : casefolded, punctuation stripped, common street/state words canonicalised
addr_nums  : digit tokens of the address, space-joined
"""
import polars as pl

from config import pq

# Legal / filler tokens dropped for name_key. Country-agnostic list; France included.
LEGAL = [
    "inc", "incorporated", "llc", "l l c", "ltd", "limited", "pvt", "private", "corp", "corporation",
    "co", "company", "llp", "lp", "plc", "pllc", "pc", "opc", "the",
    "sarl", "sas", "sasu", "eurl", "sci", "sa", "snc", "scop", "selarl",
]
DOMAIN = r"\.(com|in|net|org|co|fr|biz|info)\b"

# Address word canonicalisation (applied to whole tokens after casefold).
ADDR_MAP = {
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue", "blvd": "boulevard", "bd": "boulevard",
    "dr": "drive", "ln": "lane", "ct": "court", "pl": "place", "hwy": "highway", "pkwy": "parkway",
    "ste": "suite", "apt": "apartment", "fl": "floor", "no": "number", "nr": "near", "opp": "opposite",
    "r": "rue", "che": "chemin", "all": "allee", "rte": "route", "imp": "impasse",
    "n": "north", "s": "south", "e": "east", "w": "west",
}


def _clean(col: str) -> pl.Expr:
    e = pl.col(col).str.normalize("NFKC").str.to_lowercase()
    # strip accents from Latin letters only (generator injects fake ones: cénter, Àmicale);
    # Indic vowel signs are also combining marks and must be kept
    e = e.str.normalize("NFD").str.replace_all(r"(\p{Latin})\p{Mn}+", "$1").str.normalize("NFC")
    e = e.str.replace_all(r"^\s*(<<|>>|--|\*+|#+)\s*", "")
    e = e.str.replace_all("&", " and ")
    e = e.str.replace_all(r"\bnull\b", " ")
    e = e.str.replace_all(r"[^\p{L}\p{N}\p{M}]+", " ")
    return e.str.strip_chars().str.replace_all(r"\s+", " ")


def _drop_tokens(e: pl.Expr, words) -> pl.Expr:
    pat = r"\b(" + "|".join(w.replace(" ", r"\s") for w in sorted(words, key=len, reverse=True)) + r")\b"
    return e.str.replace_all(pat, " ").str.replace_all(r"\s+", " ").str.strip_chars()


def _map_tokens(e: pl.Expr, mapping: dict) -> pl.Expr:
    # Token-wise replacement via split/eval keeps it vectorised.
    return e.str.split(" ").list.eval(pl.element().replace(mapping)).list.join(" ")


def add_norm(df: pl.DataFrame) -> pl.DataFrame:
    name_nodomain = pl.col("name").str.to_lowercase().str.replace_all(DOMAIN, " ")
    df = df.with_columns(name_nodomain.alias("_n"))
    df = df.with_columns(_clean("_n").alias("name_norm"), _clean("addr").alias("_a")).drop("_n")
    df = df.with_columns(
        _drop_tokens(pl.col("name_norm"), LEGAL).str.replace_all(" ", "").alias("name_key"),
        _map_tokens(pl.col("_a"), ADDR_MAP).alias("addr_norm"),
    ).drop("_a")
    df = df.with_columns(
        pl.when(pl.col("name_key") == "").then(pl.col("name_norm").str.replace_all(" ", "")).otherwise(pl.col("name_key")).alias("name_key"),
        pl.col("addr_norm").str.extract_all(r"\d+").list.join(" ").alias("addr_nums"),
    )
    return df


def normalized(split: str) -> pl.DataFrame:
    """Raw records + normalised columns (not cached; ~10 s to rebuild)."""
    from io_utils import records
    return add_norm(records(split))


PREP_COLS = ["entity_id", "name", "addr", "country", "src",
             "name_roman", "addr_roman", "name_key", "addr_nums", "n_indic_unknown"]


def prepared(split: str, columns: list[str] | None = None) -> pl.DataFrame:
    """normalized() + Indic->Latin romanisation (table learned from train pairs only). Cached.
      name_roman, addr_roman : romanised name_norm / addr_norm
      name_key               : legal suffixes dropped, spaces removed, from name_roman
      n_indic_unknown        : Indic name tokens missing from the table (distractor signal)
    gid: row number, a compact UInt32 id used by blocking/features.
    columns: load only these columns.
    """
    p = pq(f"{split}_prep")
    if not p.exists():
        from translit import INDIC, add_roman, learn_table
        df = add_roman(normalized(split), learn_table())
        df = df.with_columns(
            _drop_tokens(pl.col("name_roman"), LEGAL).str.replace_all(" ", "").alias("name_key"),
            pl.col("name_roman").str.split(" ").list.eval(pl.element().str.contains(INDIC).cast(pl.UInt8)).list.sum()
              .cast(pl.UInt8).alias("n_indic_unknown"),
        ).with_columns(
            pl.when(pl.col("name_key") == "").then(pl.col("name_roman").str.replace_all(" ", "")).otherwise(pl.col("name_key")).alias("name_key"),
        )
        df.select(PREP_COLS).with_row_index("gid").write_parquet(p)
    return pl.read_parquet(p, columns=columns, memory_map=False)
