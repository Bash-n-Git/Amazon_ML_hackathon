"""Exact challenge metric: F0.5 per Source-1 entity, macro-averaged (singletons included)."""
import polars as pl

BETA2 = 0.25


def per_entity_f05(pred: pl.DataFrame, truth: pl.DataFrame, s1_ids: pl.Series) -> pl.DataFrame:
    """pred/truth: (s1, rid) pairs. Returns s1, n_pred, n_true, tp, f05 for every id in s1_ids."""
    pred = pred.select("s1", "rid").unique()
    truth = truth.select("s1", "rid").unique()
    tp = pred.join(truth, on=["s1", "rid"]).group_by("s1").len("tp")
    npred = pred.group_by("s1").len("n_pred")
    ntrue = truth.group_by("s1").len("n_true")
    df = (pl.DataFrame({"s1": s1_ids})
            .join(npred, on="s1", how="left").join(ntrue, on="s1", how="left").join(tp, on="s1", how="left")
            .with_columns(pl.col("n_pred", "n_true", "tp").fill_null(0)))
    # F_beta = (1+b2) TP / (n_pred + b2 * n_true); both empty -> 1.0
    return df.with_columns(
        pl.when((pl.col("n_pred") == 0) & (pl.col("n_true") == 0)).then(1.0)
          .otherwise((1 + BETA2) * pl.col("tp") / (pl.col("n_pred") + BETA2 * pl.col("n_true")))
          .alias("f05"))


def macro_f05(pred, truth, s1_ids, weights: pl.DataFrame | None = None) -> float:
    """weights: optional (s1, w) frame, e.g. to reweight countries to the test mix."""
    df = per_entity_f05(pred, truth, s1_ids)
    if weights is None:
        return float(df["f05"].mean())
    df = df.join(weights, on="s1", how="left")
    return float((df["f05"] * df["w"]).sum() / df["w"].sum())


def report(pred, truth, s1_meta: pl.DataFrame, test_mix: dict | None = None) -> dict:
    """s1_meta: (s1, country). Prints overall, per-country, per group-size bucket, and test-mix reweighted."""
    df = per_entity_f05(pred, truth, s1_meta["s1"]).join(s1_meta, on="s1")
    df = df.with_columns(pl.when(pl.col("n_true") == 0).then(pl.lit("0"))
                           .when(pl.col("n_true") <= 3).then(pl.lit("1-3")).otherwise(pl.lit("4+")).alias("bucket"))
    tp, npred, ntrue = df["tp"].sum(), df["n_pred"].sum(), df["n_true"].sum()
    out = {"macro_f05": round(float(df["f05"].mean()), 5),
           "pair_precision": round(tp / max(npred, 1), 4), "pair_recall": round(tp / max(ntrue, 1), 4),
           "pred_per_s1": round(npred / df.height, 3)}
    out["by_country"] = {r["country"]: round(r["f05"], 5) for r in df.group_by("country").agg(pl.col("f05").mean()).iter_rows(named=True)}
    out["by_bucket"] = {r["bucket"]: round(r["f05"], 5) for r in df.group_by("bucket").agg(pl.col("f05").mean()).sort("bucket").iter_rows(named=True)}
    if test_mix:
        cm = out["by_country"]
        known = {c: w for c, w in test_mix.items() if c in cm}
        out["test_mix_f05"] = round(sum(cm[c] * w for c, w in known.items()) / sum(known.values()), 5)
    return out


if __name__ == "__main__":
    # Challenge example: pred {47,193,812} vs truth {47,812} -> 0.714
    p = pl.DataFrame({"s1": ["A"] * 3 + ["B"], "rid": ["S2-47", "S2-193", "S3-812", "S3-1"]})
    t = pl.DataFrame({"s1": ["A", "A"], "rid": ["S2-47", "S3-812"]})
    f = per_entity_f05(p, t, pl.Series(["A", "B", "C"]))
    print(f)
    assert abs(f.filter(pl.col("s1") == "A")["f05"][0] - 0.7142857) < 1e-6
    assert f.filter(pl.col("s1") == "B")["f05"][0] == 0.0   # singleton with a prediction
    assert f.filter(pl.col("s1") == "C")["f05"][0] == 1.0   # singleton, empty prediction
    print("metric OK")
