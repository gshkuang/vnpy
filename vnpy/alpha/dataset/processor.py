from datetime import datetime

import polars as pl

from .utility import to_datetime


def process_lf_drop_na(
    lf: pl.LazyFrame, names: list[str] | None = None
) -> pl.LazyFrame:
    """Remove rows with missing values (lazy in/out)."""
    if names is None:
        names = lf.collect_schema().names()[2:]
    return lf.with_columns(pl.col(names).fill_nan(None)).drop_nulls(subset=names)


def process_lf_fill_na(
    lf: pl.LazyFrame, fill_value: float, fill_label: bool = False
) -> pl.LazyFrame:
    """Fill missing values (lazy in/out)."""
    target = pl.all() if fill_label else pl.col(lf.collect_schema().names()[2:-1])
    return lf.with_columns(target.fill_null(fill_value).fill_nan(fill_value))


def process_lf_cs_norm(
    lf: pl.LazyFrame, names: list[str], method: str  # robust/zscore
) -> pl.LazyFrame:
    """Cross-sectional normalization (lazy in/out)."""
    if method == "robust":
        # per-datetime median and MAD computed in one aggregation
        agg_exprs = []
        for c in names:
            median_expr = pl.col(c).median().alias(f"{c}_median")
            mad_expr = (pl.col(c) - pl.col(c).median()).abs().median().alias(f"{c}_mad")
            agg_exprs.extend([median_expr, mad_expr])

        stats = lf.groupby("datetime").agg(agg_exprs)
        joined = lf.join(stats, on="datetime", how="left")

        dev_cols = [
            (pl.col(c) - pl.col(f"{c}_median")).alias(f"{c}_dev") for c in names
        ]
        with_dev = joined.with_columns(dev_cols)

        norm_exprs = []
        for c in names:
            std_like = pl.col(f"{c}_mad") * 1.4826 + 1e-12
            norm = pl.col(f"{c}_dev") / std_like
            norm_exprs.append(norm.clip(-3, 3).alias(c))

        drop_cols = (
            [f"{c}_median" for c in names]
            + [f"{c}_mad" for c in names]
            + [f"{c}_dev" for c in names]
        )
        return with_dev.with_columns(norm_exprs).drop(drop_cols)
    else:
        stats = lf.groupby("datetime").agg(
            [pl.col(c).mean().alias(f"{c}_mean") for c in names]
            + [pl.col(c).std().alias(f"{c}_std") for c in names]
        )
        joined = lf.join(stats, on="datetime", how="left")

        dev_cols = [(pl.col(c) - pl.col(f"{c}_mean")).alias(f"{c}_dev") for c in names]
        with_dev = joined.with_columns(dev_cols)

        exprs = [
            (pl.col(f"{c}_dev") / (pl.col(f"{c}_std") + 1e-12)).alias(c) for c in names
        ]
        drop_cols = (
            [f"{c}_mean" for c in names]
            + [f"{c}_std" for c in names]
            + [f"{c}_dev" for c in names]
        )
        return with_dev.with_columns(exprs).drop(drop_cols)


def process_lf_robust_zscore_norm(
    lf: pl.LazyFrame,
    fit_start_time: datetime | str | None = None,
    fit_end_time: datetime | str | None = None,
    clip_outlier: bool = True,
) -> pl.LazyFrame:
    """Robust Z-Score normalization (lazy in/out, no early collect)."""
    cols = lf.collect_schema().names()[2:-1]
    base = lf.with_columns(pl.col(cols).fill_nan(None))

    if fit_start_time and fit_end_time:
        fit_start_time = to_datetime(fit_start_time)
        fit_end_time = to_datetime(fit_end_time)
        lf_fit = base.filter(
            (pl.col("datetime") >= fit_start_time)
            & (pl.col("datetime") <= fit_end_time)
        )
    else:
        lf_fit = base

    # global medians across fit window
    median_stats = lf_fit.select(
        [pl.col(c).median().alias(f"{c}_median") for c in cols]
    )
    # absolute deviations using broadcasted medians, then MAD across window
    dev_abs = lf_fit.join(median_stats, how="cross").select(
        [(pl.col(c) - pl.col(f"{c}_median")).abs().alias(f"{c}_abs_dev") for c in cols]
    )
    mad_stats = dev_abs.select(
        [pl.col(f"{c}_abs_dev").median().alias(f"{c}_mad") for c in cols]
    )

    # broadcast stats back to full frame and normalize
    joined = base.join(median_stats, how="cross").join(mad_stats, how="cross")
    norm_exprs = []
    for c in cols:
        std_like = pl.col(f"{c}_mad") * 1.4826 + 1e-12
        expr = ((pl.col(c) - pl.col(f"{c}_median")) / std_like).cast(pl.Float64)
        if clip_outlier:
            expr = expr.clip(-3, 3)
        norm_exprs.append(expr.alias(c))

    drop_cols = [f"{c}_median" for c in cols] + [f"{c}_mad" for c in cols]
    return joined.with_columns(norm_exprs).drop(drop_cols)


def process_lf_cs_rank_norm(lf: pl.LazyFrame, names: list[str]) -> pl.LazyFrame:
    exprs = [
        (
            (pl.col(c).rank("average").over("datetime") / pl.count().over("datetime"))
            - 0.5
        )
        * 3.46
        for c in names
    ]
    return lf.with_columns(exprs)


def process_stats_ts_norm(
    lf: pl.LazyFrame,
    stats_lf: pl.LazyFrame,
    names: list[str],
    method: str,  # robust/zscore
) -> pl.LazyFrame:
    joined = lf.join(stats_lf, how="cross")
    exprs: list[pl.Expr] = []
    if method == "zscore":
        for c in names:
            e = (pl.col(c) - pl.col(f"{c}_mean")) / (pl.col(f"{c}_std") + 1e-12)
            exprs.append(e.clip(-3.0, 3.0).alias(c))
        drop_cols = [f"{c}_mean" for c in names] + [f"{c}_std" for c in names]
    else:
        for c in names:
            std_like = pl.col(f"{c}_mad") * 1.4826 + 1e-12
            e = (pl.col(c) - pl.col(f"{c}_median")) / std_like
            exprs.append(e.clip(-3.0, 3.0).alias(c))
        drop_cols = [f"{c}_median" for c in names] + [f"{c}_mad" for c in names]
    return joined.with_columns(exprs).drop(drop_cols)


def process_stats_cs_norm(
    lf: pl.LazyFrame,
    cs_stats_lf: pl.LazyFrame,
    names: list[str],
    method: str,  # robust/zscore
) -> pl.LazyFrame:
    joined = lf.join(cs_stats_lf, on="datetime", how="left")
    exprs: list[pl.Expr] = []
    if method == "zscore":
        for c in names:
            e = (pl.col(c) - pl.col(f"{c}_mean")) / (pl.col(f"{c}_std") + 1e-12)
            exprs.append(e.clip(-3.0, 3.0).alias(c))
        drop_cols = [f"{c}_mean" for c in names] + [f"{c}_std" for c in names]
    else:
        for c in names:
            std_like = pl.col(f"{c}_mad") * 1.4826 + 1e-12
            e = (pl.col(c) - pl.col(f"{c}_median")) / std_like
            exprs.append(e.clip(-3.0, 3.0).alias(c))
        drop_cols = [f"{c}_median" for c in names] + [f"{c}_mad" for c in names]
    return joined.with_columns(exprs).drop(drop_cols)
