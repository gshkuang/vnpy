from datetime import datetime
import polars as pl

from .utility import to_datetime


def process_drop_na(df: pl.DataFrame, names: list[str] | None = None) -> pl.DataFrame:
    """Remove rows with missing values"""
    if names is None:
        names = df.columns[2:-1]

    for name in names:
        df = df.with_columns(
            pl.col(name).fill_nan(None)
        )
    df = df.drop_nulls(subset=names)
    return df


def process_fill_na(df: pl.DataFrame, fill_value: float, fill_label: bool = True) -> pl.DataFrame:
    """Fill missing values"""
    if fill_label:
        df = df.fill_null(fill_value)
        df = df.fill_nan(fill_value)
    else:
        df = df.with_columns(
            [pl.col(col).fill_null(fill_value).fill_nan(fill_value) for col in df.columns[2:-1]]
        )
    return df


def process_cs_norm(
    df: pl.DataFrame,
    names: list[str],
    method: str         # robust/zscore
) -> pl.DataFrame:
    """Cross-sectional normalization"""
    _df: pl.DataFrame = df.fill_nan(None)

    # Median method
    if method == "robust":
        for col in names:
            df = df.with_columns(
                _df.select(
                    (pl.col(col) - pl.col(col).median()).over("datetime").alias(col),
                )
            )

            df = df.with_columns(
                df.select(
                    pl.col(col).abs().median().over("datetime").alias("mad"),
                )
            )

            df = df.with_columns(
                (pl.col(col) / pl.col("mad") / 1.4826).clip(-3, 3).alias(col)
            ).drop(["mad"])
    # Z-Score method
    else:
        for col in names:
            df = df.with_columns(
                _df.select(
                    pl.col(col).mean().over("datetime").alias("mean"),
                    pl.col(col).std().over("datetime").alias("std"),
                )
            )

            df = df.with_columns(
                (pl.col(col) - pl.col("mean")) / pl.col("std").alias(col)
            ).drop(["mean", "std"])

    return df


def process_robust_zscore_norm(
    df: pl.DataFrame,
    fit_start_time: datetime | str | None = None,
    fit_end_time: datetime | str | None = None,
    clip_outlier: bool = True
) -> pl.DataFrame:
    """Robust Z-Score normalization - Optimized Polars version"""
    _df: pl.DataFrame = df.fill_nan(None)

    if fit_start_time and fit_end_time:
        fit_start_time = to_datetime(fit_start_time)
        fit_end_time = to_datetime(fit_end_time)
        _df = _df.filter((pl.col("datetime") >= fit_start_time) & (pl.col("datetime") <= fit_end_time))

    cols = df.columns[2:-1]
    
    # Calculate median and MAD for each column using native Polars operations
    # This avoids expensive numpy conversion
    stats_exprs = []
    for col in cols:
        median_expr = pl.col(col).median().alias(f"{col}_median")
        mad_expr = (pl.col(col) - pl.col(col).median()).abs().median().alias(f"{col}_mad")
        stats_exprs.extend([median_expr, mad_expr])
    
    # Calculate all statistics in one pass
    stats = _df.select(stats_exprs).row(0)
    
    # Create normalization expressions
    norm_exprs = []
    for i, col in enumerate(cols):
        median_val = stats[i * 2]
        mad_val = stats[i * 2 + 1]
        std_val = mad_val * 1.4826 + 1e-12
        
        normalized_col = ((pl.col(col) - median_val) / std_val).cast(pl.Float64)
        
        if clip_outlier:
            normalized_col = normalized_col.clip(-3, 3)
            
        norm_exprs.append(normalized_col.alias(col))
    
    # Apply all normalizations in one operation
    df = df.with_columns(norm_exprs)

    return df


def process_cs_rank_norm(df: pl.DataFrame, names: list[str]) -> pl.DataFrame:
    """Cross-sectional rank normalization"""
    _df: pl.DataFrame = df.fill_nan(None)

    _df = _df.with_columns([
        ((pl.col(col).rank("average").over("datetime") / pl.col("datetime").count().over("datetime")) - 0.5) * 3.46
        for col in names
    ])

    df = df.with_columns([
        _df[col].alias(col) for col in names
    ])

    return df
