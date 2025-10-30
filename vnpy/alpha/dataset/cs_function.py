"""
Cross Section Operators (LazyFrame-based)
"""

import polars as pl

from .utility import FeatProxy


def cs_rank(feature: FeatProxy) -> FeatProxy:
    """Perform cross-sectional ranking"""
    lf: pl.LazyFrame = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").rank().over("datetime")
    )
    return FeatProxy(lf)


def cs_mean(feature: FeatProxy) -> FeatProxy:
    """Calculate cross-sectional mean"""
    lf: pl.LazyFrame = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").mean().over("datetime")
    )
    return FeatProxy(lf)


def cs_std(feature: FeatProxy) -> FeatProxy:
    """Calculate cross-sectional standard deviation"""
    lf: pl.LazyFrame = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").std().over("datetime")
    )
    return FeatProxy(lf)
