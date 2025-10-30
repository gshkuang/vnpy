"""
Time Series Operators (LazyFrame-based)
"""

from typing import cast

from scipy import stats  # type: ignore
import polars as pl
import numpy as np

from .utility import FeatProxy


def ts_delay(feature: FeatProxy, window: int) -> FeatProxy:
    """Get the value from a fixed time in the past"""
    lf = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").shift(window).over("vt_symbol"),
    )
    return FeatProxy(lf)


def ts_min(feature: FeatProxy, window: int) -> FeatProxy:
    """Calculate the minimum value over a rolling window"""
    lf = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").rolling_min(window, min_samples=1).over("vt_symbol"),
    )
    return FeatProxy(lf)


def ts_max(feature: FeatProxy, window: int) -> FeatProxy:
    """Calculate the maximum value over a rolling window"""
    lf = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").rolling_max(window, min_samples=1).over("vt_symbol"),
    )
    return FeatProxy(lf)


def ts_argmax(feature: FeatProxy, window: int) -> FeatProxy:
    """Return the index of the maximum value over a rolling window"""
    lf = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data")
        .rolling_map(lambda s: cast(int, s.arg_max()) + 1, window)
        .over("vt_symbol"),
    )
    return FeatProxy(lf)


def ts_argmin(feature: FeatProxy, window: int) -> FeatProxy:
    """Return the index of the minimum value over a rolling window"""
    lf = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data")
        .rolling_map(lambda s: cast(int, s.arg_min()) + 1, window)
        .over("vt_symbol"),
    )
    return FeatProxy(lf)


def ts_rank(feature: FeatProxy, window: int) -> FeatProxy:
    """Calculate the percentile rank of the current value within the window"""
    lf = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        # 纯 Series 计算百分位：<= 当前值 的数量 / 窗口长度
        pl.col("data").rolling_map(lambda s: ((s <= s[-1]).sum() / len(s)), window).over("vt_symbol")
    )
    return FeatProxy(lf)


def ts_sum(feature: FeatProxy, window: int) -> FeatProxy:
    """Calculate the sum over a rolling window"""
    # Cast to Float64 to support boolean inputs
    lf = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").cast(pl.Float64).rolling_sum(window).over("vt_symbol"),
    )
    return FeatProxy(lf)


def ts_mean(feature: FeatProxy, window: int) -> FeatProxy:
    """Calculate the mean over a rolling window"""
    # Cast to Float64 to support boolean inputs
    lf = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").cast(pl.Float64).rolling_mean(window, min_samples=1).over("vt_symbol"),
    )
    return FeatProxy(lf)


def ts_std(feature: FeatProxy, window: int) -> FeatProxy:
    """Calculate the standard deviation over a rolling window"""
    lf = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").rolling_std(window, min_samples=1).over("vt_symbol"),
    )
    return FeatProxy(lf)


def ts_slope(feature: FeatProxy, window: int) -> FeatProxy:
    """Calculate the slope of linear regression over a rolling window"""

    def slope_py(s: pl.Series) -> float:
        ys = s.to_list()
        n = len(ys)
        if n == 0:
            return float("nan")
        mean_x = (n - 1) / 2.0
        mean_y = sum(ys) / n
        num = sum((i - mean_x) * (ys[i] - mean_y) for i in range(n))
        den = sum((i - mean_x) ** 2 for i in range(n))
        if den == 0:
            return float("nan")
        return num / den

    lf = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data")
        .rolling_map(lambda s: slope_py(s), window)
        .over("vt_symbol"),  # np.polyfit(np.arange(len(s)), s, 1)[0],
    )
    return FeatProxy(lf)


def ts_quantile(feature: FeatProxy, window: int, quantile: float) -> FeatProxy:
    """Calculate the quantile value over a rolling window"""
    lf = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data")
        .rolling_map(
            lambda s: s.quantile(quantile=quantile, interpolation="linear"), window
        )
        .over("vt_symbol"),
    )
    return FeatProxy(lf)


def ts_rsquare(feature: FeatProxy, window: int) -> FeatProxy:
    """Calculate the R-squared value of linear regression over a rolling window"""

    def rsquare_py(s: pl.Series) -> float:
        ys = s.to_list()
        n = len(ys)
        if n == 0:
            return float("nan")
        mean_x = (n - 1) / 2.0
        mean_y = sum(ys) / n
        # population variance/std（与之前 ddof=0 保持一致）
        var_x = (n * n - 1) / 12.0
        std_x = var_x ** 0.5
        # std_y（population）
        var_y = sum((y - mean_y) ** 2 for y in ys) / n
        if var_y == 0:
            return float("nan")
        std_y = var_y ** 0.5
        # covariance（population）
        cov = sum((i - mean_x) * (ys[i] - mean_y) for i in range(n)) / n
        r = cov / (std_x * std_y)
        return float(r * r)

    lf = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").rolling_map(lambda s: rsquare_py(s), window).over("vt_symbol")
    )
    return FeatProxy(lf)


def ts_resi(feature: FeatProxy, window: int) -> FeatProxy:
    """Calculate the residual of linear regression over a rolling window"""

    def resi_py(s: pl.Series) -> float:
        ys = s.to_list()
        n = len(ys)
        if n == 0:
            return float("nan")
        mean_x = (n - 1) / 2.0
        mean_y = sum(ys) / n
        num = sum((i - mean_x) * (ys[i] - mean_y) for i in range(n))
        den = sum((i - mean_x) ** 2 for i in range(n))
        if den == 0:
            return float("nan")
        m = num / den
        b = mean_y - m * mean_x
        # 残差取最后一个点
        return float(ys[-1] - (m * (n - 1) + b))

    lf = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data").rolling_map(lambda s: resi_py(s), window).over("vt_symbol")
    )
    return FeatProxy(lf)


def ts_corr(feature1: FeatProxy, feature2: FeatProxy, window: int) -> FeatProxy:
    """Calculate the correlation between two features over a rolling window"""
    lf_merged: pl.LazyFrame = feature1.df.join(
        feature2.df, on=["datetime", "vt_symbol"]
    )

    lf: pl.LazyFrame = lf_merged.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.rolling_corr("data", "data_right", window_size=window, min_samples=1)
        .over("vt_symbol")
        .alias("data"),
    )

    lf = lf.with_columns(
        pl.when(pl.col("data").is_infinite())
        .then(None)
        .otherwise(pl.col("data"))
        .alias("data")
    )

    return FeatProxy(lf)


def ts_less(feature1: FeatProxy, feature2: FeatProxy | float) -> FeatProxy:
    """Return the minimum value between two features"""
    if isinstance(feature2, FeatProxy):
        lf_merged: pl.LazyFrame = feature1.df.join(
            feature2.df, on=["datetime", "vt_symbol"]
        )
    else:
        lf_merged = feature1.df.with_columns(pl.lit(feature2).alias("data_right"))

    lf: pl.LazyFrame = lf_merged.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.min_horizontal("data", "data_right").alias("data"),
    )

    return FeatProxy(lf)


def ts_greater(feature1: FeatProxy, feature2: FeatProxy | float) -> FeatProxy:
    """Return the maximum value between two features"""
    if isinstance(feature2, FeatProxy):
        lf_merged: pl.LazyFrame = feature1.df.join(
            feature2.df, on=["datetime", "vt_symbol"]
        )

    else:
        lf_merged = feature1.df.with_columns(pl.lit(feature2).alias("data_right"))

    lf: pl.LazyFrame = lf_merged.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.max_horizontal("data", "data_right").alias("data"),
    )

    return FeatProxy(lf)


def ts_log(feature: FeatProxy) -> FeatProxy:
    """Calculate the natural logarithm of the feature"""
    lf: pl.LazyFrame = feature.df.select(
        pl.col("datetime"), pl.col("vt_symbol"), pl.col("data").log()
    )
    return FeatProxy(lf)


def ts_abs(feature: FeatProxy) -> FeatProxy:
    """Calculate the absolute value of the feature"""
    lf: pl.LazyFrame = feature.df.select(
        pl.col("datetime"), pl.col("vt_symbol"), pl.col("data").abs()
    )
    return FeatProxy(lf)
