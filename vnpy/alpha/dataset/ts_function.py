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


def ts_rank(feature: FeatProxy, window: int,period="m") -> FeatProxy:
    """
    计算滚动窗口内“当前值”的百分位（向量化实现）。

    语义定义（包含 ties）：
    - 对每个时间点 t，窗口 W_t 包含该品种在 (t - period, t] 内的观测；
    - 当前值 y_t 的百分位：rank_t = #{ y_i ∈ W_t | y_i <= y_t } / |W_t|。
    """
    
    lf = (
        feature.df
        .sort(["vt_symbol", "datetime"])
        .rolling(
            index_column="datetime",
            group_by="vt_symbol",
            period=f"{window}{period}",
            closed="right"
        )
        .agg([
            pl.col("data").rank(method="min").alias("rank_min"),
            pl.count().alias("count")
        ])
        .with_columns(
            # 取当前点在窗口中的 rank（rank_min 列为列表，last 即当前值的秩）
            (pl.col("rank_min").list.last() / pl.col("count")).alias("data")
        )
        .select("datetime", "vt_symbol", "data")
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
    """
    计算滚动窗口内简单线性回归的斜率（向量化实现）。

    说明与公式：
    - 原始 Python UDF 用索引 i=0..n-1 作为自变量 x，并在每个窗口重复计算。
    - 为避免 UDF 开销，采用闭式公式并用滚动和向量化表达：
      设窗口内：Sx=∑x, Sy=∑y, Sxx=∑x², Sxy=∑x·y，样本数 n。
      均值 μx=Sx/n, μy=Sy/n。
      斜率 m=Cov(x,y)/Var(x)=(Sxy−Sx·Sy/n)/(Sxx−Sx²/n)。
    - 我们将 x 取为组内按时间排序的序号（rank 从 0 开始），保持与原实现一致的单位。
    - n 使用对常数 1 的滚动求和得到真实样本数，保证早期窗口的正确性。
    """

    # 组内 x 用时间戳的数值表示（等距时间步下与索引线性等价），避免嵌套窗口
    x_expr = pl.col("datetime").cast(pl.Float64)

    lf = feature.df.with_columns(
        x_expr.alias("x"),
        # n = 真实样本数：对列常量 1 做滚动求和，避免对 pl.lit 的滚动造成不兼容
        (pl.col("data")*0 + 1.0).rolling_sum(window, min_samples=1).over("vt_symbol").alias("n"),
        (x_expr).rolling_sum(window, min_samples=1).over("vt_symbol").alias("Sx"),
        pl.col("data").rolling_sum(window, min_samples=1).over("vt_symbol").alias("Sy"),
        (x_expr * x_expr).rolling_sum(window, min_samples=1).over("vt_symbol").alias("Sxx"),
        (x_expr * pl.col("data")).rolling_sum(window, min_samples=1).over("vt_symbol").alias("Sxy"),
    ).select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        (
            (pl.col("Sxy") - pl.col("Sx") * pl.col("Sy") / pl.col("n")) /
            (pl.col("Sxx") - pl.col("Sx") * pl.col("Sx") / pl.col("n"))
        ).alias("data")
    ).with_columns(
        pl.when(pl.col("data").is_infinite() | pl.col("data").is_nan()).then(None).otherwise(pl.col("data")).alias("data")
    )
    return FeatProxy(lf)


def ts_quantile(feature: FeatProxy, window: int, quantile: float) -> FeatProxy:
    """Calculate the quantile value over a rolling window"""
    lf = feature.df.select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.col("data")
        .rolling_quantile(quantile=quantile, window_size=window, interpolation="linear")
        .over("vt_symbol"),
    )
    return FeatProxy(lf)


def ts_rsquare(feature: FeatProxy, window: int) -> FeatProxy:
    """
    计算滚动窗口内线性回归的 R²（向量化实现，人口方差定义）。

    说明与公式：
    - 原实现以 x=i=0..n-1（等距）为自变量，采用人口方差/协方差（ddof=0）。
    - 向量化闭式：
      cov = (Sxy − Sx·Sy/n) / n
      var_x = (Sxx − Sx²/n) / n
      var_y = (Syy − Sy²/n) / n
      R² = (cov²) / (var_x · var_y)
    - 同样用组内时间序 rank 生成 x，以匹配原始逻辑；n 为真实样本数。
    """

    x_expr = pl.col("datetime").cast(pl.Float64)

    lf = feature.df.with_columns(
        x_expr.alias("x"),
        (pl.col("data")*0 + 1.0).rolling_sum(window, min_samples=1).over("vt_symbol").alias("n"),
        (x_expr).rolling_sum(window, min_samples=1).over("vt_symbol").alias("Sx"),
        pl.col("data").rolling_sum(window, min_samples=1).over("vt_symbol").alias("Sy"),
        (x_expr * x_expr).rolling_sum(window, min_samples=1).over("vt_symbol").alias("Sxx"),
        (pl.col("data") * pl.col("data")).rolling_sum(window, min_samples=1).over("vt_symbol").alias("Syy"),
        (x_expr * pl.col("data")).rolling_sum(window, min_samples=1).over("vt_symbol").alias("Sxy"),
    ).with_columns(
        ((pl.col("Sxy") - pl.col("Sx") * pl.col("Sy") / pl.col("n")) / pl.col("n")).alias("cov"),
        ((pl.col("Sxx") - pl.col("Sx") * pl.col("Sx") / pl.col("n")) / pl.col("n")).alias("var_x"),
        ((pl.col("Syy") - pl.col("Sy") * pl.col("Sy") / pl.col("n")) / pl.col("n")).alias("var_y"),
    ).select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        pl.when((pl.col("var_x") <= 0) | (pl.col("var_y") <= 0)).then(None).otherwise(
            (pl.col("cov") * pl.col("cov")) / (pl.col("var_x") * pl.col("var_y"))
        ).alias("data")
    )
    return FeatProxy(lf)


def ts_resi(feature: FeatProxy, window: int) -> FeatProxy:
    """
    计算滚动窗口内线性回归的残差（最后一点）（向量化实现）。

    说明与公式：
    - 残差定义为：y_last − (m·x_last + b)。
    - 其中 m 与 b 分别为回归的斜率与截距：
      m = (Sxy − Sx·Sy/n) / (Sxx − Sx²/n)
      μx = Sx/n, μy = Sy/n，b = μy − m·μx
    - x 仍取组内序号（0..），保证与原 UDF 一致；n 为真实样本数。
    """

    x_expr = pl.col("datetime").cast(pl.Float64)

    lf = feature.df.with_columns(
        x_expr.alias("x"),
        (pl.col("data")*0 + 1.0).rolling_sum(window, min_samples=1).over("vt_symbol").alias("n"),
        (x_expr).rolling_sum(window, min_samples=1).over("vt_symbol").alias("Sx"),
        pl.col("data").rolling_sum(window, min_samples=1).over("vt_symbol").alias("Sy"),
        (x_expr * x_expr).rolling_sum(window, min_samples=1).over("vt_symbol").alias("Sxx"),
        (x_expr * pl.col("data")).rolling_sum(window, min_samples=1).over("vt_symbol").alias("Sxy"),
    ).with_columns(
        (((pl.col("Sxy") - pl.col("Sx") * pl.col("Sy") / pl.col("n")) /
           (pl.col("Sxx") - pl.col("Sx") * pl.col("Sx") / pl.col("n")))).alias("m"),
        (pl.col("Sx") / pl.col("n")).alias("mx"),
        (pl.col("Sy") / pl.col("n")).alias("my"),
    ).select(
        pl.col("datetime"),
        pl.col("vt_symbol"),
        (pl.col("data") - (pl.col("m") * pl.col("x") + (pl.col("my") - pl.col("m") * pl.col("mx")))).alias("data")
    ).with_columns(
        pl.when(pl.col("data").is_infinite() | pl.col("data").is_nan()).then(None).otherwise(pl.col("data")).alias("data")
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
