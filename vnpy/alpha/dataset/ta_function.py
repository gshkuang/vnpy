"""
Technical Analysis Operators (bridge LazyFrame to pandas for TA-Lib)
"""

import pandas as pd
import polars as pl
import talib

from .utility import FeatProxy


def to_pd_series(feature: FeatProxy) -> pd.Series:
    """Convert LazyFrame to pandas.Series data structure"""
    df: pl.DataFrame = feature.df.collect()
    series: pd.Series = df.to_pandas().set_index(["datetime", "vt_symbol"])["data"]
    return series


def to_pl_dataframe(series: pd.Series) -> pl.LazyFrame:
    """Convert pandas.Series to LazyFrame data structure"""
    df = pl.from_pandas(series.reset_index().rename(columns={0: "data"}))
    return df.lazy()


def ta_rsi(close: FeatProxy, window: int) -> FeatProxy:
    """Calculate RSI indicator by contract"""
    close_: pd.Series = to_pd_series(close)

    result: pd.Series = talib.RSI(close_, timeperiod=window)  # type: ignore

    lf: pl.LazyFrame = to_pl_dataframe(result)
    return FeatProxy(lf)


def ta_atr(high: FeatProxy, low: FeatProxy, close: FeatProxy, window: int) -> FeatProxy:
    """Calculate ATR indicator by contract"""
    high_: pd.Series = to_pd_series(high)
    low_: pd.Series = to_pd_series(low)
    close_: pd.Series = to_pd_series(close)

    result: pd.Series = talib.ATR(high_, low_, close_, timeperiod=window)  # type: ignore

    lf: pl.LazyFrame = to_pl_dataframe(result)
    return FeatProxy(lf)
