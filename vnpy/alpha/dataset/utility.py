from datetime import datetime
from enum import Enum
from typing import Union

import polars as pl


class FeatProxy:
    """Feature data proxy backed by LazyFrame"""

    def __init__(self, df: pl.DataFrame | pl.LazyFrame) -> None:
        """Constructor accepting eager DataFrame or LazyFrame"""
        lf: pl.LazyFrame = df.lazy() if isinstance(df, pl.DataFrame) else df
        # ensure last column is named as data
        last_name: str = lf.columns[-1]
        self.df: pl.LazyFrame = lf.rename({last_name: "data"})

        # Note: numeric expressions should place variables before numbers, e.g. a * 2

    @classmethod
    def col2proxy(cls, df: pl.DataFrame | pl.LazyFrame, column: str) -> "FeatProxy":
        lf: pl.LazyFrame = df.lazy() if isinstance(df, pl.DataFrame) else df
        return cls(lf.select(["datetime", "vt_symbol", pl.col(column).alias("data")]))

    def expr2proxy(self, expr: pl.Expr) -> "FeatProxy":
        """Convert expression to feature LazyFrame"""
        lf: pl.LazyFrame = self.df.select([
            pl.col("datetime"),
            pl.col("vt_symbol"),
            expr.alias("data")
        ])
        return FeatProxy(lf)

    def __add__(self, other: Union["FeatProxy", int, float]) -> "FeatProxy":
        """Addition operation"""
        if isinstance(other, FeatProxy):
            merged = self.df.join(other.df, on=["datetime", "vt_symbol"])  # data_right suffix on right
            expr = pl.col("data") + pl.col("data_right")
            return FeatProxy(merged.select(["datetime", "vt_symbol", expr.alias("data")]))
        else:
            return self.expr2proxy(pl.col("data") + pl.lit(other))

    def __sub__(self, other: Union["FeatProxy", int, float]) -> "FeatProxy":
        """Subtraction operation"""
        if isinstance(other, FeatProxy):
            merged = self.df.join(other.df, on=["datetime", "vt_symbol"])  # data_right suffix on right
            expr = pl.col("data") - pl.col("data_right")
            return FeatProxy(merged.select(["datetime", "vt_symbol", expr.alias("data")]))
        else:
            return self.expr2proxy(pl.col("data") - pl.lit(other))

    def __mul__(self, other: Union["FeatProxy", int, float]) -> "FeatProxy":
        """Multiplication operation"""
        if isinstance(other, FeatProxy):
            merged = self.df.join(other.df, on=["datetime", "vt_symbol"])  # data_right suffix on right
            expr = pl.col("data") * pl.col("data_right")
            return FeatProxy(merged.select(["datetime", "vt_symbol", expr.alias("data")]))
        else:
            return self.expr2proxy(pl.col("data") * pl.lit(other))

    def __rmul__(self, other: Union["FeatProxy", int, float]) -> "FeatProxy":
        """Right multiplication operation"""
        return self.__mul__(other)

    def __truediv__(self, other: Union["FeatProxy", int, float]) -> "FeatProxy":
        """Division operation"""
        if isinstance(other, FeatProxy):
            merged = self.df.join(other.df, on=["datetime", "vt_symbol"])  # data_right suffix on right
            expr = pl.col("data") / pl.col("data_right")
            return FeatProxy(merged.select(["datetime", "vt_symbol", expr.alias("data")]))
        else:
            return self.expr2proxy(pl.col("data") / pl.lit(other))

    def __abs__(self) -> "FeatProxy":
        """Get absolute value"""
        return self.expr2proxy(pl.col("data").abs())

    def __gt__(self, other: Union["FeatProxy", int, float]) -> "FeatProxy":
        """Greater than comparison"""
        if isinstance(other, FeatProxy):
            merged = self.df.join(other.df, on=["datetime", "vt_symbol"])  # data_right suffix on right
            expr = pl.col("data") > pl.col("data_right")
            return FeatProxy(merged.select(["datetime", "vt_symbol", expr.alias("data")]))
        else:
            return self.expr2proxy(pl.col("data") > pl.lit(other))

    def __ge__(self, other: Union["FeatProxy", int, float]) -> "FeatProxy":
        """Greater than or equal comparison"""
        if isinstance(other, FeatProxy):
            merged = self.df.join(other.df, on=["datetime", "vt_symbol"])  # data_right suffix on right
            expr = pl.col("data") >= pl.col("data_right")
            return FeatProxy(merged.select(["datetime", "vt_symbol", expr.alias("data")]))
        else:
            return self.expr2proxy(pl.col("data") >= pl.lit(other))

    def __lt__(self, other: Union["FeatProxy", int, float]) -> "FeatProxy":
        """Less than comparison"""
        if isinstance(other, FeatProxy):
            merged = self.df.join(other.df, on=["datetime", "vt_symbol"])  # data_right suffix on right
            expr = pl.col("data") < pl.col("data_right")
            return FeatProxy(merged.select(["datetime", "vt_symbol", expr.alias("data")]))
        else:
            return self.expr2proxy(pl.col("data") < pl.lit(other))

    def __le__(self, other: Union["FeatProxy", int, float]) -> "FeatProxy":
        """Less than or equal comparison"""
        if isinstance(other, FeatProxy):
            merged = self.df.join(other.df, on=["datetime", "vt_symbol"])  # data_right suffix on right
            expr = pl.col("data") <= pl.col("data_right")
            return FeatProxy(merged.select(["datetime", "vt_symbol", expr.alias("data")]))
        else:
            return self.expr2proxy(pl.col("data") <= pl.lit(other))

    def __eq__(self, other: Union["FeatProxy", int, float]) -> "FeatProxy":    # type: ignore
        """Equal comparison"""
        if isinstance(other, FeatProxy):
            merged = self.df.join(other.df, on=["datetime", "vt_symbol"])  # data_right suffix on right
            expr = pl.col("data") == pl.col("data_right")
            return FeatProxy(merged.select(["datetime", "vt_symbol", expr.alias("data")]))
        else:
            return self.expr2proxy(pl.col("data") == pl.lit(other))



def to_datetime(arg: datetime | str) -> datetime:
    """Convert time data type"""
    if isinstance(arg, str):
        if "-" in arg:
            fmt: str = "%Y-%m-%d"
        else:
            fmt = "%Y%m%d"

        return datetime.strptime(arg, fmt)
    else:
        return arg


class Segment(Enum):
    """Data segment enumeration values"""

    TRAIN = 1
    VALID = 2
    TEST = 3
