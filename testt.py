import polars as pl
from typing import cast

# === 构造示例数据 ===
data = pl.DataFrame({
    "datetime": [f"2024-01-01 00:0{i}:00" for i in range(10)],
    "vt_symbol": ["BTCUSDT"] * 10,
    "data": [1, 3, 2, 5, 4, 3, 6, 7, 1, 2],
})

# ✅ 转为 Polars Datetime 类型（核心）
data = data.with_columns(pl.col("datetime").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S"))

window = 3
period = "m"

# === 版本 1：rolling_map + over ===
lf1 = data.lazy().select(
    pl.col("datetime"),
    pl.col("vt_symbol"),
    pl.col("data")
    .rolling_map(lambda s: cast(int, s.arg_max()), window)
    .over("vt_symbol")
    .alias("argmax_idx"),
)

# === 版本 2：rolling(period=...) + agg ===
lf2 = data.lazy().rolling(
    index_column="datetime",
    by="vt_symbol",
    period=f"{window}{period}",
    closed="right"
).agg(pl.col("data").arg_max().alias("argmax_idx"))

# === 输出对比 ===
print("=== rolling_map(over) 版本 ===")
print(lf1.collect())

print("\n=== rolling(period, by) 版本 ===")
print(lf2.collect())

print(data.rolling(
    index_column="datetime",
    by="vt_symbol",
    period=f"{window}{period}",
    closed="right"
).agg(pl.col("data")))