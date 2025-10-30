import polars as pl
import numpy as np

# --- 1. 参数设置 ---
w = 3  # 滚动窗口大小 (window size)
group_by_col = "vt_symbol"

# --- 2. 创建示例数据 ---
data = {
    "vt_symbol": ["A"] * 5 + ["B"] * 5,
    "time": pl.datetime_range(
        start=np.datetime64("2023-01-01"),
        end=np.datetime64("2023-01-10"),
        interval="1d",
        eager=True,
    ),
    "c": [10.0, 11.0, 10.5, 12.0, 11.5, 20.0, 22.0, 21.0, 23.0, 22.5],
}
df = pl.DataFrame(data)

print("--- 原始数据 ---")
print(df)
# --- 3. 复杂表达式分解计算 ---

# 第 1 步: 计算滞后值 (c_lag = ts_delay(c, 1))
df_result = df.with_columns(
    c_lag=pl.col("c").shift(1).over(group_by_col),
)
print(df_result)
# 第 2 步: 计算信号 (up_signal, down_signal) 和它们的滚动平均
df_result = df_result.with_columns(
    # 计算 up_signal = (c > c_lag) 并转换为 1/0
    up_signal=(pl.col("c") > pl.col("c_lag")).cast(pl.Int8),
    
    # 计算 down_signal = (c < c_lag) 并转换为 1/0
    down_signal=(pl.col("c") < pl.col("c_lag")).cast(pl.Int8),
    
    # 计算 mean_up = ts_mean(up_signal, w)
    mean_up=(
        pl.col("up_signal")
        .rolling_mean(window_size=w, min_periods=1)
        .over(group_by_col)
    ),
    
    # 计算 mean_down = ts_mean(down_signal, w)
    mean_down=(
        pl.col("down_signal")
        .rolling_mean(window_size=w, min_periods=1)
        .over(group_by_col)
    ),
)


# 第 3 步: 计算最终得分 (final_score = mean_up - mean_down)
df_result = df_result.with_columns(
    final_score=(pl.col("mean_up") - pl.col("mean_down")).alias("final_score")
)

# 打印结果，并只保留核心列
print("\n--- 计算结果 (w=3) ---")
print(df_result.select(["vt_symbol", "c", "c_lag", "up_signal", "mean_up", "mean_down", "final_score"]))