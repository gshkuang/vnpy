#!/usr/bin/env python3
"""
测试polars LazyFrame.rolling函数，使用period='2i'参数
"""

import polars as pl


def test_rolling_with_2i_period():
    """
    测试rolling函数使用period='2i'参数
    '2i'表示基于索引的滚动窗口，每个窗口包含2个索引位置的数据
    """
    # 创建测试数据
    data = {"index_col": [1, 2, 3, 4, 5, 6], "value": [10, 20, 30, 40, 50, 60]}

    # 创建LazyFrame
    lf = pl.LazyFrame(data)

    # 使用rolling函数，period='2i'表示每个窗口包含2个索引位置
    result = (
        lf.rolling(index_column="index_col", period="2i")
        .agg(
            [
                pl.col("value").rank(method="min").alias("rank_min"),
                pl.col("value").count().alias("rolling_count"),
                pl.col("value").first().alias("first_value"),
                pl.col("value").last().alias("last_value"),
            ]
        )
        .collect()
    )
    print(result)
    result = (
        lf.rolling(index_column="index_col", period="2i")
        .agg(
            [
                pl.col("value").rank(method="min").alias("rank_min"),
                pl.col("value").count().alias("rolling_count"),
            ]
        )
        .with_columns(
            # 取当前点在窗口中的 rank（rank_min 列为列表，last 即当前值的秩）
            (pl.col("rank_min").list.last() / pl.col("rolling_count")).alias("data")
        )
        .collect()
    )

    print("测试结果:")
    print(result)

    # 验证结果
    expected_sums = [10, 30, 50, 70, 90, 110]  # 每个窗口的和
    expected_means = [10.0, 15.0, 25.0, 35.0, 45.0, 55.0]  # 每个窗口的平均值
    expected_counts = [1, 2, 2, 2, 2, 2]  # 每个窗口的计数

    actual_sums = result["rolling_sum"].to_list()
    actual_means = result["rolling_mean"].to_list()
    actual_counts = result["rolling_count"].to_list()

    assert actual_sums == expected_sums, f"期望和: {expected_sums}, 实际和: {actual_sums}"
    assert (
        actual_means == expected_means
    ), f"期望平均值: {expected_means}, 实际平均值: {actual_means}"
    assert (
        actual_counts == expected_counts
    ), f"期望计数: {expected_counts}, 实际计数: {actual_counts}"

    print("✅ 所有断言通过!")


def test_rolling_with_2i_period_grouped():
    """
    测试rolling函数使用period='2i'参数，并按组分组
    """
    # 创建包含分组的测试数据
    data = {
        "index_col": [1, 2, 3, 4, 1, 2, 3, 4],
        "group": ["A", "A", "A", "A", "B", "B", "B", "B"],
        "value": [10, 20, 30, 40, 15, 25, 35, 45],
    }

    # 创建LazyFrame
    lf = pl.LazyFrame(data)

    # 使用rolling函数，按组分组
    result = (
        lf.rolling(index_column="index_col", period="2i", group_by="group")
        .agg(
            [
                pl.col("value").sum().alias("rolling_sum"),
                pl.col("value").mean().alias("rolling_mean"),
            ]
        )
        .collect()
    )

    print("\n分组测试结果:")
    print(result)

    # 验证分组A的结果
    group_a_result = result.filter(pl.col("group") == "A").sort("index_col")
    group_b_result = result.filter(pl.col("group") == "B").sort("index_col")

    print("✅ 分组测试完成!")


def test_rolling_with_2i_period_different_closed():
    """
    测试rolling函数使用period='2i'参数，测试不同的closed参数
    """
    data = {"index_col": [1, 2, 3, 4, 5], "value": [10, 20, 30, 40, 50]}

    lf = pl.LazyFrame(data)

    # 测试closed='right' (默认)
    result_right = (
        lf.rolling(index_column="index_col", period="2i", closed="right")
        .agg(pl.col("value").sum().alias("rolling_sum"))
        .collect()
    )

    # 测试closed='left'
    result_left = (
        lf.rolling(index_column="index_col", period="2i", closed="left")
        .agg(pl.col("value").sum().alias("rolling_sum"))
        .collect()
    )

    # 测试closed='both'
    result_both = (
        lf.rolling(index_column="index_col", period="2i", closed="both")
        .agg(pl.col("value").sum().alias("rolling_sum"))
        .collect()
    )

    print("\nclosed='right'结果:")
    print(result_right)
    print("\nclosed='left'结果:")
    print(result_left)
    print("\nclosed='both'结果:")
    print(result_both)

    print("✅ 不同closed参数测试完成!")


if __name__ == "__main__":
    print("开始测试polars rolling函数，period='2i'...")

    try:
        test_rolling_with_2i_period()
        test_rolling_with_2i_period_grouped()
        test_rolling_with_2i_period_different_closed()
        print("\n🎉 所有测试通过!")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        raise
