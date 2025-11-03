"""
SQL构建器 - 用于DuckDB统计计算的SQL语句构建

将复杂的SQL构建逻辑封装到类中，提高代码可读性和维护性。
"""

from datetime import datetime
from typing import Literal, Optional


class DuckDBSQLBuilder:
    """DuckDB SQL语句构建器"""

    def __init__(self, glob_path: str, features: list[str]):
        self.glob_path = glob_path
        self.features = features

    def _build_where_clause(
        self,
        fit_start_time: Optional[datetime] = None,
        fit_end_time: Optional[datetime] = None,
    ) -> str:
        """构建WHERE子句，确保时间参数为DuckDB可解析的字符串"""

        clauses: list[str] = []
        if fit_start_time is not None:
            start_str = fit_start_time.strftime("%Y-%m-%d %H:%M:%S")
            clauses.append(f"datetime >= TIMESTAMP '{start_str}'")
        if fit_end_time is not None:
            end_str = fit_end_time.strftime("%Y-%m-%d %H:%M:%S")
            clauses.append(f"datetime <= TIMESTAMP '{end_str}'")
        return (" WHERE " + " AND ".join(clauses)) if clauses else ""

    def build_global_zscore_stats(self, where_clause: str) -> str:
        """构建全局Z-score统计SQL"""
        select_exprs = []
        for c in self.features:
            select_exprs.extend(
                [f"avg({c}) AS {c}_mean", f"stddev_samp({c}) AS {c}_std"]
            )

        return f"""
        SELECT {', '.join(select_exprs)}
        FROM read_parquet('{self.glob_path}')
        {where_clause}
        """

    def build_global_robust_stats(self, where_clause: str) -> tuple[str, str]:
        """构建全局Robust统计SQL - 返回(median_sql, mad_sql)"""
        # 中位数SQL
        median_exprs = [f"median({c}) AS {c}_median" for c in self.features]
        median_sql = f"""
        SELECT {', '.join(median_exprs)}
        FROM read_parquet('{self.glob_path}')
        {where_clause}
        """

        # MAD SQL
        mad_exprs = [
            f"median(abs({c} - med.{c}_median)) AS {c}_mad" for c in self.features
        ]
        mad_sql = f"""
        WITH med AS ({median_sql})
        SELECT {', '.join(mad_exprs)}
        FROM read_parquet('{self.glob_path}') AS t, med
        {where_clause}
        """

        return median_sql, mad_sql

    def build_cross_sectional_zscore_stats(self, where_clause: str) -> str:
        """构建截面Z-score统计SQL"""
        select_exprs = ["datetime"]
        for c in self.features:
            select_exprs.extend(
                [f"avg({c}) AS {c}_mean", f"stddev_samp({c}) AS {c}_std"]
            )

        return f"""
        SELECT {', '.join(select_exprs)}
        FROM read_parquet('{self.glob_path}')
        {where_clause}
        GROUP BY datetime
        ORDER BY datetime
        """

    def build_cross_sectional_robust_stats(self, where_clause: str) -> tuple[str, str]:
        """构建截面Robust统计SQL - 返回(median_sql, mad_sql)"""
        # 按日期分组的中位数SQL
        median_exprs = ["datetime"] + [
            f"median({c}) AS {c}_median" for c in self.features
        ]
        median_sql = f"""
        SELECT {', '.join(median_exprs)}
        FROM read_parquet('{self.glob_path}')
        {where_clause}
        GROUP BY datetime
        """

        # 按日期分组的MAD SQL
        mad_exprs = ["m.datetime"] + [
            f"median(abs(t.{c} - m.{c}_median)) AS {c}_mad" for c in self.features
        ]
        mad_sql = f"""
        WITH m AS ({median_sql})
        SELECT {', '.join(mad_exprs)}
        FROM read_parquet('{self.glob_path}') AS t 
        JOIN m USING(datetime)
        {where_clause}
        GROUP BY m.datetime
        ORDER BY m.datetime
        """

        return median_sql, mad_sql

    # ==== 新增：数据切分查询 ====
    def build_period_select(
        self, selected_cols: list[str], where_clause: str, shuffle: bool = True
    ) -> str:
        """
        构建按时间区间的 SELECT 语句，用于生成训练/验证/测试切分。

        参数
        - selected_cols: 需要选择的列（不包含 datetime/vt_symbol 时也可）
        - where_clause: 由 _build_where_clause 生成的 WHERE 子句
        - shuffle: 是否随机排序（ORDER BY random()）
        """
        cols_csv = ", ".join(selected_cols)
        order_clause = (
            "ORDER BY random()" if shuffle else "ORDER BY datetime, vt_symbol"
        )
        return f"""
        SELECT {cols_csv}
        FROM read_parquet('{self.glob_path}')
        {where_clause}
        {order_clause}
        """


def build_split_select_sql(
    glob_path: str,
    selected_cols: list[str],
    fit_start_time: Optional[datetime] = None,
    fit_end_time: Optional[datetime] = None,
    shuffle: bool = True,
) -> str:
    """
    统一入口：构建按时间区间的切分查询 SQL。
    仅选择 selected_cols，WHERE 由时间区间构造，支持随机排序。
    """
    builder = DuckDBSQLBuilder(
        glob_path, features=[]
    )  # features 未用，仅复用 where 构造
    where_clause = builder._build_where_clause(fit_start_time, fit_end_time)
    return builder.build_period_select(selected_cols, where_clause, shuffle)


def build_stats_sql(
    glob_path: str,
    features: list[str],
    method: Literal["zscore", "robust"],
    stats_type: Literal["global", "cross_sectional"],
    fit_start_time: Optional[datetime | str] = None,
    fit_end_time: Optional[datetime | str] = None,
) -> str | tuple[str, str]:
    """
    统一的SQL构建入口函数

    Returns:
        - 对于zscore方法: 返回单个SQL字符串
        - 对于robust方法: 返回(median_sql, mad_sql)元组
    """
    builder = DuckDBSQLBuilder(glob_path, features)
    where_clause = builder._build_where_clause(fit_start_time, fit_end_time)

    if stats_type == "global":
        if method == "zscore":
            sql = builder.build_global_zscore_stats(where_clause)
            # print(f"[DuckDB SQL][global][zscore]\n{sql}")
            return sql
        else:  # robust
            median_sql, mad_sql = builder.build_global_robust_stats(where_clause)
            # print(f"[DuckDB SQL][global][robust][median]\n{median_sql}")
            # print(f"[DuckDB SQL][global][robust][mad]\n{mad_sql}")
            return (median_sql, mad_sql)
    else:  # cross_sectional
        if method == "zscore":
            sql = builder.build_cross_sectional_zscore_stats(where_clause)
            # print(f"[DuckDB SQL][cross_sectional][zscore]\n{sql}")
            return sql
        else:  # robust
            median_sql, mad_sql = builder.build_cross_sectional_robust_stats(
                where_clause
            )
            # print(f"[DuckDB SQL][cross_sectional][robust][median]\n{median_sql}")
            # print(f"[DuckDB SQL][cross_sectional][robust][mad]\n{mad_sql}")
            return (median_sql, mad_sql)
