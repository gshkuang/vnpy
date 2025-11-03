"""
DuckDB + Polars 预处理用于大型因子 Parquet 文件。

目标
- 使用 DuckDB 跨多 Parquet 文件高效计算统计量。
- 使用 Polars 逐文件顺序归一化，保证内存占用可控。

特性
- 列归一化：zscore 或 robust（median/MAD）。
- 行归一化：按 datetime 的截面 zscore 或 robust。

设计
- 与 `processor.py` 风格保持一致：每个操作封装为 `process_*`。
- SQL 构造函数独立封装，逻辑清晰易维护。

说明
- DuckDB 支持 `read_parquet('.../*.parquet')` 的高效读取。
- Polars 逐文件 lazy + streaming sink，避免一次性加载。
- 全局统计通过 literal 广播，行统计通过 datetime 进行 join。
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional, Tuple

import duckdb
import polars as pl
import tqdm

from vnpy.alpha.dataset.processor import (process_stats_cs_norm,  # 统一处理函数
                                          process_stats_ts_norm)
from vnpy.alpha.dataset.sql_builder import (build_split_select_sql,
                                            build_stats_sql)
from vnpy.alpha.dataset.utility import to_datetime
from vnpy.alpha.logger import log_time_memory


def _auto_detect_feature_cols(sample_file: Path) -> list[str]:
    """Detect feature columns from a sample parquet: exclude keys."""
    schema = pl.scan_parquet(str(sample_file)).collect_schema()
    names = schema.names()
    return [c for c in names if c not in {"datetime", "vt_symbol", "label"}]


# 归一化细节函数迁移至processor.process，保留此文件为流程编排


def get_ts_stats(
    con: duckdb.DuckDBPyConnection,
    glob_path: str,
    features: list[str],
    fit_start_time: Optional[datetime | str] = None,
    fit_end_time: Optional[datetime | str] = None,
    method: Literal["zscore", "robust"] = "zscore",
) -> pl.DataFrame:
    """Processor: compute global stats across all files via DuckDB."""
    start = time.time()
    sql_result = build_stats_sql(
        glob_path=glob_path,
        features=features,
        method=method,
        stats_type="global",
        fit_start_time=fit_start_time,
        fit_end_time=fit_end_time,
    )

    if method == "zscore":
        df = con.execute(sql_result).pl()
    else:  # robust
        median_sql, mad_sql = sql_result
        med_df = con.execute(median_sql).pl()
        mad_df = con.execute(mad_sql).pl()
        df = med_df.join(mad_df, how="cross")
    end = time.time()
    print(
        f"Time cost: {end - start}, method: {method}, fit_start_time: {fit_start_time}, fit_end_time: {fit_end_time}"
    )
    return df


def process_cs_stats(
    con: duckdb.DuckDBPyConnection,
    glob_path: str,
    features: list[str],
    fit_start_time: Optional[datetime | str] = None,
    fit_end_time: Optional[datetime | str] = None,
    method: Literal["zscore", "robust"] = "zscore",
) -> pl.DataFrame:
    """Processor: compute per-datetime stats via DuckDB."""
    start = time.time()
    sql_result = build_stats_sql(
        glob_path=glob_path,
        features=features,
        method=method,
        stats_type="cross_sectional",
        fit_start_time=fit_start_time,
        fit_end_time=fit_end_time,
    )

    if method == "zscore":
        df = con.execute(sql_result).pl()
    else:  # robust
        median_sql, mad_sql = sql_result
        dt_med_df = con.execute(median_sql).pl()
        dt_mad_df = con.execute(mad_sql).pl()
        df = dt_med_df.join(dt_mad_df, on="datetime", how="inner")
    end = time.time()
    print(
        f"Time cost: {end - start}, method: {method}, fit_start_time: {fit_start_time}, fit_end_time: {fit_end_time}"
    )
    return df


@log_time_memory
def comupte_norm_stats(
    feat_dir: Path,
    stats_out_dir: Path,
    fit_start_time: Optional[datetime | str] = None,
    fit_end_time: Optional[datetime | str] = None,
    row_method: Literal["zscore", "robust"] = "zscore",
    col_method: Literal["zscore", "robust"] = "robust",
    batch_size=100,
    con=duckdb.connect(),
) -> None:
    """
    Compute global and per-datetime statistics across all parquet files using DuckDB.

    Writes parquet files under `stats_out_dir`:
    - global: `<stats_out_dir>/global_stats.parquet`
    - per-datetime: `<stats_out_dir>/datetime_stats.parquet`

    Returns list of feature column names.
    """
    # Determine feature columns
    parquet_files = sorted(feat_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {feat_dir}")

    feature_cols = _auto_detect_feature_cols(parquet_files[0])
    features = list(feature_cols)

    # ---------- Global stats ----------
    glob_path = str((feat_dir / "*.parquet").as_posix())
    global_df = pl.DataFrame()
    for idx in range(0, len(features), batch_size):
        feature_batch = features[idx : idx + batch_size]
        global_df = global_df.hstack(
            get_ts_stats(
                con=con,
                glob_path=glob_path,
                features=feature_batch,
                fit_start_time=fit_start_time,
                fit_end_time=fit_end_time,
                method=col_method,
            )
        )
    global_df.write_parquet(stats_out_dir / "global_stats.parquet")
    # Log stats preview
    print("[Stats][global] shape:", global_df.shape)

    # ---------- Per-datetime stats ----------
    dt_df = process_cs_stats(
        con=con,
        glob_path=glob_path,
        features=["label"],  # 只针对label做cs归一化
        method=row_method,
    )
    dt_df.write_parquet(stats_out_dir / "datetime_stats.parquet")
    # Log stats preview
    print("[Stats][datetime] shape:", dt_df.shape)


@log_time_memory
def normalize_feat(
    feat_dir: Path,
    stats_dir: Path,
    out_dir: Path,
    row_method: Literal["zscore", "robust"] = "zscore",
    col_method: Literal["zscore", "robust"] = "robust",
) -> None:
    """
    Sequentially normalize each parquet with Polars using precomputed stats.

    - Row-wise: join with per-datetime stats and standardize per feature.
    - Column-wise: broadcast global stats via literals for per-feature standardization.
    - Writes output files to `out_dir` with the same file names.
    """

    parquet_files = sorted(feat_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {feat_dir}")
    feature_cols = _auto_detect_feature_cols(parquet_files[0])
    features = list(feature_cols)

    # 使用stats_lf进行两阶段归一化：先列（global），后行（cross-sectional）
    global_stats_path = stats_dir / "global_stats.parquet"
    global_stats_lf = pl.scan_parquet(str(global_stats_path))
    dt_stats_path = stats_dir / "datetime_stats.parquet"
    dt_lf = pl.scan_parquet(str(dt_stats_path))

    for file in tqdm.tqdm(parquet_files, desc="Normalizing features"):
        lf = pl.scan_parquet(str(file))
        lf = process_stats_ts_norm(lf, global_stats_lf, features, col_method)
        lf = process_stats_cs_norm(lf, dt_lf, ["label"], row_method)
        lf = lf.drop_nans()
        lf = lf.drop_nulls()
        out_path = out_dir / file.name
        lf.sink_parquet(str(out_path))


@log_time_memory
def save_duckdb_splits(
    feat_dir: Path,
    out_dir: Path,
    train_period: Tuple[datetime | str, datetime | str],
    valid_period: Tuple[datetime | str, datetime | str],
    test_period: Tuple[datetime | str, datetime | str],
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> None:
    """
    使用 DuckDB 按时间区间直接切分数据，并保存到指定目录，不返回数据。

    生成文件：
    - `<out_dir>/train.parquet`
    - `<out_dir>/valid.parquet`
    - `<out_dir>/test.parquet`
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    if seed is not None:
        try:
            con.execute(f"PRAGMA random_seed={int(seed)}")
        except Exception:
            pass

    glob_path = str((feat_dir / "*.parquet").as_posix())

    sample_files = sorted(feat_dir.glob("*.parquet"))
    if not sample_files:
        raise FileNotFoundError(f"No parquet files found in {feat_dir}")
    schema_names = pl.scan_parquet(str(sample_files[0])).collect_schema().names()
    feature_cols = [
        c for c in schema_names if c not in {"datetime", "vt_symbol", "label"}
    ]

    @log_time_memory
    def _save_period(start: datetime | str, end: datetime | str, name: str) -> None:
        # 选择包含键列 + 特征列 + label，确保下游可用键列
        selected_cols = ["datetime", "vt_symbol"] + feature_cols + ["label"]
        sql = build_split_select_sql(
            glob_path,
            selected_cols,
            fit_start_time=to_datetime(start),
            fit_end_time=to_datetime(end),
            shuffle=shuffle,
        )
        df = con.execute(sql).pl()
        df.write_parquet(out_dir / f"{name}.parquet")
        print(f"[Split][{name}] shape:", df.shape, " start:", start, " end:", end)

    _save_period(*train_period, name="train")
    _save_period(*valid_period, name="valid")
    _save_period(*test_period, name="test")


def feat_norm_pipeline(
    feat_dir: Path,
    stats_dir: Path,
    out_dir: Path,
    fit_start: Optional[datetime] = None,
    fit_end: Optional[datetime] = None,
    row_method: Literal["zscore", "robust"] = "zscore",
    col_method: Literal["zscore", "robust"] = "robust",
) -> None:
    stats_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    comupte_norm_stats(
        feat_dir=feat_dir,
        stats_out_dir=stats_dir,
        fit_start_time=fit_start,
        fit_end_time=fit_end,
        row_method=row_method,
        col_method=col_method,
    )
    normalize_feat(
        feat_dir=feat_dir,
        stats_dir=stats_dir,
        out_dir=out_dir,
        row_method=row_method,
        col_method=col_method,
    )


if __name__ == "__main__":
    # 仅演示切分保存，不返回数据
    lab_path = Path("/home/lai/test_v1/lab/crypto_1m")
    feat_dir = lab_path.joinpath("feat_norm")
    split_dir = lab_path.joinpath("splits")

    train_period = (
        datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        datetime(2022, 12, 31, 23, 59, 0, tzinfo=timezone.utc),
    )
    valid_period = (
        datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        datetime(2023, 6, 30, 23, 59, 0, tzinfo=timezone.utc),
    )
    test_period = (
        datetime(2023, 7, 1, 0, 0, 0, tzinfo=timezone.utc),
        datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    )

    save_duckdb_splits(
        feat_dir=feat_dir,
        out_dir=split_dir,
        train_period=train_period,
        valid_period=valid_period,
        test_period=test_period,
        shuffle=True,
        seed=42,
    )
