import time
from datetime import datetime
from typing import cast
from collections.abc import Callable
from multiprocessing import get_context
from multiprocessing.context import BaseContext

import polars as pl
import pandas as pd
from tqdm import tqdm
from alphalens.utils import get_clean_factor_and_forward_returns  # type: ignore
from alphalens.tears import create_full_tear_sheet  # type: ignore
import pickle
from pathlib import Path
from ..logger import logger
from .utility import to_datetime, Segment, FeatProxy


class AlphaDataset:
    """Alpha dataset template class"""

    def __init__(
        self,
        df: pl.DataFrame,
        train_period: tuple[str, str],
        valid_period: tuple[str, str],
        test_period: tuple[str, str],
        process_type: str = "append",
    ) -> None:
        """Constructor"""
        self.df: pl.DataFrame = df

        # DataFrames for processed data
        self.result_df: pl.DataFrame
        # self.raw_df: pl.DataFrame
        self.infer_df: pl.DataFrame
        self.learn_df: pl.DataFrame

        # New version
        self.data_periods: dict[Segment, tuple[str, str]] = {
            Segment.TRAIN: train_period,
            Segment.VALID: valid_period,
            Segment.TEST: test_period,
        }

        # Feature storage using DataProxy (LazyFrame-based)
        self.features: dict[str, FeatProxy] = {}
        self.feature_results: dict[str, pl.DataFrame] = {}
        self.label_feature: FeatProxy | None = None

        self.process_type: str = process_type
        self.infer_processors: list = []
        self.learn_processors: list = []

    def add_feature(
        self,
        name: str,
        feature: FeatProxy | None = None,
        result: pl.DataFrame | None = None,
    ) -> None:
        """
        Add a feature. Prefer DataProxy; optionally accept a ready DataFrame result.
        """
        if feature is not None and result is not None:
            raise ValueError("Only one of 'feature' or 'result' can be provided")

        if feature is not None:
            self.features[name] = feature
        elif result is not None:
            self.feature_results[name] = result

    def set_label(self, feature: FeatProxy) -> None:
        """
        Set the label feature
        """
        self.label_feature = feature

    def add_processor(
        self, task: str, processor: Callable[[pl.DataFrame], None]
    ) -> None:
        """
        Add a feature preprocessor
        """
        if task == "infer":
            self.infer_processors.append(processor)
        else:
            self.learn_processors.append(processor)

    def prepare_data(
        self, filters: dict | None = None, max_workers: int | None = None
    ) -> None:
        """
        Generate required data
        """
        # 收集各特征计算后的 DataFrame（包含键列与命名列）
        feat_dfs: list[pl.DataFrame] = []

        # Calculate DataProxy features
        logger.info("开始计算 DataProxy 因子特征")
        t_feat_start = time.time()

        tasks: list[tuple[str, FeatProxy]] = list(self.features.items())
        if self.label_feature is not None:
            tasks.append(("label", self.label_feature))

        if tasks:
            # Simplify: compute features sequentially to avoid pickling LazyFrame
            args: list[tuple[str, FeatProxy]] = [(name, feature) for name, feature in tasks]
            context: BaseContext = get_context("spawn")

            with context.Pool(processes=max_workers) as pool:
                            # Calculate all expressions in parallel
                it = pool.imap(calculate_feature, args)
                for result in tqdm(it, total=len(args)):
                    feat_dfs.append(result)

        t_feat_end = time.time()
        logger.info(f"DataProxy 因子计算完成 | 数量: {len(tasks)} | 耗时: {t_feat_end - t_feat_start:.3f}s")

        # 按键 join 合并所有特征与标签
        t_join_start = time.time()
        self.result_df = self.df
        for i, fdf in enumerate(feat_dfs, start=1):
            self.result_df = self.result_df.join(fdf, on=["datetime", "vt_symbol"], how="inner")
        t_join_end = time.time()
        logger.info(f"按键合并因子到 DataFrame | 合并数量: {len(feat_dfs)} | 耗时: {t_join_end - t_join_start:.3f}s")

        # Merge result data factor features
        logger.info("开始合并结果数据因子特征")
        t_merge_start = time.time()

        for name, feature_result in tqdm(self.feature_results.items()):
            feature_result = feature_result.rename({"data": name})
            self.result_df = self.result_df.join(
                feature_result, on=["datetime", "vt_symbol"], how="inner"
            )

        t_merge_end = time.time()
        logger.info(
            f"合并结果数据因子特征完成 | 数量: {len(self.feature_results)} | 耗时: {t_merge_end - t_merge_start:.3f}s"
        )

        # Generate raw data
        t_raw_start = time.time()
        raw_df = self.result_df.fill_null(float("nan"))
        t_raw_end = time.time()
        logger.info(f"生成 raw_df（填充 NaN）| 耗时: {t_raw_end - t_raw_start:.3f}s")

        if filters:
            logger.info("开始筛选成分股数据")
            t_filter_start = time.time()

            # 构造区间 DataFrame 并转换为 datetime
            ranges_rows: list[dict] = []
            for vt_symbol, ranges in filters.items():
                for start, end in ranges:
                    ranges_rows.append(
                        {
                            "vt_symbol": vt_symbol,
                            "range_start": to_datetime(start),
                            "range_end": to_datetime(end),
                        }
                    )

            ranges_df = pl.DataFrame(ranges_rows).sort(["vt_symbol", "range_start"])

            # join_asof 需要按时间列排序
            # raw_df = raw_df.sort(["vt_symbol", "datetime"])
            # 对齐每条数据到最近的区间起点（同 vt_symbol）
            raw_df = raw_df.lazy().join_asof(
                ranges_df.lazy(),
                left_on="datetime",
                right_on="range_start",
                by="vt_symbol",
                strategy="backward",
            ).filter(
                pl.col("range_start").is_not_null()
                & (pl.col("datetime") <= pl.col("range_end"))
            ).select(raw_df.columns).collect()

            t_filter_end = time.time()
            logger.info(f"筛选成分股数据完成 | 合计合并条目: {len(filters)} | 耗时: {t_filter_end - t_filter_start:.3f}s")

        # Only keep feature columns
        t_select_sort_start = time.time()
        select_columns: list[str] = ["datetime", "vt_symbol"] + raw_df.columns[
            self.df.width :
        ]
        raw_df = raw_df.select(select_columns).sort(["datetime", "vt_symbol"])
        t_select_sort_end = time.time()
        logger.info(f"选择特征列并排序完成 | 列数: {len(select_columns)} | 耗时: {t_select_sort_end - t_select_sort_start:.3f}s")

        # Generate inference data
        self.infer_df = raw_df
        print(self.infer_df.head())
        print(self.infer_df.shape)
        print(self.infer_processors)
        for i, processor in enumerate(self.infer_processors, start=1):
            t_proc_start = time.time()
            self.infer_df = processor(df=self.infer_df)
            t_proc_end = time.time()
            proc_name = getattr(processor, "__name__", processor.__class__.__name__)
            logger.info(f"infer 处理器[{i}] {proc_name} 完成 | 耗时: {t_proc_end - t_proc_start:.3f}s")

        # Generate learning data
        t_learn_assign_start = time.time()
        if self.process_type == "append":
            self.learn_df = self.infer_df
        else:
            self.learn_df = raw_df
        t_learn_assign_end = time.time()
        logger.info(f"学习数据赋值完成（process_type={self.process_type}）| 耗时: {t_learn_assign_end - t_learn_assign_start:.3f}s")

        for i, processor in enumerate(self.learn_processors, start=1):
            t_proc_start = time.time()
            self.learn_df = processor(df=self.learn_df)
            t_proc_end = time.time()
            proc_name = getattr(processor, "__name__", processor.__class__.__name__)
            logger.info(f"learn 处理器[{i}] {proc_name} 完成 | 耗时: {t_proc_end - t_proc_start:.3f}s")

    # def fetch_raw(self, segment: Segment) -> pl.DataFrame:
    #     """
    #     Get raw data for a specific segment
    #     """
    #     start, end = self.data_periods[segment]
    #     return query_by_time(self.raw_df, start, end)

    def fetch_infer(self, segment: Segment) -> pl.DataFrame:
        """
        Get inference data for a specific segment
        """
        start, end = self.data_periods[segment]
        return query_by_time(self.infer_df, start, end)

    def fetch_learn(self, segment: Segment) -> pl.DataFrame:
        """
        Get learning data for a specific segment
        """
        start, end = self.data_periods[segment]
        return query_by_time(self.learn_df, start, end)

    def show_feature_performance(self, name: str) -> None:
        """
        Perform performance analysis for a feature
        """
        starts: list[datetime] = []
        ends: list[datetime] = []

        for period in self.data_periods.values():
            starts.append(to_datetime(period[0]))
            ends.append(to_datetime(period[1]))

        start: datetime = min(starts)
        end: datetime = max(ends)

        # Select range
        df: pl.DataFrame = query_by_time(self.result_df, start, end)

        # Extract feature
        feature_df: pd.DataFrame = df.select(
            ["datetime", "vt_symbol", name]
        ).to_pandas()
        feature_df.set_index(["datetime", "vt_symbol"], inplace=True)
        # print(feature_df.head())
        freq: str = pd.infer_freq(feature_df.index.levels[0])
        print(f"infer freq: {freq}")
        feature_df.index.levels[0].freq = freq

        feature_s: pd.Series = feature_df[name]

        # Extract price
        price_df: pd.DataFrame = df.select(
            ["datetime", "vt_symbol", "close"]
        ).to_pandas()
        price_df = price_df.pivot(index="datetime", columns="vt_symbol", values="close")

        # Merge data
        clean_data: pd.DataFrame = get_clean_factor_and_forward_returns(
            feature_s, price_df, quantiles=10
        )

        # Perform analysis
        create_full_tear_sheet(clean_data)

    def show_signal_performance(self, signal: pl.DataFrame) -> None:
        """
        Perform performance analysis for prediction signals
        """
        # Get signal start and end times
        start: datetime = cast(datetime, signal["datetime"].min())
        end: datetime = cast(datetime, signal["datetime"].max())

        # Select range
        df: pl.DataFrame = query_by_time(self.result_df, start, end)

        # Extract feature
        signal_df: pd.DataFrame = signal.to_pandas()
        signal_df.set_index(["datetime", "vt_symbol"], inplace=True)
        signal_s: pd.Series = signal_df["signal"]
        freq: str = pd.infer_freq(signal_df.index.levels[0])
        print(f"infer freq: {freq}")
        signal_df.index.levels[0].freq = freq

        # Extract price
        price_df: pd.DataFrame = df.select(
            ["datetime", "vt_symbol", "close"]
        ).to_pandas()
        price_df = price_df.pivot(index="datetime", columns="vt_symbol", values="close")

        # Merge data
        clean_data: pd.DataFrame = get_clean_factor_and_forward_returns(
            signal_s, price_df, max_loss=1.0, quantiles=10
        )

        # Perform analysis
        create_full_tear_sheet(clean_data)

    def save(self, path: Path ) -> None:
        """
        将数据集保存到目录：每个DataFrame为parquet，元数据为pkl。
        仅保存轻量元数据，避免表达式/处理器等不可序列化对象。
        """

        # 保存元数据（分段与处理类型）
        periods: dict[str, tuple[str, str]] = {
            seg.name: period for seg, period in self.data_periods.items()
        }
        meta = {
            "schema_version": 1,
            "process_type": self.process_type,
            "periods": periods
        }
        with open(path.joinpath("meta.pkl"), "wb") as f:
            pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)

        # 逐个保存DataFrame为parquet（存在才保存）
        for attr in ["df", "result_df",  "infer_df", "learn_df"]:
            df_obj = getattr(self, attr, None)
            if isinstance(df_obj, pl.DataFrame):
                df_obj.write_parquet(path.joinpath(f"{attr}.parquet"))

    @classmethod
    def load(cls, path: Path) -> "AlphaDataset":
        """
        从目录加载数据集：读取元数据与各DataFrame的parquet。
        """
        meta_file = path.joinpath("meta.pkl")
        if not meta_file.exists():
            raise FileNotFoundError(f"Dataset meta file not found: {meta_file}")

        with open(meta_file, "rb") as f:
            meta = pickle.load(f)

        periods: dict[str, tuple[str, str]] = meta.get("periods", {})
        train = periods.get("TRAIN", ("", ""))
        valid = periods.get("VALID", ("", ""))
        test = periods.get("TEST", ("", ""))
        process_type = meta.get("process_type", "append")

        # df 为必需
        df_file = path.joinpath("df.parquet")
        if not df_file.exists():
            raise FileNotFoundError(f"Dataset main df file not found: {df_file}")
        df = pl.read_parquet(df_file)

        dataset = cls(df, train, valid, test, process_type=process_type)

        for attr in ["result_df","infer_df", "learn_df"]:
            file = path.joinpath(f"{attr}.parquet")
            if file.exists():
                setattr(dataset, attr, pl.read_parquet(file))

        return dataset


def query_by_time(
    df: pl.DataFrame, start: datetime | str = "", end: datetime | str = ""
) -> pl.DataFrame:
    """
    Filter DataFrame based on time range
    """
    if start:
        start = to_datetime(start)
        df = df.filter(pl.col("datetime") >= start)

    if end:
        end = to_datetime(end)
        df = df.filter(pl.col("datetime") <= end)

    return df.sort(["datetime", "vt_symbol"])


def calculate_feature(
    args: tuple[str, FeatProxy],
) -> pl.DataFrame:
    """
    计算单个特征：收集 LazyFrame 为 DataFrame，重命名为特征名，并保留键列。
    返回形如 [datetime, vt_symbol, <name>] 的 DataFrame。
    """
    start = time.time()

    name, feature = args
    df: pl.DataFrame = feature.df.collect(engine="gpu")
    df = df.rename({"data": name})

    end = time.time()
    print(f"Feature calculation {name} took: {end - start} seconds")

    return df.select(["datetime", "vt_symbol", name])
