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
    ) -> None:
        """Constructor"""
        self.df: pl.DataFrame = df

        # New version
        self.data_periods: dict[Segment, tuple[str, str]] = {
            Segment.TRAIN: train_period,
            Segment.VALID: valid_period,
            Segment.TEST: test_period,
        }

        # 特征集合
        self.features: dict[str, FeatProxy] = {}
        self.label_feature: tuple[str, FeatProxy] | None = None

        # 分段缓存（统一）
        self.segment_data: dict[Segment, pl.DataFrame] = {}

        # 处理器统一（用于 infer/learn）
        self.learn_processors: list = []

    def add_feature(
        self,
        name: str,
        feature: FeatProxy
    ) -> None:
        """Add a feature and record meta attributes for unified management"""
        self.features[name] = feature
    

    def set_label(self,  feature: FeatProxy) -> None:
        """Set label feature as (name, feature) tuple"""
        self.label_feature = ("label", feature)
        
    @property
    def select_columns(self) -> list[str]:
        feature_cols: list[str] = list(self.features.keys())
        label_cols: list[str] = [self.label_feature[0]] if self.label_feature else []
        return ["datetime", "vt_symbol","close"]+ feature_cols + label_cols

    def add_processor(
        self, processor: Callable[[pl.DataFrame], None]
    ) -> None:
        self.learn_processors.append(processor)

    def prepare_data(
        self,  max_workers: int=4
    ) -> None:
        """
        Generate required data once, then cache per segment to avoid duplication.
        - 计算特征与标签得到 result_df
        - 按元属性选择列生成 raw_df（键列 + 特征 + 标签）
        - 处理器链执行一次，得到 processed_df
        - processed_df 按周期切片，统一放入 segment_data（相同引用，避免重复）
        """
        feats: list[pl.Series] = []
    
        logger.info("开始计算 因子特征")
        t_feat_start = time.time()
    
        # 仅在 label_feature 存在时加入任务，避免 None 导致错误
        tasks: list[tuple[str, FeatProxy]] = list(self.features.items())
        if self.label_feature is not None:
            tasks.append(self.label_feature)
    
        if tasks:
            args: list[tuple[str, FeatProxy]] = [(name, feature) for name, feature in tasks]
            context: BaseContext = get_context("spawn")
            with context.Pool(processes=max_workers) as pool:
                it = pool.imap(calculate_feature, args)
                for result in tqdm(it, total=len(args)):
                    feats.append(result)
    
        t_feat_end = time.time()
        result_df = self.df.with_columns(feats).fill_null(float("nan")).select(self.select_columns).sort(["datetime", "vt_symbol"])
        
        logger.info(f"因子计算完成 | 数量: {len(tasks)} | 耗时: {t_feat_end - t_feat_start:.3f}s")

        # 处理器链只执行一次，得到 processed_df
        logger.info("开始执行处理器链")
        for i, processor in enumerate(self.learn_processors, start=1):
            t_proc_start = time.time()
            result_df = processor(df=result_df)
            t_proc_end = time.time()
            proc_name = getattr(processor, "__name__", processor.__class__.__name__)
            logger.info(f"处理器[{i}] {proc_name} 完成 | 耗时: {t_proc_end - t_proc_start:.3f}s")

    
        # 按周期分段缓存（合并为 segment_data）
        logger.info("按周期切片缓存处理结果")
        for seg, (start, end) in self.data_periods.items():
            seg_df = query_by_time(result_df, start, end)
            self.segment_data[seg] = seg_df
    
        logger.info("数据准备完成：处理一次、按周期缓存、去重存储")


    def fetch_feat(self, segment: Segment) -> pl.DataFrame:
        return self.segment_data[segment]

    def show_feature_performance(self, name: str) -> None:
        combined_df: pl.DataFrame = pl.concat([
            self.segment_data[Segment.TRAIN],
            self.segment_data[Segment.VALID],
            self.segment_data[Segment.TEST],
        ])
        signal: pl.DataFrame = combined_df.select(
            ["datetime", "vt_symbol", pl.col(name).alias("signal")]
        )
        self.show_signal_performance(signal)


    def show_signal_performance(self, signal: pl.DataFrame) -> None:
        """
        Perform performance analysis for prediction signals
        """
        # Get signal start and end times
        start: datetime = cast(datetime, signal["datetime"].min())
        end: datetime = cast(datetime, signal["datetime"].max())

        # Select range
        df: pl.DataFrame = query_by_time(self.df, start, end)

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
        保存到目录：仅保存分段数据为三个 parquet，元数据为 pkl。
        """
        # 保存元数据（分段与 schema）
        periods: dict[str, tuple[str, str]] = {
            seg.name: period for seg, period in self.data_periods.items()
        }
        meta = {
            "schema_version": 2,
            "periods": periods
        }
        with open(path.joinpath("meta.pkl"), "wb") as f:
            pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
    
        # 仅保存按周期的处理后数据，三个 parquet
        for seg in [Segment.TRAIN, Segment.VALID, Segment.TEST]:
            df_obj = self.segment_data.get(seg)
            if isinstance(df_obj, pl.DataFrame):
                df_obj.write_parquet(path.joinpath(f"segment_{seg.name.lower()}.parquet"))

    @classmethod
    def load(cls, path: Path) -> "AlphaDataset":
        """
        从目录加载数据集：读取元数据与每段 DataFrame（三个 parquet）。
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
    
        # 读入各段 parquet
        segment_map: dict[Segment, pl.DataFrame] = {}
        for seg in [Segment.TRAIN, Segment.VALID, Segment.TEST]:
            file = path.joinpath(f"segment_{seg.name.lower()}.parquet")
            if file.exists():
                segment_map[seg] = pl.read_parquet(file)
            else:
                segment_map[seg] = pl.DataFrame()
    
        # 构造 df：用三段合并（用于保持构造签名与基本功能）
        non_empty = [segment_map[s] for s in [Segment.TRAIN, Segment.VALID, Segment.TEST] if segment_map[s].height > 0]
        df = pl.concat(non_empty, how="diagonal_relaxed") if non_empty else pl.DataFrame()
    
        dataset = cls(df, train, valid, test)
    
        # 设置统一的分段缓存
        dataset.segment_data = segment_map
    
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

    return df


def calculate_feature(
    args: tuple[str, FeatProxy],
) -> pl.Series:
    """
    计算单个特征：收集 LazyFrame 为 DataFrame，重命名为特征名，并保留键列。
    返回形如 [datetime, vt_symbol, <name>] 的 DataFrame。
    """
    start = time.time()

    name, feature = args
    result = feature.df.collect(engine="gpu")["data"].alias(name)

    end = time.time()
    print(f"Feature calculation {name} took: {end - start} seconds")

    return result
