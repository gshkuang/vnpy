import time
from datetime import datetime
from typing import cast
from collections.abc import Callable
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
import os
import psutil
import polars as pl
import pandas as pd
from tqdm import tqdm
from alphalens.utils import get_clean_factor_and_forward_returns  # type: ignore
from alphalens.tears import create_full_tear_sheet  # type: ignore
import pickle
from pathlib import Path
from ..logger import logger,log_time_memory
from .utility import to_datetime, Segment, FeatProxy
from .processor import process_lf_drop_na
from .feature_pipeline import feat_norm_pipeline,save_duckdb_splits
class AlphaDataset:
    """Alpha dataset template class"""

    def __init__(
        self,
        df: pl.DataFrame,
        train_period: tuple[str, str],
        valid_period: tuple[str, str],
        test_period: tuple[str, str],
        lab_dir: str,
    ) -> None:
        """Constructor"""
        self.df: pl.DataFrame =df.unique(subset=["datetime", "vt_symbol"], keep="first").sort(["datetime", "vt_symbol"])
        self.lab_dir: Path = Path(lab_dir)

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

        # 处理器统一（用于 lazy infer/learn）
        self.learn_processors: list[Callable[[pl.LazyFrame], pl.LazyFrame]] = []

        # collect 开关：默认流式
        self.collect_engine: str = "streaming"

    def set_collect_engine(self, engine: str) -> None:
        """Set collect engine, e.g., 'streaming', 'gpu'"""
        self.collect_engine = engine

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
        return ["datetime", "vt_symbol"]+ feature_cols + label_cols

    def add_processor(
        self, processor: Callable[[pl.LazyFrame], pl.LazyFrame]
    ) -> None:
        """Register a lazy processor: LazyFrame -> LazyFrame"""
        self.learn_processors.append(processor)

    def prepare_features(
        self,  batch_size: int = 100, symbol: str | None = None
    ) -> None:
        """
        Generate required data once, then cache per segment to avoid duplication.
        - 计算特征与标签得到 result_df
        - 按元属性选择列生成 raw_df（键列 + 特征 + 标签）
        - 处理器链执行一次，得到 processed_df
        - processed_df 按周期切片，统一放入 segment_data（相同引用，避免重复）
        
        内存优化策略：
        - 流式合并：逐批合并特征，避免累积所有LazyFrame
        - 内存监控：动态调整批次大小，防止内存溢出
        - 及时清理：每步完成后立即清理中间对象
        """

        logger.info("开始计算 因子特征")
        t_feat_start = time.time()
    
        # 仅在 label_feature 存在时加入任务，避免 None 导致错误
        tasks: list[tuple[str, FeatProxy]] = list(self.features.items())
        if self.label_feature is not None:
            tasks.append(self.label_feature)
    
        # 初始化结果为原始数据的LazyFrame
        result_df: pl.DataFrame = self.df
        
        if tasks:
            @log_time_memory
            def _iter_batches(seq: list[tuple[str, FeatProxy]], size: int):
                for i in range(0, len(seq), size):
                    yield seq[i:i+size]
            for batch in _iter_batches(tasks, batch_size):
                series_list = [calculate_feature((name, feature),self.collect_engine) for name, feature in batch] 
                result_df = result_df.hstack(series_list)

        t_feat_end = time.time()

        # 将计算好的 result_df 持久化到磁盘，释放内存
        result_lf =result_df.lazy().with_columns(pl.col(self.features.keys()).fill_nan(float("nan")))
        result_lf = result_lf.select(self.select_columns).sort(["datetime", "vt_symbol"])
        result_lf=process_lf_drop_na(result_lf)
        
        feat_dir =self.lab_dir.joinpath("feat")
        feat_dir.mkdir(parents=True, exist_ok=True)
        features_file =  feat_dir.joinpath( symbol+".parquet")
        result_lf.sink_parquet(features_file)
        logger.info(f"原始特征已写入磁盘: {features_file} | 行数: {result_df.height}")

        logger.info(f"持久化特征结果 | 数量: {len(tasks)} | 耗时: {t_feat_end - t_feat_start:.3f}s")

    def process_features(self):
        feat_norm_pipeline(
            feat_dir=self.lab_dir.joinpath("feat"),
            stats_dir=self.lab_dir.joinpath("stats"),
            out_dir=self.lab_dir.joinpath("feat_norm"),
            fit_start=to_datetime(self.data_periods[Segment.TRAIN][0]),
            fit_end=to_datetime(self.data_periods[Segment.TRAIN][1]),
        )

        save_duckdb_splits(
            feat_dir=self.lab_dir.joinpath("feat_norm"),
            out_dir=self.lab_dir.joinpath("splits"),
            train_period=self.data_periods[Segment.TRAIN],
            valid_period=self.data_periods[Segment.VALID],
            test_period=self.data_periods[Segment.TEST],
            shuffle=True,
            seed=42,
        )
        
    def _preprocess_feat(self,result_lf:pl.LazyFrame):
                # 处理器链仅构建 lazy pipeline，最终统一 collect
        logger.info("开始构建处理器 lazy pipeline")
        for i, processor in enumerate(self.learn_processors, start=1):
            result_lf = processor(lf=result_lf)
        # 统一 collect（默认流式）
        logger.info(f"开始最终 collect，engine={self.collect_engine}")
        result_df = result_lf.collect(engine=self.collect_engine)

        # 按周期分段缓存（合并为 segment_data）
        for seg, (start, end) in self.data_periods.items():
            seg_df = query_by_time(result_df, start, end)
            self.segment_data[seg] = seg_df

        logger.info("特征准备完成")


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
        ).sort(["datetime", "vt_symbol"])
        self.show_signal_performance(signal)


    def show_signal_performance(self, signal: pl.DataFrame,quantiles=2) -> None:
        """
        Perform performance analysis for prediction signals
        """
        # Get signal start and end times
        start: datetime = cast(datetime, signal["datetime"].min())
        end: datetime = cast(datetime, signal["datetime"].max())
        logger.info(f"signal period: {start} - {end}")
        # Select range
        df: pl.DataFrame = query_by_time(self.df, start, end)

        # Extract feature
        signal_df: pd.DataFrame = signal.to_pandas()
        signal_df.set_index(["datetime", "vt_symbol"], inplace=True)
        signal_s: pd.Series = signal_df["signal"]
        freq: str = pd.infer_freq(signal_df.index.levels[0])
        logger.info(f"infer freq: {freq}")
        signal_df.index.levels[0].freq = freq

        # Extract price
        price_df: pd.DataFrame = df.select(
            ["datetime", "vt_symbol", "close"]
        ).to_pandas()
        price_df = price_df.pivot(index="datetime", columns="vt_symbol", values="close")
        # Merge data
        clean_data: pd.DataFrame = get_clean_factor_and_forward_returns(
            signal_s, price_df, max_loss=1.0, quantiles=quantiles
        )

        # Perform analysis
        create_full_tear_sheet(clean_data)

    def save(self, path: Path ) -> None:
        """
        保存到目录：保存元数据 pkl、原始数据 raw_df.parquet、各分段 processed parquet。
        """
        # 确保目录存在
        path.mkdir(parents=True, exist_ok=True)

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

        # 保存原始数据（未处理的 self.df）
        if isinstance(self.df, pl.DataFrame) and self.df.height > 0:
            self.df.write_parquet(path.joinpath("raw_df.parquet"))

        # 仅保存按周期的处理后数据，三个 parquet
        for seg in [Segment.TRAIN, Segment.VALID, Segment.TEST]:
            df_obj = self.segment_data.get(seg)
            if isinstance(df_obj, pl.DataFrame):
                df_obj.write_parquet(path.joinpath(f"segment_{seg.name.lower()}.parquet"))

    @classmethod
    def load(cls, path: Path,name:str) -> "AlphaDataset":
        """
        从目录加载数据集：读取元数据、原始数据 raw_df.parquet（若存在）与各分段 DataFrame。
        """
        datapath = path.joinpath("dataset",name)
        meta_file = datapath.joinpath("meta.pkl")
        if not meta_file.exists():
            raise FileNotFoundError(f"Dataset meta file not found: {meta_file}")

        with open(meta_file, "rb") as f:
            meta = pickle.load(f)

        periods: dict[str, tuple[str, str]] = meta.get("periods", {})
        train = periods.get("TRAIN", ("", ""))
        valid = periods.get("VALID", ("", ""))
        test = periods.get("TEST", ("", ""))

        # 原始数据文件路径
        raw_file = datapath.joinpath("raw_df.parquet")

        # 读入各段 parquet
        segment_map: dict[Segment, pl.DataFrame] = {}
        for seg in [Segment.TRAIN, Segment.VALID, Segment.TEST]:
            file = datapath.joinpath(f"segment_{seg.name.lower()}.parquet")
            if file.exists():
                segment_map[seg] = pl.read_parquet(file)
            else:
                segment_map[seg] = pl.DataFrame()

        # 构造 df：优先读取原始数据；否则用三段合并（用于保持构造签名与基本功能）
        if raw_file.exists():
            df = pl.read_parquet(raw_file)
        else:
            non_empty = [segment_map[s] for s in [Segment.TRAIN, Segment.VALID, Segment.TEST] if segment_map[s].height > 0]
            df = pl.concat(non_empty, how="diagonal_relaxed") if non_empty else pl.DataFrame()

        dataset = cls(df, train, valid, test,path)

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
    args: tuple[str, FeatProxy],engine="gpu",
) -> pl.Series:
    """
    计算单个特征：收集 LazyFrame 为 DataFrame，重命名为特征名，并保留键列。
    返回形如 [datetime, vt_symbol, <name>] 的 DataFrame。
    """
    start = time.time()

    name, feature = args
    result = feature.df.collect(engine=engine).sort(["datetime", "vt_symbol"])["data"].alias(name)

    end = time.time()
    #print(f"Feature calculation {name} took: {end - start} seconds")

    return result
