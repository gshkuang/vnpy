#!/usr/bin/env python
# coding: utf-8

# 整洁化后的投研工作流（MLP）

import os
from datetime import datetime
from functools import partial

import polars as pl

from vnpy.alpha import AlphaLab, Segment, AlphaDataset, AlphaModel, logger, to_datetime
from vnpy.trader.constant import Interval
from vnpy.alpha.dataset import (
    process_drop_na,
    process_robust_zscore_norm,
    process_fill_na,
    process_cs_rank_norm,
)
from vnpy.alpha.dataset.datasets.alpha_158 import Alpha158


# ========== 全局配置 ==========
NAME = "300_mlp_crypto_1m"
INDEX_SYMBOL: str = "CRYPTO_INDEX_1M"
START: str = "2020-01-01"
END: str = "2025-09-03"
INTERVAL: Interval = Interval.MINUTE
EXTENDED_DAYS: int = 100
LAB_DIR = "./lab/crypto_1m"
TRAIN_PERIOD: tuple[str, str] = ("2020-01-01", "2023-12-31")
VALID_PERIOD: tuple[str, str] = ("2024-01-01", "2024-12-31")
TEST_PERIOD: tuple[str, str] = ("2025-01-01", "2025-09-03")


def build_dataset(lab: AlphaLab) -> AlphaDataset:
    """加载数据并构建数据集，包含特征与预处理器。"""
    # 加载成分股代码
    component_symbols: list[str] = lab.load_component_symbols(INDEX_SYMBOL, START, END)
    logger.info(f"成分股数量: {len(component_symbols)}")
    # 为演示与开发限定数量（可按需调整）
    component_symbols = component_symbols[:10]

    # 加载行情数据
    df: pl.DataFrame = lab.load_bar_df(component_symbols, INTERVAL, START, END, EXTENDED_DAYS)
    df = df.unique(subset=["datetime", "vt_symbol"], keep="first").sort(["datetime", "vt_symbol"])
    logger.info(f"原始数据形状: {df.shape}")

    # 创建数据集对象
    dataset: AlphaDataset = Alpha158(
        df,
        train_period=TRAIN_PERIOD,
        valid_period=VALID_PERIOD,
        test_period=TEST_PERIOD,
        period=INTERVAL.value.lower()[-1:],
    )

    # 添加数据预处理器
    fit_start_time: datetime = to_datetime(TRAIN_PERIOD[0])
    fit_end_time: datetime = to_datetime(TRAIN_PERIOD[1])
    dataset.add_processor(partial(process_robust_zscore_norm, fit_start_time=fit_start_time, fit_end_time=fit_end_time))
    dataset.add_processor(partial(process_fill_na, fill_value=0, fill_label=False))
    dataset.add_processor(partial(process_drop_na, names=["label"]))
    dataset.add_processor(partial(process_cs_rank_norm, names=["label"]))

    return dataset


def train_model(lab: AlphaLab, dataset: AlphaDataset) -> AlphaModel:
    """训练 MLP 模型并保存。"""
    from vnpy.alpha.model.models.mlp_model import MlpModel
    import numpy as np

    kwargs = {
        "input_size": 158,
        "hidden_sizes": (256,),
        "lr": 0.002,
        "optimizer": "adam",
        "n_epochs": 8000,
        "batch_size": 8192,
        "weight_decay": 0.0002,
        "seed": 42,
        "device": "cuda",
    }

    model: AlphaModel = MlpModel(**kwargs)

    # 观察训练集形状
    df_train = dataset.fetch_feat(Segment.TRAIN)
    logger.info(f"TRAIN shape: {df_train.shape}")

    # 训练
    model.fit(dataset)
    model.detail()

    # 保存模型
    lab.save_model(NAME, model)
    return model


def predict_and_save_signal(lab: AlphaLab, model: AlphaModel, dataset: AlphaDataset) -> pl.DataFrame:
    """在测试集上预测并保存信号。"""
    import numpy as np

    pre: np.ndarray = model.predict(dataset, Segment.TEST)
    df_t: pl.DataFrame = dataset.fetch_feat(Segment.TEST)
    df_t = df_t.with_columns(pl.Series(pre).alias("signal"))
    signal: pl.DataFrame = df_t["datetime", "vt_symbol", "signal"]

    # 绩效检查与保存
    #dataset.show_signal_performance(signal)
    lab.save_signal(NAME, signal)
    return signal


def run_backtesting(lab: AlphaLab, signal: pl.DataFrame, vt_symbols: list[str]) -> None:
    """执行策略回测并展示图表。"""
    import importlib
    from vnpy.alpha.strategy import BacktestingEngine
    import vnpy.alpha.strategy.strategies.equity_demo_strategy as equity_demo_strategy

    importlib.reload(equity_demo_strategy)
    EquityDemoStrategy = equity_demo_strategy.EquityDemoStrategy

    engine = BacktestingEngine(lab)
    engine.set_parameters(
        vt_symbols=vt_symbols,
        interval=Interval.DAILY,
        start=datetime(2022, 1, 1),
        end=datetime(2024, 10, 31),
        capital=100000000,
    )
    setting = {"top_k": 30, "n_drop": 3, "hold_thresh": 3}
    engine.add_strategy(EquityDemoStrategy, setting, signal)

    engine.load_data()
    engine.run_backtesting()
    engine.calculate_result()
    engine.calculate_statistics()
    engine.show_chart()


def main() -> None:

    lab: AlphaLab = AlphaLab(LAB_DIR)

    # 构建并准备数据集
    dataset = build_dataset(lab)
    #dataset.prepare_features(max_workers=2)
    dataset.process_features()
    # 缓存数据集到文件
    lab.save_dataset(NAME, dataset)
    # # 训练模型
    # model = train_model(lab, dataset)
    # # 预测与保存信号
    # signal = predict_and_save_signal(lab, model, dataset)
    # # 加载成分股代码用于回测
    # vt_symbols: list[str] = lab.load_component_symbols(INDEX_SYMBOL, START, END)[:10]
    # run_backtesting(lab, signal, vt_symbols)


if __name__ == "__main__":
    main()