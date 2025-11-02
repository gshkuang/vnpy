#!/usr/bin/env python
# coding: utf-8

# 整洁化后的投研工作流（MLP）

import os
from datetime import datetime
from functools import partial

import polars as pl

from vnpy.alpha import AlphaLab, Segment, AlphaDataset, AlphaModel, logger, to_datetime
from vnpy.alpha.dataset.feature_pipeline import save_duckdb_splits
from pathlib import Path
from vnpy.trader.constant import Interval
# 新流程使用 AlphaDataset 的归一化管道，无需逐个添加处理器
from vnpy.alpha.dataset.datasets.alpha_158 import Alpha158


# ========== 全局配置 ==========
NAME = "300_mlp_crypto_1m"
INDEX_SYMBOL: str = "CRYPTO_INDEX_1M"
START: str = "2020-01-01"
END: str = "2025-09-03"
INTERVAL: Interval = Interval.MINUTE
EXTENDED_DAYS: int = 100
LAB_DIR = "/home/lai/test_v1/lab/crypto_1m"
TRAIN_PERIOD: tuple[str, str] = ("2020-01-01", "2023-12-31")
VALID_PERIOD: tuple[str, str] = ("2024-01-01", "2024-12-31")
TEST_PERIOD: tuple[str, str] = ("2025-01-01", "2025-09-03")

def train_model(lab: AlphaLab, splits_dir: str) -> AlphaModel:
    """使用保存的 Parquet 切分直接训练 MLP 模型并保存。"""
    from vnpy.alpha.model.models.mlp_model import MlpModel

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
    model.fit_splits(splits_dir)
    model.detail()

    lab.save_model(NAME, model)
    return model


def predict_and_save_signal(lab: AlphaLab, model: AlphaModel, test_parquet: str) -> pl.DataFrame:
    """在测试切分上预测并保存信号。切分文件不含键列，返回仅包含预测值。"""
    import pandas as pd
    pre = model.predict_splits(test_parquet)
    df = pd.read_parquet(test_parquet)
    signal = pl.DataFrame({"signal": pre})
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
    """加载数据并构建数据集，包含特征与预处理器。"""
    # 加载成分股代码
    component_symbols: list[str] = lab.load_component_symbols(INDEX_SYMBOL, START, END)
    logger.info(f"成分股数量: {len(component_symbols)}")
    # 为演示与开发限定数量（可按需调整）
    component_symbols = component_symbols[:300]
    for i,symbol in enumerate(component_symbols):
        df: pl.DataFrame|None=lab.load_bar_df([symbol], INTERVAL, START, END, EXTENDED_DAYS)
        if df is None:
            continue
        
        logger.info(f"{i+1}/{len(component_symbols)} {symbol}原始数据形状: {df.shape}")

        # 创建数据集对象
        dataset: AlphaDataset = Alpha158(
            df,
            train_period=TRAIN_PERIOD,
            valid_period=VALID_PERIOD,
            test_period=TEST_PERIOD,
            period=INTERVAL.value.lower()[-1:],
            lab_dir=LAB_DIR,
        )

        #写入原始特征到磁盘
        dataset.prepare_features(symbol=symbol)
    # df: pl.DataFrame|None=lab.load_bar_df(component_symbols, INTERVAL, START, END, EXTENDED_DAYS)
    # dataset: AlphaDataset = Alpha158(
    #     df,
    #     train_period=TRAIN_PERIOD,
    #     valid_period=VALID_PERIOD,
    #     test_period=TEST_PERIOD,
    #     period=INTERVAL.value.lower()[-1:],
    #     lab_dir=LAB_DIR,
    # )
    # dataset.process_features()


    # 训练与预测
    #model = train_model(lab, str(LAB_DIR+ "/splits"))
    # signal = predict_and_save_signal(lab, model, LAB_DIR+ "/splits/test.parquet")
    # print(signal.head())
    # # 加载成分股代码用于回测
    # vt_symbols: list[str] = lab.load_component_symbols(INDEX_SYMBOL, START, END)[:10]
    # run_backtesting(lab, signal, vt_symbols)


if __name__ == "__main__":
    main()