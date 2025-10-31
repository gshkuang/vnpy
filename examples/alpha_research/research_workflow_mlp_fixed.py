#!/usr/bin/env python
# coding: utf-8

# 过滤Alphalens的warning
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# 加载模块
import polars as pl
import multiprocessing
from datetime import datetime
from functools import partial

from vnpy.trader.constant import Interval
from vnpy.alpha import AlphaLab
from vnpy.alpha.dataset import (
    AlphaDataset,
    process_drop_na,
    process_robust_zscore_norm,
    process_fill_na,
    process_cs_rank_norm,
    to_datetime
)
from vnpy.alpha.dataset.datasets.alpha_158 import Alpha158


# 参数配置常量（仅定义而不执行）
name = "300_mlp_crypto_1m"
index_symbol: str = "CRYPTO_INDEX_1M"
start: str = "2020-01-01"
end: str = "2025-09-03"
interval: Interval = Interval.MINUTE
extended_days: int = 100

train_period: tuple[str, str] = ("2020-01-01", "2023-12-31")
valid_period: tuple[str, str] = ("2024-01-01", "2024-12-31")
test_period: tuple[str, str] = ("2025-01-01", "2025-09-03")

def main():
    # 创建数据中心与加载符号（移入 main，避免导入期执行）
    lab: AlphaLab = AlphaLab("./lab/crypto_1m")

    component_symbols: list[str] = lab.load_component_symbols(index_symbol, start, end)[:10]
    component_symbols = [s for s in component_symbols if not s.startswith("FUSDT")]
    print(component_symbols)

    # 加载成分股数据
    df: pl.DataFrame = lab.load_bar_df(component_symbols, interval, start, end, extended_days)
    df = df.unique(subset=["datetime", "vt_symbol"], keep="first")
    print(df.head())

    # 创建数据集对象与预处理器
    dataset: AlphaDataset = Alpha158(
        df,
        train_period=train_period,
        valid_period=valid_period,
        test_period=test_period,
    )

    fit_start_time: datetime = to_datetime(train_period[0])
    fit_end_time: datetime = to_datetime(train_period[1])
    print(f"fit_start_time: {fit_start_time}")
    print(f"fit_end_time: {fit_end_time}")

    dataset.add_processor("infer", partial(process_robust_zscore_norm, fit_start_time=fit_start_time, fit_end_time=fit_end_time))
    dataset.add_processor("infer", partial(process_fill_na, fill_value=0, fill_label=False))
    dataset.add_processor("learn", partial(process_drop_na, names=["label"]))
    dataset.add_processor("learn", partial(process_cs_rank_norm, names=["label"]))

    # 收集指数成分过滤器并准备数据（多进程）
    #filters: dict[str, list[str]] = lab.load_component_filters(index_symbol, start, end)
    dataset.prepare_data( max_workers=1)
    lab.save_dataset(name, dataset)

    # 模型训练
    from vnpy.alpha import Segment, AlphaModel
    from vnpy.alpha.model.models.mlp_model import MlpModel
    import numpy as np

    dataset_loaded: AlphaDataset = lab.load_dataset(name)

    # 检查CUDA可用性并设置device
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    kwargs = {
        "input_size": 158,
        "hidden_sizes": (256,),
        "lr": 0.002,
        "optimizer": "adam",
        "n_epochs": 8000,
        "batch_size": 8192,
        "weight_decay": 0.0002,
        "device": device,  # 添加device配置
        "seed": 42
    }
    model: AlphaModel = MlpModel(**kwargs)
    print("model kwargs:")
    print(kwargs)
    print(f"Model device: {model.device}")
    
    # 如果使用CUDA，显示GPU内存使用情况
    if device == "cuda":
        print(f"GPU Memory before training: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
    
    model.fit(dataset_loaded)
    model.detail()
    lab.save_model(name, model)

    # 预测信号
    model_loaded: AlphaModel = lab.load_model(name)
    pre: np.ndarray = model_loaded.predict(dataset_loaded, Segment.TEST)
    df_t: pl.DataFrame = dataset_loaded.fetch_feat(Segment.TEST)
    df_t = df_t.with_columns(pl.Series(pre).alias("signal"))
    signal: pl.DataFrame = df_t["datetime", "vt_symbol", "signal"]
    dataset_loaded.show_signal_performance(signal)
    lab.save_signal(name, signal)

    # 策略回测
    from vnpy.alpha.strategy import BacktestingEngine
    import importlib
    import vnpy.alpha.strategy.strategies.equity_demo_strategy as equity_demo_strategy

    importlib.reload(equity_demo_strategy)
    EquityDemoStrategy = equity_demo_strategy.EquityDemoStrategy

    engine = BacktestingEngine(lab)
    engine.set_parameters(
        vt_symbols=component_symbols,
        interval=Interval.MINUTE,
        start=datetime(2025, 1, 1),
        end=datetime(2025, 10, 31),
        capital=100000000
    )

    setting = {"top_k": 3, "n_drop": 2, "hold_thresh": 3}
    engine.add_strategy(EquityDemoStrategy, setting, signal)

    engine.load_data()
    engine.run_backtesting()
    engine.calculate_result()
    engine.calculate_statistics()
    engine.show_chart()
    engine.show_performance(benchmark_symbol=index_symbol)


if __name__ == "__main__":
    # 兼容 spawn 环境，防止导入期触发多进程错误
    multiprocessing.freeze_support()
    multiprocessing.set_start_method('fork', force=True)  # 可选，强制使用 fork
    main()

