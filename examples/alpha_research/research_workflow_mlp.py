#!/usr/bin/env python
# coding: utf-8

"""
投研工作流（MLP） - 配置驱动 + 子进程编排

通过 workflow.yaml 驱动各阶段（load_data, prepare_features, process_features, train_model,
predict_signal, backtesting），各阶段尽可能在子进程执行，降低内存相互影响。
"""
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List

import polars as pl
import yaml

import vnpy.alpha.strategy.strategies.equity_demo_strategy as equity_demo_strategy
from vnpy.alpha.config.strategy_config import StrategyConfig
from vnpy.trader.optimize import OptimizationSetting
from vnpy.alpha import AlphaDataset, AlphaLab, AlphaModel, logger
from vnpy.alpha.config.dataset import DATASET_CONFIG
from vnpy.alpha.dataset.datasets.alpha_158 import Alpha158
from vnpy.alpha.model.models.mlp_model import MlpModel
from vnpy.alpha.strategy import BacktestingEngine
from vnpy.trader.constant import Interval

# -----------------------------
# Step 注册机制
# -----------------------------
STEP_REGISTRY: Dict[str, Callable] = {}


def register_step(name: str):
    """装饰器：注册一个 step 到 registry"""

    def decorator(func):
        STEP_REGISTRY[name] = func
        return func

    return decorator


@dataclass
class WorkflowContext:
    name: str
    lab_dir: str
    index_symbol: str
    start: str
    end: str
    interval: str
    extended_days: int
    train_period: tuple[str, str]
    valid_period: tuple[str, str]
    test_period: tuple[str, str]
    lab: AlphaLab
    component_symbols: List[str]

    @property
    def interval_enum(self) -> Interval:
        mapping = {
            "minute": Interval.MINUTE,
            "daily": Interval.DAILY,
            "hourly": Interval.HOUR,
        }
        return mapping.get(self.interval.lower(), Interval.MINUTE)


def load_yaml(path: str | os.PathLike) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_context(cfg: Dict[str, Any]) -> WorkflowContext:
    wf = cfg["workflow"]
    general = wf.get("general", {})

    # 初始化 Lab 与成分股列表（全局）
    lab_dir = general.get("lab_dir")
    lab: AlphaLab = AlphaLab(lab_dir)
    index_symbol = general.get("index_symbol")
    start = general.get("start")
    end = general.get("end")

    component_symbols = lab.load_component_symbols(index_symbol, start, end)

    return WorkflowContext(
        name=general.get("name"),
        lab_dir=lab_dir,
        index_symbol=index_symbol,
        start=start,
        end=end,
        interval=general.get("interval"),
        extended_days=int(general.get("extended_days")),
        train_period=tuple(general.get("train_period")),
        valid_period=tuple(general.get("valid_period")),
        test_period=tuple(general.get("test_period")),
        lab=lab,
        component_symbols=component_symbols,
    )


@register_step("load_data")
def step_load_data(ctx: WorkflowContext, step_cfg: Dict[str, Any]) -> None:
    lab: AlphaLab = ctx.lab
    component_symbols = ctx.component_symbols
    logger.info(f"成分股数量: {len(component_symbols)}")

    for i, symbol in enumerate(component_symbols):
        df: pl.DataFrame | None = lab.load_bar_df(
            [symbol], ctx.interval_enum, ctx.start, ctx.end, ctx.extended_days
        )
        if df is None or df.height == 0:
            logger.info(f"{symbol} 缺少数据，跳过")
            continue
        logger.info(f"{i+1}/{len(component_symbols)} {symbol} 原始数据形状: {df.shape}")
        dataset: AlphaDataset = Alpha158(
            df,
            train_period=ctx.train_period,
            valid_period=ctx.valid_period,
            test_period=ctx.test_period,
            period=ctx.interval_enum,
            lab_dir=ctx.lab_dir,
        )
        dataset.prepare_features(symbol=symbol)


@register_step("process_features")
def step_process_features(ctx: WorkflowContext, step_cfg: Dict[str, Any]) -> None:
    lab: AlphaLab = ctx.lab
    # 选择一个标的以构造数据集（仅用于提供 schema 与 period）
    symbols = ctx.component_symbols
    if not symbols:
        raise RuntimeError("无成分数据，无法进行特征处理")
    df: pl.DataFrame | None = lab.load_bar_df(
        [symbols[0]], ctx.interval_enum, ctx.start, ctx.end, ctx.extended_days
    )
    if df is None or df.height == 0:
        raise RuntimeError("无法加载用于构造数据集的示例数据")

    dataset: AlphaDataset = Alpha158(
        df,
        train_period=ctx.train_period,
        valid_period=ctx.valid_period,
        test_period=ctx.test_period,
        period=ctx.interval_enum,
        lab_dir=ctx.lab_dir,
    )
    dataset.process_features()


@register_step("train_model")
def step_train_model(ctx: WorkflowContext, step_cfg: Dict[str, Any]) -> None:
    lab: AlphaLab = ctx.lab
    splits_dir = str(
        Path(ctx.lab_dir) / DATASET_CONFIG.get("paths", {}).get("splits_dir", "splits")
    )

    kwargs = step_cfg.get("config", {})
    model: AlphaModel = MlpModel(**kwargs)
    model.fit(splits_dir)
    model.detail()
    lab.save_model(ctx.name, model)


@register_step("predict_signal")
def step_predict_signal(ctx: WorkflowContext, step_cfg: Dict[str, Any]) -> None:
    lab: AlphaLab = ctx.lab

    # 加载模型
    model = lab.load_model(ctx.name)
    if model is None:
        raise FileNotFoundError(f"模型 {ctx.name} 不存在，请先执行 train_model")

    test_parquet = str(
        Path(ctx.lab_dir)
        / DATASET_CONFIG.get("paths", {}).get("splits_dir", "splits")
        / "test.parquet"
    )
    preds = model.predict(test_parquet)

    # 使用 scan_parquet 仅收集键列，避免不必要内存
    lf_keys = pl.scan_parquet(test_parquet).select(["datetime", "vt_symbol"]).collect()
    out = lf_keys.with_columns(pl.Series("signal", preds))
    lab.save_signal(ctx.name, out)


@register_step("backtesting")
def step_backtesting(ctx: WorkflowContext, step_cfg: Dict[str, Any]) -> None:
    lab: AlphaLab = ctx.lab
    # 加载信号
    signal = lab.load_signal(ctx.name)
    if signal is None:
        raise FileNotFoundError("信号文件不存在，请先执行 predict_signal")

    vt_symbols = ctx.component_symbols

    engine = BacktestingEngine(lab)
    bt_cfg = step_cfg.get("config", {})

    start_dt = datetime.fromisoformat(bt_cfg.get("start", ctx.train_period[0]))
    end_dt = datetime.fromisoformat(bt_cfg.get("end", ctx.valid_period[1]))
    capital = float(bt_cfg.get("capital", 100000000))

    engine.set_parameters(
        vt_symbols=vt_symbols,
        interval=ctx.interval_enum,
        start=start_dt,
        end=end_dt,
        capital=capital,
    )
    setting = bt_cfg.get("setting", {"top_k": 30, "n_drop": 3, "min_days": 3})
    engine.add_strategy(equity_demo_strategy.EquityDemoStrategy, setting, signal)

    engine.load_data()
    engine.run_backtesting()
    engine.calculate_result()
    engine.calculate_statistics()
    engine.show_chart()


@register_step("optimize_params")
def step_optimize_params(ctx: WorkflowContext, step_cfg: Dict[str, Any]) -> None:
    """使用 Optuna 进行参数调优"""
    lab: AlphaLab = ctx.lab
    # 加载信号
    signal = lab.load_signal(ctx.name)
    if signal is None:
        raise FileNotFoundError("信号文件不存在，请先执行 predict_signal")

    vt_symbols = ctx.component_symbols

    engine = BacktestingEngine(lab)
    opt_cfg = step_cfg.get("config", {})

    start_dt = datetime.fromisoformat(opt_cfg.get("start", ctx.train_period[0]))
    end_dt = datetime.fromisoformat(opt_cfg.get("end", ctx.valid_period[1]))
    capital = float(opt_cfg.get("capital", 100000000))

    engine.set_parameters(
        vt_symbols=vt_symbols,
        interval=ctx.interval_enum,
        start=start_dt,
        end=end_dt,
        capital=capital,
    )

    # 通过策略管理器加载策略及参数范围
    strategy_config_path = opt_cfg.get(
        "strategy_config_path",
        str(
            Path(__file__).resolve().parents[2]
            / "vnpy"
            / "alpha"
            / "strategy"
            / "strategy_configs.yaml"
        ),
    )
    strategy_name = opt_cfg.get("strategy_name", "EquityDemoStrategy")

    manager = StrategyManager(config_file=strategy_config_path)
    strategy_class = manager.get_strategy_class(strategy_name)
    base_setting = manager.get_default_params(strategy_name)
    param_ranges = manager.get_param_ranges(strategy_name)

    # 添加策略与信号（使用默认参数作为初始上下文）
    engine.add_strategy(strategy_class, base_setting, signal)

    # 构造 OptimizationSetting
    optimization_setting = OptimizationSetting()
    # 固定参数（默认参数里存在但未包含在范围内的）
    for k, v in base_setting.items():
        if k not in param_ranges:
            optimization_setting.add_parameter(k, float(v))

    # 需要优化的参数范围
    for name, rng in param_ranges.items():
        # 支持 [min, max, step] 的数值范围
        if isinstance(rng, list) and len(rng) == 3:
            optimization_setting.add_parameter(
                name, float(rng[0]), float(rng[1]), float(rng[2])
            )
        else:
            # 其他情况按固定值处理
            optimization_setting.add_parameter(
                name, float(rng[0]) if isinstance(rng, list) else float(rng)
            )

    target_name = opt_cfg.get("target_name", "sharpe_ratio")
    optimization_setting.set_target(target_name)

    n_trials = int(opt_cfg.get("n_trials", 50))
    timeout = opt_cfg.get("timeout", None)
    direction = opt_cfg.get("direction", "maximize")

    # 运行 Optuna 调优
    results = engine.run_optuna_optimization(
        optimization_setting,
        output=True,
        n_trials=n_trials,
        timeout=timeout,
        direction=direction,
    )

    # 输出最好结果并进行一次回测展示
    if results:
        best_params, best_value, stats = results[0]
        logger.info(f"最佳参数: {best_params}, 目标({target_name}): {best_value}")

        # 用最佳参数重新回测与展示
        engine = BacktestingEngine(lab)
        engine.set_parameters(
            vt_symbols=vt_symbols,
            interval=ctx.interval_enum,
            start=start_dt,
            end=end_dt,
            capital=capital,
        )
        engine.add_strategy(strategy_class, best_params, signal)
        engine.load_data()
        engine.run_backtesting()
        engine.calculate_result()
        engine.calculate_statistics()
        engine.show_chart()


# -----------------------------
# 子进程调用逻辑
# -----------------------------
def run_step_subprocess(step_type, config_path):
    cmd = [sys.executable, __file__, "run-step", step_type, "--config", config_path]
    logger.info(f"启动子进程: {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # 行缓冲
    )

    # 实时逐行读取日志
    for line in process.stdout:
        print(f"[子进程-{step_type}] {line}", end="")

    process.wait()

    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)


def main() -> None:
    # 主入口：读取 YAML，按 steps 顺序在子进程执行（尊重 enabled 开关）
    config_path = os.environ.get(
        "WORKFLOW_CONFIG", str(Path(__file__).parent / "config" / "workflow.yaml")
    )
    cfg = load_yaml(config_path)
    steps = cfg["workflow"].get("steps", [])
    for step in steps:
        step_type: str = step.get("type")
        enabled: bool = step.get("enabled", True)
        if not enabled:
            logger.info(f"跳过步骤（未启用）: {step_type}")
            continue
        logger.info(f"开始执行步骤: {step_type}")
        run_step_subprocess(step_type, config_path)
        logger.info(f"步骤完成: {step_type}")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "run-step":
        step_type = sys.argv[2]
        config_path = sys.argv[4] if "--config" in sys.argv else "config/workflow.yaml"
        cfg = load_yaml(config_path)
        ctx = build_context(cfg)
        step_cfg = next(
            (s for s in cfg["workflow"]["steps"] if s["type"] == step_type), {}
        )
        if step_type not in STEP_REGISTRY:
            raise ValueError(f"未知的步骤类型: {step_type}")
        STEP_REGISTRY[step_type](ctx, step_cfg)
    else:
        main()
