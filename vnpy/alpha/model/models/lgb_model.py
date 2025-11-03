from typing import cast

import os
from pathlib import Path
import numpy as np
import polars as pl
import lightgbm as lgb
import matplotlib.pyplot as plt

from vnpy.alpha.model import AlphaModel


class LgbModel(AlphaModel):
    """LightGBM ensemble learning algorithm"""

    def __init__(
        self,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        num_boost_round: int = 1000,
        early_stopping_rounds: int = 50,
        log_evaluation_period: int = 1,
        seed: int | None = None
    ):
        """
        Parameters
        ----------
        learning_rate : float
            Learning rate
        num_leaves : int
            Number of leaf nodes
        num_boost_round : int
            Maximum number of training rounds
        early_stopping_rounds : int
            Number of rounds for early stopping
        log_evaluation_period : int
            Interval rounds for printing training logs
        seed : int | None
            Random seed
        """
        self.params: dict = {
            "objective": "mse",
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "seed": seed
        }

        self.num_boost_round: int = num_boost_round
        self.early_stopping_rounds: int = early_stopping_rounds
        self.log_evaluation_period: int = log_evaluation_period

        self.model: lgb.Booster | None = None

    # Legacy _prepare_data(dataset) removed. Use splits-based training.

    # Legacy fit(dataset) removed. Use fit(splits_dir).

    # Legacy predict(dataset, segment) removed. Use predict(parquet_path).

    # ===== 基于 Parquet 切分的训练/预测 =====
    def fit(self, splits_dir: str | os.PathLike) -> None:
        """从保存的切分文件 `train.parquet` 与 `valid.parquet` 进行训练。"""
        splits_path = Path(splits_dir)
        train_path = splits_path / "train.parquet"
        valid_path = splits_path / "valid.parquet"
        if not train_path.exists() or not valid_path.exists():
            raise FileNotFoundError("train.parquet 或 valid.parquet 不存在于切分目录")

        import pandas as pd
        df_train = pd.read_parquet(train_path)
        df_valid = pd.read_parquet(valid_path)

        feat_cols = [c for c in df_train.columns if c not in ["datetime", "vt_symbol", "label"]]
        X_train = df_train[feat_cols]
        y_train = df_train["label"].values
        X_valid = df_valid[feat_cols]
        y_valid = df_valid["label"].values

        train_ds = lgb.Dataset(X_train, label=y_train)
        valid_ds = lgb.Dataset(X_valid, label=y_valid)
        self.model = lgb.train(
            self.params,
            train_ds,
            num_boost_round=self.num_boost_round,
            valid_sets=[train_ds, valid_ds],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(self.early_stopping_rounds),
                lgb.log_evaluation(self.log_evaluation_period),
            ],
        )

    def predict(self, parquet_path: str | os.PathLike) -> np.ndarray:
        """从保存的切分文件（如 `test.parquet`）读取特征并预测。"""
        if self.model is None:
            raise ValueError("model is not fitted yet!")
        import pandas as pd
        df = pd.read_parquet(parquet_path)
        feat_cols = [c for c in df.columns if c not in ["datetime", "vt_symbol", "label"]]
        data: np.ndarray = df[feat_cols].to_numpy()
        return cast(np.ndarray, self.model.predict(data))

    def detail(self) -> None:
        """
        Display model details with feature importance plots

        Generates two plots showing feature importance based on
        'split' and 'gain' metrics.

        Returns
        -------
        None
        """
        if not self.model:
            return

        for importance_type in ["split", "gain"]:
            ax: plt.Axes = lgb.plot_importance(
                self.model,
                max_num_features=50,
                importance_type=importance_type,
                figsize=(10, 20)
            )
            ax.set_title(f"Feature Importance ({importance_type})")
