from typing import cast

import os
from pathlib import Path
import numpy as np
import polars as pl
import lightgbm as lgb
import matplotlib.pyplot as plt

from vnpy.alpha.dataset import AlphaDataset, Segment
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

    def _prepare_data(self, dataset: AlphaDataset) -> list[lgb.Dataset]:
        """
        Prepare data for training and validation

        Parameters
        ----------
        dataset : AlphaDataset
            The dataset containing features and labels

        Returns
        -------
        list[lgb.Dataset]
            List of LightGBM datasets for training and validation
        """
        ds: list[lgb.Dataset] = []

        # Process training and validation separately
        for segment in [Segment.TRAIN, Segment.VALID]:
            # Get data for learning
            df: pl.DataFrame = dataset.fetch_feat(segment)
            #df = df.sort(["datetime", "vt_symbol"])

            # Convert to numpy arrays
            data = df.select(df.columns[2: -1]).to_pandas()
            label = np.array(df["label"])

            # Add training data
            ds.append(lgb.Dataset(data, label=label))

        return ds

    def fit(self, dataset: AlphaDataset) -> None:
        """
        Fit the model using the dataset

        Parameters
        ----------
        dataset : AlphaDataset
            The dataset containing features and labels

        Returns
        -------
        None
        """
        # Prepare task data
        ds: list[lgb.Dataset] = self._prepare_data(dataset)

        # Execute model training
        self.model = lgb.train(
            self.params,
            ds[0],
            num_boost_round=self.num_boost_round,
            valid_sets=ds,
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(self.early_stopping_rounds),      # Early stopping callback
                lgb.log_evaluation(self.log_evaluation_period)       # Logging callback
            ]
        )

    def predict(self, dataset: AlphaDataset, segment: Segment) -> np.ndarray:
        """
        Make predictions using the trained model

        Parameters
        ----------
        dataset : AlphaDataset
            The dataset containing features
        segment : Segment
            The segment to make predictions on

        Returns
        -------
        np.ndarray
            Prediction results

        Raises
        ------
        ValueError
            If the model has not been fitted yet
        """
        # Check if model exists
        if self.model is None:
            raise ValueError("model is not fitted yet!")

        # Get data for inference
        df: pl.DataFrame = dataset.fetch_feat(segment)
        df = df.sort(["datetime", "vt_symbol"])

        # Convert to numpy array
        data: np.ndarray = df.select(df.columns[2: -1]).to_numpy()

        # Return prediction results
        result: np.ndarray = cast(np.ndarray, self.model.predict(data))
        return result

    # ===== 基于 Parquet 切分的训练/预测 =====
    def fit_splits(self, splits_dir: str | os.PathLike) -> None:
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

    def predict_splits(self, parquet_path: str | os.PathLike) -> np.ndarray:
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
