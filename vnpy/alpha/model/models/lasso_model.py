import os
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.linear_model import Lasso      # type: ignore

from vnpy.alpha import (
    AlphaModel,
    Segment,
    logger
)


class LassoModel(AlphaModel):
    """LASSO regression learning algorithm"""

    def __init__(
        self,
        alpha: float = 0.0005,
        max_iter: int = 1000,
        random_state: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        alpha : float
            Regularization parameter
        max_iter : int
            Maximum number of iterations
        random_state : int
            Random seed
        """
        self.alpha: float = alpha
        self.max_iter: int = max_iter
        self.random_state: int | None = random_state

        self.model: Lasso = None

        self.feature_names: list[str] = []

    # Legacy fit(dataset) removed. Use fit(splits_dir) instead.

    # Legacy predict(dataset, segment) removed. Use predict(parquet_path) instead.

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

        df_train = pd.concat([df_train, df_valid], axis=0)
        # 切分文件已无键列，直接按特征与 label 构造
        self.feature_names = [c for c in df_train.columns if c not in ["datetime", "vt_symbol", "label"]]
        X: np.ndarray = df_train[self.feature_names].to_numpy()
        y: np.ndarray = df_train["label"].to_numpy()

        self.model = Lasso(
            alpha=self.alpha,
            max_iter=self.max_iter,
            random_state=self.random_state,
            fit_intercept=False,
            copy_X=False,
        )
        self.model.fit(X, y)

    def predict(self, parquet_path: str | os.PathLike) -> np.ndarray:
        """从保存的切分文件（如 `test.parquet`）读取特征并预测。"""
        if self.model is None:
            raise ValueError("model is not fitted yet!")
        import pandas as pd
        df = pd.read_parquet(parquet_path)
        feat_cols = [c for c in df.columns if c not in ["datetime", "vt_symbol", "label"]]
        data: np.ndarray = df[feat_cols].to_numpy()
        return self.model.predict(data)

    def detail(self) -> None:
        """
        Output detailed information about the model

        Displays feature importance based on the coefficients
        of the LASSO model, showing only non-zero features
        sorted by absolute value.
        """
        # Get feature coefficients
        coef: np.ndarray = self.model.coef_

        # Extract feature coefficients
        data: list[tuple[str, float]] = list(zip(self.feature_names, coef, strict=False))

        # Filter non-zero features
        data = [x for x in data if x[1]]

        # Sort by absolute value
        data.sort(key=lambda x: abs(x[1]), reverse=True)

        # Filter out features with very small coefficients
        data = [x for x in data if round(x[1], 6) != 0]

        # Print feature importance
        logger.info(f"LASSO模型特征总数量: {len(data)}")

        for name, importance in data:
            logger.info(f"{name}: {importance:.6f}")
