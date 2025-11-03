from abc import ABCMeta, abstractmethod
from typing import Any
import os
import numpy as np


class AlphaModel(metaclass=ABCMeta):
    """Template class for machine learning algorithms"""

    @abstractmethod
    def fit(self, splits_dir: str | os.PathLike) -> None:
        """
        Train the model using pre-split parquet files located in `splits_dir`.
        Expected files: `train.parquet` and `valid.parquet`.
        """
        pass

    @abstractmethod
    def predict(self, parquet_path: str | os.PathLike) -> np.ndarray:
        """
        Make predictions using a single parquet file (e.g., `test.parquet`).
        """
        pass

    def detail(self) -> Any:
        """
        Output detailed information about the model
        """
        return
