import copy
from collections import defaultdict
from typing import Literal, cast

import os
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import mean_squared_error      # type: ignore
import torch
import torch.nn as nn
import torch.optim as optim

from vnpy.alpha import (
    AlphaModel,
    Segment,
    logger
)



class MlpModel(AlphaModel):
    """
    Multi-Layer Perceptron Model

    Alpha factor prediction model implemented using multi-layer perceptron, with main features including:
    1. Building and training multi-layer perceptron neural networks
    2. Predicting Alpha factor values
    3. Model evaluation and feature importance analysis
    4. Support for early stopping and overfitting prevention
    5. Support for MSE loss function
    6. Optional Adam or SGD optimizer
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: tuple[int] = (256,),
        lr: float = 0.001,
        n_epochs: int = 300,
        batch_size: int = 2000,
        early_stop_rounds: int = 50,
        eval_steps: int = 20,
        optimizer: Literal["sgd", "adam"] = "adam",
        weight_decay: float = 0.0,
        device: str = "cpu",
        seed: int | None = None
    ) -> None:
        """
        Initialize MLP model

        Parameters
        ----------
        input_size : int, default 360
            Input feature dimension
        hidden_sizes : tuple[int], default (256,)
            Number of neurons in hidden layers
        lr : float, default 0.001
            Learning rate
        n_epochs : int, default 300
            Maximum training steps
        batch_size : int, default 2000
            Number of samples per batch
        early_stop_rounds : int, default 50
            Early stopping rounds, training stops if validation loss doesn't improve within these rounds
        eval_steps : int, default 20
            Evaluate model every this many steps
        optimizer : Literal["sgd", "adam"], default "adam"
            Optimizer type, options are "sgd" or "adam"
        weight_decay : float, default 0.0
            L2 regularization coefficient
        seed : Optional[int], optional
            Random seed for reproducibility
        device : str, default "cpu"
            Training device
        """
        # Save model hyperparameters
        self.input_size: int = input_size
        self.hidden_sizes: tuple[int] = hidden_sizes
        self.lr: float = lr
        self.n_epochs: int = n_epochs
        self.batch_size: int = batch_size
        self.early_stop_rounds: int = early_stop_rounds
        self.eval_steps: int = eval_steps
        self.device: str = device
        self.fitted: bool = False
        self.feature_names: list[str] = []
        self.best_step: int | None = None

        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Set loss function type
        self._scorer = mean_squared_error

        # Initialize model
        self.model: nn.Module = MlpNetwork(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
        )

        # Move model to specified device
        self.model = self.model.to(device)

        # Set optimizer
        optimizer_name = optimizer.lower()
        if optimizer_name == "adam":
            self.optimizer: optim.Optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            raise NotImplementedError(f"optimizer {optimizer} is not supported!")

        # Set learning rate scheduler
        self.scheduler: optim.lr_scheduler.ReduceLROnPlateau = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            threshold=0.0001,
            threshold_mode="rel",
            cooldown=0,
            min_lr=0.00001,
            eps=1e-08,
        )

    # Legacy dataset-based fit removed. Use fit(splits_dir) instead.

    def _train_step(
        self,
        train_valid_data: dict[str, dict[Segment, torch.Tensor]],
        train_samples: int
    ) -> float:
        """
        Execute one training step

        Parameters
        ----------
        train_valid_data : dict
            Training and validation data
        train_samples : int
            Number of training samples

        Returns
        -------
        float
            Current batch loss value
        """
        batch_loss = AverageMeter()
        self.model.train()
        self.optimizer.zero_grad()

        # Randomly select batch data
        batch_indices = np.random.choice(train_samples, self.batch_size)
        batch_features = train_valid_data["x"][Segment.TRAIN][batch_indices]
        batch_labels = train_valid_data["y"][Segment.TRAIN][batch_indices]

        # Forward and backward propagation
        predictions = self.model(batch_features)
        cur_loss = self._loss_fn(predictions, batch_labels)
        cur_loss.backward()

        # Update model parameters
        self.optimizer.step()
        batch_loss.update(cur_loss.item())

        return batch_loss.val

    def _evaluate_step(
        self,
        train_valid_data: dict[str, dict[Segment, torch.Tensor]],
        evaluation_results: dict[Segment, list[float]],
        step: int,
        train_loss: float,
        early_stop_count: int,
        best_valid_score: float
    ) -> tuple[int, float, dict[str, torch.Tensor] | None]:
        """
        Evaluate current model performance

        Parameters
        ----------
        train_valid_data : dict
            Training and validation data
        evaluation_results : dict
            Evaluation results record
        step : int
            Current training step
        train_loss : float
            Current training loss
        early_stop_count : int
            Count of steps without improvement
        best_valid_score : float
            Best validation loss

        Returns
        -------
        tuple[int, float, dict] | None
            Returns updated early stop count, best validation loss, and best model parameters
        """
        early_stop_count += 1
        train_loss /= self.eval_steps

        # Evaluate model on validation set
        with torch.no_grad():
            self.model.eval()

            data: torch.Tensor = train_valid_data["x"][Segment.VALID]
            pred: torch.Tensor = cast(torch.Tensor, self._predict_batch(data, return_cpu=False))
            valid_loss = self._loss_fn(pred, train_valid_data["y"][Segment.VALID])

            loss_val = valid_loss.item()

        # Record evaluation results
        logger.info(f"[Step {step}]: train_loss {train_loss:.6f}, valid_loss {loss_val:.6f}")
        evaluation_results[Segment.TRAIN].append(train_loss)
        evaluation_results[Segment.VALID].append(loss_val)

        # Update best model if validation performance improves
        best_params = None
        if loss_val < best_valid_score:
            logger.info(f"\t验证集损失从 {best_valid_score:.6f} 降低到 {loss_val:.6f}")
            best_valid_score = loss_val
            self.best_step = step
            early_stop_count = 0
            best_params = copy.deepcopy(self.model.state_dict())

        # Update learning rate
        if self.scheduler is not None:
            self.scheduler.step(metrics=valid_loss, epoch=step)

        return early_stop_count, best_valid_score, best_params

    def _loss_fn(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate loss value

        Parameters
        ----------
        pred : torch.Tensor
            Model predictions
        target : torch.Tensor
            Target true values

        Returns
        -------
        torch.Tensor
            Calculated loss value
        """
        pred, target = pred.reshape(-1), target.reshape(-1)
        loss: torch.Tensor = nn.MSELoss()(pred, target)
        return loss

    def _predict_batch(self, data: torch.Tensor, return_cpu: bool = True) -> np.ndarray | torch.Tensor:
        """
        Neural network prediction function

        Parameters
        ----------
        data : torch.Tensor
            Input data
        return_cpu : bool, default True
            Whether to return CPU tensor
        step : Optional[int], optional
            Current training step

        Returns
        -------
        np.ndarray | torch.Tensor
            Model prediction results
        """
        data = data.to(self.device)

        predictions: list[torch.Tensor] = []

        self.model.eval()

        with torch.no_grad():
            batch_size: int = 8096
            for i in range(0, len(data), batch_size):
                x: torch.Tensor = data[i: i + batch_size]
                predictions.append(self.model(x.to(self.device)).detach().reshape(-1))

        if return_cpu:
            return cast(np.ndarray, np.concatenate([pr.cpu().numpy() for pr in predictions]))
        else:
            return torch.cat(predictions, dim=0)

    # Legacy dataset-based predict removed. Use predict(parquet_path) instead.

    # ===== 基于 Parquet 切分的训练/预测 =====
    def fit(self, splits_dir: str | os.PathLike) -> None:
        """从保存的切分文件 `train.parquet` 与 `valid.parquet` 进行训练。"""
        splits_path = Path(splits_dir)
        train_path = splits_path / "train.parquet"
        valid_path = splits_path / "valid.parquet"
        if not train_path.exists() or not valid_path.exists():
            raise FileNotFoundError("train.parquet 或 valid.parquet 不存在于切分目录")

        # 使用 Polars scan_parquet 流式读取，减少内存峰值
        lf_train = pl.scan_parquet(str(train_path))
        lf_valid = pl.scan_parquet(str(valid_path))

        cols = lf_train.columns
        feat_cols = [c for c in cols if c not in ["datetime", "vt_symbol", "label"]]

        # 收集为 numpy（可进一步分块，但先一次 collect）
        df_train = lf_train.select(feat_cols + ["label"]).drop_nulls().collect()
        df_valid = lf_valid.select(feat_cols + ["label"]).drop_nulls().collect()

        x_train = torch.from_numpy(df_train.select(feat_cols).to_numpy()).float().to(self.device)
        y_train = torch.from_numpy(df_train["label"].to_numpy().reshape(-1)).float().to(self.device)
        x_valid = torch.from_numpy(df_valid.select(feat_cols).to_numpy()).float().to(self.device)
        y_valid = torch.from_numpy(df_valid["label"].to_numpy().reshape(-1)).float().to(self.device)

        self.feature_names = feat_cols

        train_valid_data: dict[str, dict] = defaultdict(dict)
        train_valid_data["x"][Segment.TRAIN] = x_train
        train_valid_data["y"][Segment.TRAIN] = y_train
        train_valid_data["x"][Segment.VALID] = x_valid
        train_valid_data["y"][Segment.VALID] = y_valid

        evaluation_results: dict = {Segment.TRAIN: [], Segment.VALID: []}

        early_stop_count: int = 0
        train_loss: float = 0
        best_valid_score: float = np.inf
        best_params = None

        train_samples: int = train_valid_data["y"][Segment.TRAIN].shape[0]
        logger.info(f"开始训练模型, 总样本数: {train_samples}")

        for step in range(1, self.n_epochs + 1):
            if early_stop_count >= self.early_stop_rounds:
                logger.info("达到早停条件,训练结束")
                break

            batch_loss = self._train_step(train_valid_data, train_samples)
            train_loss += batch_loss

            if step % 10 == 0:
                logger.info(f"Step {step}/{self.n_epochs}, Batch Loss: {batch_loss:.6f}")

            if step % self.eval_steps == 0 or step == self.n_epochs:
                early_stop_count, best_valid_score, best_params = self._evaluate_step(
                    train_valid_data,
                    evaluation_results,
                    step,
                    train_loss,
                    early_stop_count,
                    best_valid_score,
                )
                train_loss = 0

        if best_params:
            self.model.load_state_dict(best_params)
        self.fitted = True

    def predict(self, parquet_path: str | os.PathLike) -> np.ndarray:
        """从保存的切分文件（如 `test.parquet`）读取特征并预测。"""
        if not self.fitted:
            raise ValueError("model is not fitted yet!")
        # 使用 scan_parquet 流式读取，保留键列以便外层可用
        lf = pl.scan_parquet(str(parquet_path))
        lf = lf.sort(["datetime", "vt_symbol"])
        cols = lf.columns
        feat_cols = [c for c in cols if c not in ["datetime", "vt_symbol", "label"]]
        df = lf.select(feat_cols).collect()
        data = torch.from_numpy(df.to_numpy()).float()
        return cast(np.ndarray, self._predict_batch(data))

    def _check_tensor_nan(self, tensor: torch.Tensor, name: str) -> None:
        """
        Check if tensor contains NaN values

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor to check
        name : str
            Tensor name

        Returns
        -------
        None
        """
        if torch.isnan(tensor).any():
            print(f"NaN values detected: {name}")

    def detail(self) -> pd.DataFrame | None:
        """
        Output MLP model detail information

        Returns
        -------
        pd.DataFrame
            Feature importance dataframe
        """
        if not self.fitted:
            logger.info("模型尚未训练，无法显示详细信息")
            return None

        # 显示模型基本信息
        logger.info(f"输入特征维度: {self.input_size}")
        logger.info(f"隐藏层大小: {self.hidden_sizes}")

        # 计算模型总参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"模型总参数量: {total_params:,}")

        # 显示训练状态信息
        logger.info(f"训练设备: {self.device}")
        logger.info(f"当前学习率: {self.lr}")
        logger.info(f"批次大小: {self.batch_size}")

        # Calculate feature importance
        importance_df = self._calculate_feature_importance()
        return importance_df

    def _calculate_feature_importance(self) -> pd.DataFrame:
        """
        Calculate feature importance

        Returns
        -------
        pd.DataFrame
            Feature importance dataframe
        """
        self.model.eval()
        importance_dict = {}

        test_data = torch.randn(1000, self.input_size).to(self.device)
        base_pred = self.model(test_data).detach()

        noise_level = 0.1
        for i, feature_name in enumerate(self.feature_names):
            perturbed_data = test_data.clone()
            perturbed_data[:, i] += torch.randn(1000).to(self.device) * noise_level

            with torch.no_grad():
                new_pred = self.model(perturbed_data)
                importance = torch.std(torch.abs(new_pred - base_pred)).item()
                importance_dict[feature_name] = importance

        df = pd.DataFrame({
            'Feature': list(importance_dict.keys()),
            'Importance': list(importance_dict.values())
        })
        df = df.sort_values('Importance', ascending=False)
        df = df.set_index('Feature')

        return df


class AverageMeter:
    """
    Class for calculating and storing average and current values

    Attributes
    ----------
    val : float
        Current value
    avg : float
        Average value
    sum : float
        Sum
    count : int
        Count
    """

    def __init__(self) -> None:
        """
        Initialize AverageMeter

        Returns
        -------
        None
        """
        self.reset()

    def reset(self) -> None:
        """
        Reset all statistics

        Returns
        -------
        None
        """
        self.val: float = 0
        self.avg: float = 0
        self.sum: float = 0
        self.count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        """
        Update statistics

        Parameters
        ----------
        val : float
            Current value
        n : int, default 1
            Current batch size

        Returns
        -------
        None
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MlpNetwork(nn.Module):
    """
    Deep Neural Network Model Structure

    Used to build multi-layer perceptron network structure, supporting multiple hidden layers
    and different activation functions.

    Attributes
    ----------
    network : nn.ModuleList
        List of neural network layers
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 1,
        hidden_sizes: tuple[int] = (256,),
        activation: str = "LeakyReLU"
    ) -> None:
        """
        Constructor

        Parameters
        ----------
        input_size : int
            Input feature dimension, i.e., number of features per sample
        output_size : int, default 1
            Output dimension, used for predicting target values
        hidden_sizes : tuple[int], default (256,)
            Tuple of hidden layer neuron counts, e.g., (256, 128) represents two hidden layers
            with 256 and 128 neurons respectively
        activation : str, default "LeakyReLU"
            Activation function type, options:
            - "LeakyReLU": Leaky ReLU function
            - "SiLU": Sigmoid Linear Unit function
        """
        super().__init__()

        # Build network layers
        layers: list[nn.Module] = []
        layer_sizes = [input_size] + list(hidden_sizes)

        # Input layer Dropout
        layers.append(nn.Dropout(0.05))

        # Build hidden layers
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:], strict=False):
            # Add a neural network block: linear layer + batch normalization + activation function
            layers.extend([
                nn.Linear(in_size, out_size),
                nn.BatchNorm1d(out_size),
                self._get_activation(activation)
            ])

        # Output layer
        layers.extend([
            nn.Dropout(0.05),
            nn.Linear(hidden_sizes[-1], output_size)
        ])

        # Combine all layers into a sequence
        self.network = nn.ModuleList(layers)

        # Initialize network weights
        self._initialize_weights()

    def _get_activation(self, name: str) -> nn.Module:
        """
        Get specified activation function layer

        Parameters
        ----------
        name : str
            Activation function name

        Returns
        -------
        nn.Module
            Activation function layer instance

        Raises
        ------
        ValueError
            When an unsupported activation function type is specified
        """
        if name == "LeakyReLU":
            return nn.LeakyReLU(negative_slope=0.1)
        elif name == "SiLU":
            return nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation function type: {name}")

    def _initialize_weights(self) -> None:
        """
        Initialize network weight parameters

        Uses Kaiming initialization method for all linear layers, which is particularly
        suitable for deep networks using LeakyReLU activation functions.

        Returns
        -------
        None
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight,
                    a=0.1,                  # LeakyReLU negative slope
                    mode="fan_in",          # Scale using input node count
                    nonlinearity="leaky_relu"
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation calculation

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor, shape (batch_size, input_size)

        Returns
        -------
        torch.Tensor
            Model output tensor, shape (batch_size, output_size)
        """
        # Pass through all layers in the network sequentially
        for layer in self.network:
            x = layer(x)
        return x
