from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchmetrics
from sklearn.metrics import mean_absolute_error, r2_score
from torch import optim

from pipelines.models.simple_linear_cnn_multi_out_3 import SimpleLinearCnnMO3, ConvolutionalSimpleModel


class MultiOutCnnHandler:
    def __init__(self, cnn_type: str, model_params: Optional[Dict[str, any]] = None, optimizer_params: Optional[Dict[str, any]] = None, criterion=None):
        """
        Initialize ModelHandler with a model type (cnn_type), optional model parameters, optimizer parameters, and a criterion.
        """
        self.cnn_type = cnn_type
        self.model_params = model_params if model_params is not None else {}  # Default to empty dictionary if None
        self.model = self.create_model()  # Initializes the model based on cnn_type
        self.optimizer_params = optimizer_params if optimizer_params else {"lr": 0.01}
        self.optimizer = optim.SGD(self.model.parameters(), **self.optimizer_params)
        self.criterion = criterion if criterion else nn.MSELoss()
        self.rmse_metric = torchmetrics.MeanSquaredError(squared=False)

    def create_model(self):
        """Creates and initializes the model based on cnn_type."""
        if self.cnn_type == "linear_cnn":
            return SimpleLinearCnnMO3(**self.model_params)
        elif self.cnn_type == "convolutional_cnn":
            return ConvolutionalSimpleModel(**self.model_params)
        else:
            raise ValueError(f"Unknown model type: {self.cnn_type}")

    def train_model(self, dataloader, epochs=10):
        """Train the model and return a DataFrame with training metrics for each epoch."""
        self.model.train()
        epochs_data = []

        for epoch in range(epochs):
            running_loss = 0.0
            self.rmse_metric.reset()

            for batch, y1, y2, y3 in dataloader:
                self.optimizer.zero_grad()

                outputs_y1, outputs_y2, outputs_y3 = self.model(batch)
                loss_y1 = self.criterion(outputs_y1, y1)
                loss_y2 = self.criterion(outputs_y2, y2)
                loss_y3 = self.criterion(outputs_y3, y3)

                loss = loss_y1 + loss_y2 + loss_y3
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                self.rmse_metric.update(outputs_y1, y1)
                self.rmse_metric.update(outputs_y2, y2)
                self.rmse_metric.update(outputs_y3, y3)

            epoch_rmse = self.rmse_metric.compute().item()
            epochs_data.append({
                "epoch": epoch + 1,
                "average_loss": running_loss / len(dataloader),
                "rmse": epoch_rmse
            })

        return pd.DataFrame(epochs_data)

    def evaluate_multi_output_metrics(self, test_loader, criterion):
        self.model.eval()

        rmse_y1 = torchmetrics.MeanSquaredError(squared=False)
        rmse_y2 = torchmetrics.MeanSquaredError(squared=False)
        rmse_y3 = torchmetrics.MeanSquaredError(squared=False)

        test_loss = 0.0
        total_test_loss = []

        all_y1_true, all_y2_true, all_y3_true = [], [], []
        all_y1_pred, all_y2_pred, all_y3_pred = [], [], []

        with torch.no_grad():
            for inputs, y1, y2, y3 in test_loader:
                outputs_y1, outputs_y2, outputs_y3 = self.model(inputs)

                loss_y1 = criterion(outputs_y1, y1)
                loss_y2 = criterion(outputs_y2, y2)
                loss_y3 = criterion(outputs_y3, y3)

                loss = loss_y1 + loss_y2 + loss_y3
                test_loss += loss.item()
                total_test_loss.append(loss.item())

                rmse_y1.update(outputs_y1, y1)
                rmse_y2.update(outputs_y2, y2)
                rmse_y3.update(outputs_y3, y3)

                all_y1_true.extend(y1.cpu().numpy())
                all_y2_true.extend(y2.cpu().numpy())
                all_y3_true.extend(y3.cpu().numpy())

                all_y1_pred.extend(outputs_y1.cpu().numpy())
                all_y2_pred.extend(outputs_y2.cpu().numpy())
                all_y3_pred.extend(outputs_y3.cpu().numpy())

        rmse_y1_value = rmse_y1.compute().item()
        rmse_y2_value = rmse_y2.compute().item()
        rmse_y3_value = rmse_y3.compute().item()

        # numpy arrays para MAE y R²
        all_y1_true = np.array(all_y1_true)
        all_y2_true = np.array(all_y2_true)
        all_y3_true = np.array(all_y3_true)

        all_y1_pred = np.array(all_y1_pred)
        all_y2_pred = np.array(all_y2_pred)
        all_y3_pred = np.array(all_y3_pred)

        # Calculando MAE and R² for each output
        mae_y1 = mean_absolute_error(all_y1_true, all_y1_pred)
        mae_y2 = mean_absolute_error(all_y2_true, all_y2_pred)
        mae_y3 = mean_absolute_error(all_y3_true, all_y3_pred)

        r2_y1 = r2_score(all_y1_true, all_y1_pred)
        r2_y2 = r2_score(all_y2_true, all_y2_pred)
        r2_y3 = r2_score(all_y3_true, all_y3_pred)

        results_df = pd.DataFrame(
            {
                "Metric": ["RMSE", "MAE", "R²"],
                "Common Flares (y1)": [rmse_y1_value, mae_y1, r2_y1],
                "Moderate Flares (y2)": [rmse_y2_value, mae_y2, r2_y2],
                "Severe Flares (y3)": [rmse_y3_value, mae_y3, r2_y3]
            }
        )

        rmse_y1.reset()
        rmse_y2.reset()
        rmse_y3.reset()

        return results_df, total_test_loss

    def save_model(self, model_path: str):
        """Save the model, optimizer, criterion, and other settings to a single file."""
        save_state = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "criterion": self.criterion.__class__.__name__,
            "optimizer_params": self.optimizer_params,
            "model_params": self.model_params,
            "cnn_type": self.cnn_type,
        }
        torch.save(save_state, model_path)
        print(f"Model and parameters saved at {model_path}")

    def load_model(self, model_path: str):
        loaded_state = torch.load(model_path)

        # Recreate the model, optimizer, and criterion with saved configurations
        self.cnn_type = loaded_state["cnn_type"]
        self.model_params = loaded_state["model_params"]
        self.model = self.create_model()  # Reinitialize the model
        self.model.load_state_dict(loaded_state["model_state_dict"])

        self.optimizer_params = loaded_state["optimizer_params"]
        self.optimizer = torch.optim.SGD(self.model.parameters(), **self.optimizer_params)
        self.optimizer.load_state_dict(loaded_state["optimizer_state_dict"])

        criterion_class = getattr(torch.nn, loaded_state["criterion"])
        self.criterion = criterion_class()

        print(f"Model and parameters loaded from {model_path}")
