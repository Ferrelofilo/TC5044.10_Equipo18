from utils.logger_setup import setup_logger
import json
import os
import sys
from torchinfo import summary

from dataclasses import dataclass

import mlflow
import mlflow.pytorch
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Inicializa el logger
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

logger = setup_logger(__name__)


@dataclass
class TrainingSession:
    """Clase que encapsula el proceso de entrenamiento y manejo de artefactos del modelo."""

    model: nn.Module
    optimizer: optim.Optimizer
    criterion: nn.Module
    rmse_metric: any
    train_loader: DataLoader
    epochs: int = 10
    output_folder: str = "models/"

    def run_training(self):
        """Ejecuta el ciclo de entrenamiento del modelo."""
        self.model.train()
        epochs_data = []

        with mlflow.start_run():
            # Log hyperparameters and initial settings
            mlflow.log_param("epochs", self.epochs)
            mlflow.log_param("optimizer", type(self.optimizer).__name__)
            mlflow.log_param("learning_rate", self.optimizer.param_groups[0]["lr"])

            for epoch in range(self.epochs):
                running_loss = 0.0
                self.rmse_metric.reset()

                for batch, y_train_combined in self.train_loader:
                    self.optimizer.zero_grad()

                    y1, y2, y3 = (
                        y_train_combined[:, 0:1],
                        y_train_combined[:, 1:2],
                        y_train_combined[:, 2:3],
                    )

                    outputs_y1, outputs_y2, outputs_y3 = self.model(batch)

                    loss_y1 = self.criterion(outputs_y1, y1)
                    loss_y2 = self.criterion(outputs_y2, y2)
                    loss_y3 = self.criterion(outputs_y3, y3)

                    loss = loss_y1 + loss_y2 + loss_y3
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item()

                    self.rmse_metric.update(outputs_y1, y1, index=0)
                    self.rmse_metric.update(outputs_y2, y2, index=1)
                    self.rmse_metric.update(outputs_y3, y3, index=2)

                rmse_y1, rmse_y2, rmse_y3 = self.rmse_metric.compute()
                average_loss = running_loss / len(self.train_loader)
                logger.debug(
                    f"Época [{epoch+1}/{self.epochs}] - Pérdida: {average_loss}, "
                    f"RMSE Y1: {rmse_y1:.4f}, RMSE Y2: {rmse_y2:.4f}, RMSE Y3: {rmse_y3:.4f}"
                )

                params = {
                    "epochs": self.epochs,
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                    "metric_function": str(self.rmse_metric),
                }
                # Log training parameters.
                mlflow.log_params(params)
                mlflow.log_metric("epoch_loss", average_loss, step=epoch)
                mlflow.log_metric("rmse_y1", rmse_y1.item(), step=epoch)
                mlflow.log_metric("rmse_y2", rmse_y2.item(), step=epoch)
                mlflow.log_metric("rmse_y3", rmse_y3.item(), step=epoch)

                epochs_data.append(
                    {
                        "epoch": epoch + 1,
                        "average_loss": average_loss,
                        "rmse_y1": rmse_y1.item(),
                        "rmse_y2": rmse_y2.item(),
                        "rmse_y3": rmse_y3.item(),
                    }
                )

            mlflow.pytorch.log_model(self.model, "model")
            with open("model_summary.txt", "w") as f:
                f.write(str(summary(self.model)))
            mlflow.log_artifact(
                local_path="model_summary.txt", artifact_path="model_summary.txt"
            )

        # Devuelve los datos del entrenamiento como un DataFrame
        return pd.DataFrame(epochs_data)

    def save_model_artifacts(self, epochs_df: pd.DataFrame):
        """Guarda los artefactos del modelo y los datos del entrenamiento como JSON y CSV."""
        os.makedirs(self.output_folder, exist_ok=True)
        # date_str = datetime.now().strftime("%Y-%m-%d")
        artifacts_dir = os.path.join(self.output_folder, "artifacts_")
        os.makedirs(artifacts_dir, exist_ok=True)

        model_weights_path = os.path.join(artifacts_dir, "model_weights.pth")
        torch.save(self.model.state_dict(), model_weights_path)

        # Crea un archivo JSON con los metadatos del modelo
        model_artifacts = {
            "model_name": "SimpleLinearCnnMO3",
            "input_len": self.model.first_dense.in_features,
            "out_features1": self.model.first_dense.out_features,
            "out_features2": self.model.second_dense.out_features,
            "model_weights_path": model_weights_path,
            "training_epochs": len(epochs_df),
            "training_loss": epochs_df["average_loss"].tolist(),
            # "training_rmse": epochs_df["rmse"].tolist(),
        }

        # Guarda los metadatos en un archivo JSON
        json_path = os.path.join(artifacts_dir, "model_artifacts.json")
        with open(json_path, "w") as json_file:
            json.dump(model_artifacts, json_file, indent=4)

        logger.debug(f"Artefactos del modelo guardados en {json_path}")

        csv_path = os.path.join(artifacts_dir, "training_metrics.csv")
        epochs_df.to_csv(csv_path, index=False)
        logger.debug(f"Métricas de entrenamiento guardadas en {csv_path}")


if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
    from pipelines.models.simple_linear_cnn_multi_out_3 import SimpleLinearCnnMO3
    from pipelines.utils.rmse_metric import RMSEMetric

    # Rutas para cargar los datos de entrenamiento
    X_train_path = sys.argv[1]
    y_train_path = sys.argv[2]
    output_folder = sys.argv[3]

    # Cargar los datos de entrenamiento
    X_train = torch.load(X_train_path)
    y_train = torch.load(y_train_path)
    # f"Tipo de y_train: {type(y_train)}")

    # Si y_train es una lista (cada elemento es un tensor separado para cada categoría), debemos concatenarlos
    if isinstance(y_train, list):
        y_train_combined = torch.cat(y_train, dim=1)
    else:
        y_train_combined = y_train

    # Crea un DataLoader para los datos de entrenamiento
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train_combined)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = SimpleLinearCnnMO3(input_len=X_train.shape[1])

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    rmse_metric = RMSEMetric()

    # Crear la sesión de entrenamiento
    session = TrainingSession(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        rmse_metric=rmse_metric,
        train_loader=train_loader,
        epochs=10,
        output_folder=output_folder,
    )

    # Ejecuta el entrenamiento
    epochs_df = session.run_training()
    logger.debug("Entrenamiento completado.")

    # Guarda los artefactos del modelo
    session.save_model_artifacts(epochs_df=epochs_df)
