import os
import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import json

# Initialize logger
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from utils.logger_setup import setup_logger

logger = setup_logger(__name__)


@dataclass
class ModelEvaluator:
    """Clase que encapsula el proceso de evaluación del modelo."""

    model: torch.nn.Module
    dataloader: DataLoader
    criterion: torch.nn.Module
    rmse_metric: any
    output_folder: str
    model_path: str = "models/flare_model.pth"  # Ruta donde se encuentra el modelo entrenado

    def load_model(self):
        """Carga los pesos del modelo desde un archivo."""
        self.model.load_state_dict(torch.load(self.model_path))
        logger.debug(f"Modelo cargado desde {self.model_path}")

    def evaluate(self):
        """Realiza la evaluación del modelo en los datos de prueba."""
        self.model.eval()
        total_loss = 0.0
        self.rmse_metric.reset()

        with torch.no_grad():
            for batch, y_test_combined in self.dataloader:
                # Separar las etiquetas de salida en tres columnas
                y1, y2, y3 = (
                    y_test_combined[:, 0:1],
                    y_test_combined[:, 1:2],
                    y_test_combined[:, 2:3],
                )

                outputs_y1, outputs_y2, outputs_y3 = self.model(batch)

                loss_y1 = self.criterion(outputs_y1, y1)
                loss_y2 = self.criterion(outputs_y2, y2)
                loss_y3 = self.criterion(outputs_y3, y3)

                total_loss += (loss_y1 + loss_y2 + loss_y3).item()

                self.rmse_metric.update(outputs_y1, y1, index=0)
                self.rmse_metric.update(outputs_y2, y2, index=1)
                self.rmse_metric.update(outputs_y3, y3, index=2)
        avg_loss = total_loss / len(self.dataloader)
        rmse = np.mean(self.rmse_metric.compute())
        logger.debug(f"Evaluación completa - Pérdida: {avg_loss:.4f}, RMSE: {rmse:.4f}")
        return avg_loss, rmse

    def save_evaluation_metrics(self, avg_loss, rmse):
        """Guarda las métricas de evaluación en un archivo CSV y JSON."""
        os.makedirs(self.output_folder, exist_ok=True)

        # Guardar las métricas en un archivo CSV
        csv_path = os.path.join(self.output_folder, "evaluation_metrics.csv")
        df = pd.DataFrame({"average_loss": [avg_loss], "rmse": [rmse]})
        df.to_csv(csv_path, index=False)
        logger.debug(f"Métricas de evaluación guardadas en {csv_path}")

        # Guardar las métricas en un archivo JSON
        json_path = os.path.join(self.output_folder, "evaluation_metrics.json")
        evaluation_metrics = {"average_loss": str(avg_loss), "rmse": str(rmse)}
        with open(json_path, "w") as json_file:
            json.dump(evaluation_metrics, json_file, indent=4)
        logger.debug(f"Métricas de evaluación guardadas en {json_path}")


if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
    from pipelines.models.simple_linear_cnn_multi_out_3 import SimpleLinearCnnMO3
    from pipelines.utils.rmse_metric import RMSEMetric

    # Obtiene los argumentos de la línea de comandos
    X_test_path = sys.argv[1]
    y_test_path = sys.argv[2]
    model_path = sys.argv[3]
    batch_size = int(sys.argv[4])
    eval_folder_path = sys.argv[5]

    # Carga los datos de prueba
    X_test = torch.load(X_test_path)
    y_test = torch.load(y_test_path)

    # Si y_test es una lista (cada elemento es un tensor separado para cada categoría), debemos concatenarlos
    if isinstance(y_test, list):
        y_test_combined = torch.cat(y_test, dim=1)
    else:
        y_test_combined = y_test

    # Prepara el DataLoader para los datos de prueba
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test_combined)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Inicializa el modelo y la métrica RMSE
    model = SimpleLinearCnnMO3(input_len=X_test.shape[1])
    criterion = torch.nn.MSELoss()
    rmse_metric = RMSEMetric()
    # Crea la sesión de evaluación
    evaluator = ModelEvaluator(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        rmse_metric=rmse_metric,
        model_path=model_path,
        output_folder=eval_folder_path,
    )

    # Carga el modelo y realiza la evaluación
    evaluator.load_model()
    avg_loss, rmse = evaluator.evaluate()

    # Guarda los resultados de la evaluación
    evaluator.save_evaluation_metrics(avg_loss, rmse)
