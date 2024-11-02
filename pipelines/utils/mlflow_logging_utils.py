import mlflow
from torchinfo import summary

from pipelines.src.evaluate_v2 import test_loader


def mlflow_epochs_logs(epoch_df):
    """
    Registra métricas de entrenamiento en MLflow para cada época del entrenamiento.

    Args:
        epoch_df (pd.DataFrame): DataFrame que contiene las métricas de cada época, incluyendo 'average_loss' y 'rmse'.
    """
    for _, row in epoch_df.iterrows():
        step = row["epoch"]
        mlflow.log_metric("train_average_loss", row["average_loss"], step=step)
        mlflow.log_metric("train_rmse", row["rmse"], step=step)


def mlflow_torch_params(model, optimizer, additional_params=None):
    """
    Registra los parámetros del modelo y del optimizador en MLflow, incluyendo el tipo de optimizador,
    criterio de pérdida y tasa de aprendizaje.

    Args:
        model (torch.nn.Module): Modelo de PyTorch.
        optimizer (torch.optim.Optimizer): Optimizador utilizado en el modelo.
        additional_params (dict, opcional): Parámetros adicionales que se desean registrar.
    """
    ml_params = {
        "optimizer": type(model.optimizer).__name__,
        "criterion": type(model.criterion).__name__,
        "learning_rate": model.optimizer.param_groups[0]["lr"],
    }
    for key, value in optimizer.defaults.items():
        if f"optimizer_{key}" not in ml_params:
            ml_params[f"optimizer_{key}"] = value

    if additional_params:
        ml_params.update(additional_params)

    mlflow.log_params(ml_params)


def mlflow_model_log_summary(model):
    """
    Guarda y registra en MLflow un resumen del modelo, incluyendo su estructura y parámetros.

    Args:
        model (torch.nn.Module): Modelo de PyTorch que se desea registrar y resumir.
    """
    with open("model_summary.txt", "w") as f:
        f.write(str(summary(model)))
    mlflow.log_artifact("model_summary.txt")
    mlflow.pytorch.log_model(model, "model")


def mlflow_evaluate_metrics(results_df, test_loss_avg):
    """
    Registra métricas de evaluación en MLflow, incluyendo pérdida promedio de prueba y RMSE para cada salida.

    Args:
        results_df (pd.DataFrame): DataFrame que contiene las métricas de evaluación para cada categoría de flare.
        test_loss_avg (float): Promedio de pérdida de prueba.
    """
    mlflow.log_metric("average_test_loss", test_loss_avg)
    for index, row in results_df.iterrows():
        category = row["Metric"]
        mlflow.log_metric(f"{category}_RMSE_y1", row["Common Flares (y1)"])
        mlflow.log_metric(f"{category}_RMSE_y2", row["Moderate Flares (y2)"])
        mlflow.log_metric(f"{category}_RMSE_y3", row["Severe Flares (y3)"])

    results_df.to_csv("evaluation_metrics.csv", index=False)
    mlflow.log_artifact("evaluation_metrics.csv")
