from pipelines.models import MultiOutCnnHandler
from pipelines.utils import (
    create_dataloader,
    mlflow_epochs_logs,
    mlflow_torch_params,
    mlflow_model_log_summary
)

import json
import sys
import os
import yaml
import torch
import mlflow

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


def load_params():
    """
    Carga los parámetros de configuración desde un archivo YAML.
    Returns:
        dict: Diccionario con la configuración cargada desde "params.yaml".
    """
    with open("params.yaml", "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


params = load_params()


def cnn_model_train(X_train_path, y_train_path, model_type):
    """
    Entrena un modelo CNN utilizando los datos de entrenamiento y el tipo de modelo especificado.
    Args:
        X_train_path (str): Ruta al archivo con los datos de características de entrenamiento.
        y_train_path (str): Ruta al archivo con los datos de etiquetas de entrenamiento.
        model_type (str): Tipo de modelo CNN a utilizar en el entrenamiento.
    Returns:
        MultiOutCnnHandler: Objeto manejador de CNN entrenado con los datos.
    """
    x = torch.load(X_train_path)
    y = torch.load(y_train_path)
    train_loader = create_dataloader(x, y, batch_size=32, shuffle=True)
    model_params = params["models"][model_type]

    mo_cnn_handler = MultiOutCnnHandler(cnn_type=model_type, model_params=model_params)

    epoch_df = mo_cnn_handler.train_model(train_loader, epochs=10)

    mlflow_epochs_logs(epoch_df)

    ml_params = {"epochs": 10, "batch_size": 32, "shuffle": True}
    mlflow_torch_params(mo_cnn_handler.model, mo_cnn_handler.optimizer, mo_cnn_handler.criterion, additional_params=ml_params)

    return mo_cnn_handler


if __name__ == "__main__":
    X_train_path = sys.argv[1]
    y_train_path = sys.argv[2]
    model_type = sys.argv[3]
    run_id_out = params["mlflow"]["runs"]  # path to save run id json file
    model_dir = params["data"]["models"]  # params model location
    model_path = f"{model_dir}/{model_type}_model.pth"  # model outfile

    mlflow.set_experiment(params["mlflow"]["experiment_name"])
    # mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        with open(run_id_out, "w") as f:
            json.dump({model_type: run_id}, f)

        mlflow.set_tag("model_type", model_type)
        cnn = cnn_model_train(X_train_path, y_train_path, model_type)

        mlflow_model_log_summary(cnn.model)

        cnn.save_model(model_path)
