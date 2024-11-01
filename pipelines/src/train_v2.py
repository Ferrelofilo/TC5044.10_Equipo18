import json
import sys

import torchmetrics
import yaml
import sys
import pandas as pd
import torch
import joblib
import mlflow
from torch import optim, nn
from torchinfo import summary

from pipelines.models.cnn_handler import MultiOutCnnHandler
from pipelines.models.simple_linear_cnn_multi_out_3 import SimpleLinearCnnMO3, train_model, ConvolutionalSimpleModel
from pipelines.utils.data_utils import create_dataloader
from pipelines.utils.mlflow_logging_utils import mlflow_epochs_logs, mlflow_torch_params, mlflow_model_log_summary


def load_params():
    with open("params.yaml", 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


params = load_params()


def cnn_model_train(X_train_path, y_train_path, model_type):
    x = torch.load(X_train_path)
    y = torch.load(y_train_path)
    train_loader = create_dataloader(x, y, batch_size=32, shuffle=True)
    model_params = params['models'][model_type]

    mo_cnn_handler = MultiOutCnnHandler(cnn_type=model_type,model_params=model_params)

    epoch_df = mo_cnn_handler.train_model(train_loader,epochs=10)

    mlflow_epochs_logs(epoch_df)

    ml_params = {"epochs": 10, "batch_size": 32, "shuffle": True}
    mlflow_torch_params(mo_cnn_handler.model, mo_cnn_handler.optimizer, additional_params=ml_params)

    return mo_cnn_handler


if __name__ == "__main__":
    X_train_path = sys.argv[1]
    y_train_path = sys.argv[2]
    model_type = sys.argv[3]
    run_id_out = params['mlflow']['runs']  # path to save run id json file
    model_dir = params['data']['models']  # params model location
    model_path = f"{model_dir}/{model_type}_model.pth"  # model outfile

    mlflow.set_experiment(params['mlflow']['experiment_name'])
    #mlflow.set_tracking_uri(params['mlflow']['tracking_uri'])

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        with open(run_id_out, "w") as f:
            json.dump({model_type: run_id}, f)

        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("phase", "training")
        cnn = cnn_model_train(X_train_path, y_train_path, model_type)

        mlflow_model_log_summary(cnn.model)

        cnn.save_model(model_path)

