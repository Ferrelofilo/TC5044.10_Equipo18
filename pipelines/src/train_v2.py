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
from pipelines.models.simple_linear_cnn_multi_out_3 import SimpleLinearCnnMO3, train_model, ConvolutionalSimpleModel
from pipelines.utils.data_utils import create_dataloader
from pipelines.utils.mlflow_logging_utils import mlflow_epochs_logs, mlflow_torch_params


def load_params():
    with open("params.yaml", 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


params = load_params()


# Default settings for Linear CNN and Convolutional CNN
def create_cnn_model(cnn_type="linear_cnn"):
    model_params = params['models'][cnn_type]
    if cnn_type == 'linear_cnn':
        model = SimpleLinearCnnMO3(**model_params)
    elif model_type == 'convolutional_cnn':
        model = ConvolutionalSimpleModel(**model_params)

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    rmse_metric = torchmetrics.MeanSquaredError(squared=False)
    return model, optimizer, criterion, rmse_metric


def train_model_type(X_train_path, y_train_path, model_type):
    x = torch.load(X_train_path)
    y = torch.load(y_train_path)
    train_loader = create_dataloader(x, y, batch_size=32, shuffle=True)

    model, optimizer, criterion, rmse_metric = create_cnn_model(cnn_type=model_type)

    epoch_df = train_model(model, train_loader, optimizer, criterion, rmse_metric, epochs=10)

    mlflow_epochs_logs(epoch_df)

    ml_params = {"epochs": 10, "batch_size": 32, "shuffle": True}
    mlflow_torch_params(model, optimizer, additional_params=ml_params)

    return model


if __name__ == "__main__":
    X_train_path = sys.argv[1]
    y_train_path = sys.argv[2]
    model_type = sys.argv[3]
    run_id_out = sys.argv[4]  # Json Outfile
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
        model_trained = train_model_type(X_train_path, y_train_path, model_type)

        with open("model_summary.txt", "w") as f:
            f.write(str(summary(model_trained)))
        mlflow.log_artifact("model_summary.txt")
        mlflow.pytorch.log_model(model_trained, "model")

    torch.save(model_trained, model_path)
