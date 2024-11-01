import json
import sys

import torchmetrics
import yaml
import sys
import pandas as pd
import torch
import joblib
import mlflow
from tensorboard import summary
from torch import optim, nn

from pipelines.models.simple_linear_cnn_multi_out_3 import SimpleLinearCnnMO3, train_model
from pipelines.utils.data_utils import create_dataloader
def load_params():
    with open("params.yaml", 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


params = load_params()


def create_model(input_len=9):
    model = SimpleLinearCnnMO3(input_len=input_len)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    rmse_metric = torchmetrics.MeanSquaredError(squared=False)
    return model, optimizer, criterion, rmse_metric


def train_model_v2(X_train_path, y_train_path, model_type):
    x = torch.load(X_train_path)
    y = torch.load(y_train_path)

    train_loader = create_dataloader(x,y,batch_size=32, shuffle=True)
    model, optimizer, criterion, rmse_metric = create_model(input_len=x.shape[1])

    epoch_df = train_model(model, train_loader, optimizer, criterion, rmse_metric, epochs=10)

    for _, row in epoch_df.iterrows():
        step = row["epoch"]
        mlflow.log_metric("train_average_loss", row["average_loss"], step=step)
        mlflow.log_metric("train_rmse", row["rmse"], step=step)

    return model


if __name__ == "__main__":
    X_train_path = sys.argv[1]
    y_train_path = sys.argv[2]
    model_type = sys.argv[3]
    run_id_out = sys.argv[4]
    model_dir = params['data']['models']
    model_path = f"{model_dir}/{model_type}_model.pth"

    mlflow.set_experiment(params['mlflow']['experiment_name'])
    mlflow.set_tracking_uri(params['mlflow']['tracking_uri'])

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        with open(run_id_out, "w") as f:
            json.dump({"run_id": run_id}, f)

        mlflow.set_tag("phase", "training")
        model_trained = train_model(X_train_path, y_train_path)
        with open("model_summary.txt", "w") as f:
            f.write(str(summary(model_trained)))
        mlflow.log_artifact("model_summary.txt")
        mlflow.pytorch.log_model(model_trained, "model")

    torch.save(model_trained, model_path)
