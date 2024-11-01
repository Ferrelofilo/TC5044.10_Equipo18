import mlflow


def mlflow_epochs_logs(epoch_df):
    for _, row in epoch_df.iterrows():
        step = row["epoch"]
        mlflow.log_metric("train_average_loss", row["average_loss"], step=step)
        mlflow.log_metric("train_rmse", row["rmse"], step=step)


def mlflow_torch_params(model, optimizer, additional_params=None):
    ml_params = {
                 "optimizer": type(model.optimizer).__name__,
                 "criterion": type(model.criterion).__name__,
                 "learning_rate": model.optimizer.param_groups[0]["lr"]}
    for key, value in optimizer.defaults.items():
        if f"optimizer_{key}" not in ml_params:
            ml_params[f"optimizer_{key}"] = value

    if additional_params:
        ml_params.update(additional_params)

    mlflow.log_params(ml_params)