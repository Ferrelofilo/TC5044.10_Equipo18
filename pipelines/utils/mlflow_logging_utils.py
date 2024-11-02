import mlflow
from torchinfo import summary

def mlflow_epochs_logs(epoch_df):
    for _, row in epoch_df.iterrows():
        step = int(row["epoch"])
        mlflow.log_metric("train_average_loss", row["average_loss"], step=step)
        mlflow.log_metric("train_rmse", row["rmse"], step=step)


def mlflow_torch_params(model, optimizer,criterion, additional_params=None):
    ml_params = {
                 "optimizer": type(optimizer).__name__,
                 "criterion": type(criterion).__name__,
                 "learning_rate": optimizer.param_groups[0]["lr"]}
    for key, value in optimizer.defaults.items():
        if f"optimizer_{key}" not in ml_params:
            ml_params[f"optimizer_{key}"] = value

    if additional_params:
        ml_params.update(additional_params)

    mlflow.log_params(ml_params)


def mlflow_model_log_summary(model):
    with open("model_summary.txt", "w",encoding='utf-8') as f:
        f.write(str(summary(model)))
    mlflow.log_artifact("model_summary.txt")
    mlflow.pytorch.log_model(model, "model")


def mlflow_evaluate_metrics(results_df, test_loss_avg):
    mlflow.log_metric("average_test_loss", test_loss_avg)
    for index, row in results_df.iterrows():
        category = row['Metric']
        mlflow.log_metric(f"{category}_RMSE_y1", row["Common Flares (y1)"])
        mlflow.log_metric(f"{category}_RMSE_y2", row["Moderate Flares (y2)"])
        mlflow.log_metric(f"{category}_RMSE_y3", row["Severe Flares (y3)"])

    results_df.to_csv("evaluation_metrics.csv", index=False)
    mlflow.log_artifact("evaluation_metrics.csv")

