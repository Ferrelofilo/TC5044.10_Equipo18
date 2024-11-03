import pandas as pd
from unittest.mock import Mock, patch, call
import torch.nn as nn
import torch.optim as optim

from pipelines.utils.mlflow_logging_utils import mlflow_epochs_logs, mlflow_torch_params, mlflow_model_log_summary, mlflow_evaluate_metrics


@patch("mlflow.log_metric", return_value=Mock())
def test_mlflow_epochs_logs(mock_log_metric):
    # Mock DataFrame to simulate epoch data
    epoch_df = pd.DataFrame({
        "epoch": [1, 2],
        "average_loss": [0.1, 0.05],
        "rmse": [0.2, 0.1]
    })

    mlflow_epochs_logs(epoch_df)

    # Verify calls to mlflow.log_metric for each row
    calls = [
        call("train_average_loss", 0.1, step=1),
        call("train_rmse", 0.2, step=1),
        call("train_average_loss", 0.05, step=2),
        call("train_rmse", 0.1, step=2)
    ]
    mock_log_metric.assert_has_calls(calls, any_order=False)


@patch("mlflow.log_params", return_value=Mock())
def test_mlflow_torch_params(mock_log_params):
    # Mock model and optimizer for the test
    model = nn.Linear(10, 2)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    additional_params = {"batch_size": 32}

    mlflow_torch_params(model, optimizer, criterion, additional_params)

    # Extract the actual parameters passed to mlflow.log_params
    actual_params = mock_log_params.call_args[0][0]

    # Expected parameters that should be logged
    expected_params = {
        "optimizer": "SGD",
        "criterion": "MSELoss",
        "learning_rate": 0.01,
        "batch_size": 32,
        "optimizer_momentum": optimizer.defaults.get("momentum", 0)
    }

    # Check that all expected parameters are in the actual parameters
    for key, value in expected_params.items():
        assert actual_params[key] == value


@patch("mlflow.log_artifact", return_value=Mock())
@patch("mlflow.pytorch.log_model", return_value=Mock())
def test_mlflow_model_log_summary(mock_log_model, mock_log_artifact):
    # Mock model for the test
    model = nn.Linear(10, 2)

    mlflow_model_log_summary(model)
    # Verify that summary was called and the artifact was logged
    mock_log_artifact.assert_called_once_with("model_summary.txt")
    mock_log_model.assert_called_once_with(model, "model")


@patch("mlflow.log_metric", return_value=Mock())
@patch("mlflow.log_artifact", return_value=Mock())
def test_mlflow_evaluate_metrics(mock_log_artifact, mock_log_metric):
    # Mock DataFrame to simulate results data
    results_df = pd.DataFrame({
        "Metric": ["RMSE", "MAE", "R²"],
        "Common Flares (y1)": [0.2, 0.1, 0.9],
        "Moderate Flares (y2)": [0.25, 0.15, 0.85],
        "Severe Flares (y3)": [0.3, 0.2, 0.8]
    })
    test_loss_avg = 0.15

    mlflow_evaluate_metrics(results_df, test_loss_avg)

    # Verify calls to log_metric for test loss and evaluation metrics
    mock_log_metric.assert_any_call("average_test_loss", test_loss_avg)
    metric_calls = [
        call("eval_RMSE_y1", 0.2),
        call("eval_RMSE_y2", 0.25),
        call("eval_RMSE_y3", 0.3),
        call("eval_MAE_y1", 0.1),
        call("eval_MAE_y2", 0.15),
        call("eval_MAE_y3", 0.2),
        call("eval_R²_y1", 0.9),
        call("eval_R²_y2", 0.85),
        call("eval_R²_y3", 0.8)
    ]
    mock_log_metric.assert_has_calls(metric_calls, any_order=True)

    # Verify that the artifact for evaluation metrics is logged
    mock_log_artifact.assert_called_once_with("evaluation_metrics.csv")
