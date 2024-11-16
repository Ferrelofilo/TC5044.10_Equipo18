import torch.optim as optim
import torch.nn as nn
import torchmetrics
from pipelines.models import MultiOutCnnHandler
import pytest

@pytest.mark.parametrize("optimizer_class, optimizer_params", [
    (optim.SGD, {"lr": 0.01}),
    (optim.Adam, {"lr": 0.001})
])
@pytest.mark.parametrize("criterion", [
    nn.MSELoss(),
    nn.L1Loss()
])
@pytest.mark.parametrize("rmse_metric", [
    torchmetrics.MeanSquaredError(squared=False),
    torchmetrics.MeanAbsoluteError()
])
def test_model_settings(optimizer_class, optimizer_params, criterion, rmse_metric):
    model_params = {"input_len": 10, "out_features1": 64, "out_features2": 32, "bias": True}
    cnn_type = "linear_cnn"

    # Initialize handler with different configurations
    handler = MultiOutCnnHandler(
        cnn_type=cnn_type,
        model_params=model_params,
        optimizer_params=optimizer_params,
        criterion=criterion
    )
    handler.optimizer = optimizer_class(handler.model.parameters(), **optimizer_params)
    handler.rmse_metric = rmse_metric

    # Assertions for model settings
    assert handler.cnn_type == cnn_type
    assert handler.model_params == model_params

    # Check optimizer settings
    assert isinstance(handler.optimizer, optimizer_class)
    for param, value in optimizer_params.items():
        assert handler.optimizer.param_groups[0][param] == value

    # Check criterion
    assert isinstance(handler.criterion, criterion.__class__)

    # Check rmse_metric
    assert isinstance(handler.rmse_metric, rmse_metric.__class__)

    print(f"Test passed with {optimizer_class.__name__}, {criterion.__class__.__name__}, and {rmse_metric.__class__.__name__}")

