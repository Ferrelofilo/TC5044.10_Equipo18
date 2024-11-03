import torch
from pipelines.models import MultiOutCnnHandler
import pytest
import os

# Get the absolute path of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Navigate up to the project root directory
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../"))

DATASET_FILE_PATH = os.path.join(BASE_DIR, "data", "raw", "flare_data2_df.csv")


def test_create_model_instance():
    model_params = {
        "input_len": 10,
        "out_features1": 64,
        "out_features2": 32,
        "bias": True
    }
    cnn_type = "linear_cnn"
    handler = MultiOutCnnHandler(cnn_type=cnn_type, model_params=model_params)
    assert handler.model is not None
    assert handler.model_params == model_params
    assert handler.cnn_type == cnn_type


def test_train_model_metrics():
    model_params = {
        "input_len": 10,
        "out_features1": 64,
        "out_features2": 32,
        "bias": True
    }
    cnn_type = "linear_cnn"
    handler = MultiOutCnnHandler(cnn_type=cnn_type, model_params=model_params)

    # Mock DataLoader
    train_loader = [(torch.randn(32, 10), torch.randn(32, 1), torch.randn(32, 1), torch.randn(32, 1)) for _ in range(10)]
    epochs = 2

    epoch_df = handler.train_model(train_loader, epochs=epochs)
    assert 'average_loss' in epoch_df.columns
    assert 'rmse' in epoch_df.columns
    assert len(epoch_df) == epochs


def test_evaluate_multi_output_metrics():
    model_params = {
        "input_len": 10,
        "out_features1": 64,
        "out_features2": 32,
        "bias": True
    }
    cnn_type = "linear_cnn"
    handler = MultiOutCnnHandler(cnn_type=cnn_type, model_params=model_params)

    # Mock DataLoader
    test_loader = [(torch.randn(32, 10), torch.randn(32, 1), torch.randn(32, 1), torch.randn(32, 1)) for _ in range(10)]

    results_df, total_test_loss = handler.evaluate_multi_output_metrics(test_loader, handler.criterion)
    assert 'Common Flares (y1)' in results_df.columns
    assert 'Moderate Flares (y2)' in results_df.columns
    assert 'Severe Flares (y3)' in results_df.columns
    assert len(total_test_loss) == len(test_loader)


@pytest.mark.parametrize("model_params", [
    {"input_len": 10, "out_features1": 64, "out_features2": 32, "bias": True},
    {"input_len": 10, "out_features1": 128, "out_features2": 64, "bias": False}
])
def test_metrics_compared_to_different_params(model_params):
    cnn_type = "linear_cnn"
    handler = MultiOutCnnHandler(cnn_type=cnn_type, model_params=model_params)

    # Mock DataLoader
    train_loader = [(torch.randn(32, 10), torch.randn(32, 1), torch.randn(32, 1), torch.randn(32, 1)) for _ in range(10)]
    test_loader = [(torch.randn(32, 10), torch.randn(32, 1), torch.randn(32, 1), torch.randn(32, 1)) for _ in range(10)]

    # Train and evaluate
    handler.train_model(train_loader, epochs=2)
    results_df, total_test_loss = handler.evaluate_multi_output_metrics(test_loader, handler.criterion)

    assert 'Common Flares (y1)' in results_df.columns
    assert 'Moderate Flares (y2)' in results_df.columns
    assert 'Severe Flares (y3)' in results_df.columns
    assert len(total_test_loss) == len(test_loader)
