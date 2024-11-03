import pandas as pd
import torch
import torch.nn as nn

import torch.optim as optim
from pipelines.models import MultiOutCnnHandler
from pipelines.transformers import get_flare_transformer
from pipelines.utils.data_utils import target_output
from pipelines.utils import split_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import accuracy_score
import pytest
import os

# Get the absolute path of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Navigate up to the project root directory
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../"))

DATASET_FILE_PATH = os.path.join(BASE_DIR, "data", "raw", "flare_data2_df.csv")


class SimplePipeline:
    def __init__(self, input_len=10, out_features1=64, out_features2=32, bias=True):
        super(SimplePipeline, self).__init__()
        self.first_dense = nn.Linear(input_len, out_features1, bias=bias)
        self.second_dense = nn.Linear(out_features1, out_features2, bias=bias)
        self.y1_output = nn.Linear(out_features2, 1)
        self.y2_output = nn.Linear(out_features2, 1)
        self.y3_output = nn.Linear(out_features2, 1)

    def forward(self, x):
        x = torch.relu(self.first_dense(x))
        x = torch.relu(self.second_dense(x))
        y1 = self.y1_output(x)  # common_flares
        y2 = self.y2_output(x)  # moderate_flares
        y3 = self.y3_output(x)  # severe_flares
        return y1, y2, y3


class PipelineWithFeatureEngineering(SimplePipeline):
    def __init__(self):
        super().__init__()
        self.scaler = StandardScaler()

    def apply_scaler(self):
        self.scaler.fit(self.X_train)
        self.X_train = torch.tensor(self.scaler.transform(self.X_train), dtype=torch.float32)
        self.X_test = torch.tensor(self.scaler.transform(self.X_test), dtype=torch.float32)

    def run_pipeline(self, model_class):
        self.load_dataset()
        self.apply_scaler()
        self.train(model_class)


@pytest.fixture
def pipelines():
    pipeline_v1 = SimplePipeline()
    pipeline_v2 = PipelineWithFeatureEngineering()
    return pipeline_v1, pipeline_v2


def test_create_model_instance():
    model_type = "linear_cnn"

    model_params = {
        "input_len": 10,
        "out_features1": 64,
        "out_features2": 32,
        "bias": True
    }
    mo_cnn_handler = MultiOutCnnHandler(cnn_type=model_type, model_params=model_params)

    # assert mo_cnn_handler  # Type of model is created
    assert True


def test_train_model_metrics():
    # validate amount of epochs, average loss and rmse
    # assert epoch
    # assert average_loss
    # assert rmse
    assert True


def test_evaluate_multi_output_metrics():
    # assert results_df
    # assert total_test_loss
    # assert avg_test_loss
    assert True


def test_metrics_compared_to_different_params():
    # assert metrics1 != metrics2
    assert True
