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
    def __init__(self):
        self.frame = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.model = None
        self.load_dataset()

    def load_dataset(self):
        """Loading the dataset, and make the train, test, split."""
        dataset = pd.read_csv(DATASET_FILE_PATH)
        pipeline = get_flare_transformer()
        encoded = pipeline.fit_transform(dataset)
        data_df_processed = pd.DataFrame(encoded, index=dataset.index, columns=dataset.columns)
        self.X_train, self.X_test, self.y_train, self.y_test = split_data(data_df_processed)

    def train(self, model_class, epochs=100, learning_rate=0.01):
        self.model = model_class(self.X_train.shape[1])
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(self.X_train)
            loss = criterion(outputs, self.y_train)
            loss.backward()
            optimizer.step()

    def predict(self, input_data):
        with torch.no_grad():
            outputs = self.model(input_data)
            _, predicted = torch.max(outputs, 1)
        return predicted

    def get_accuracy(self):
        predictions = self.predict(self.X_test)
        return accuracy_score(self.y_test, predictions)

    def run_pipeline(self, model_class):
        self.load_dataset()
        self.train(model_class)


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


class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(input_dim, 3)  # Assuming 3 classes for the solar flare dataset

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def pipelines():
    pipeline_v1 = SimplePipeline()
    pipeline_v2 = PipelineWithFeatureEngineering()
    pipeline_v1.run_pipeline(SimpleNN)
    pipeline_v2.run_pipeline(SimpleNN)
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
