from pipelines.src.load_data import load_data
import pandas as pd
import os

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import pytest

# Get the absolute path of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Navigate up to the project root directory
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../"))

DATASET_FILE_PATH = os.path.join(BASE_DIR, "data", "raw", "flare_data2_df.csv")

# Getting the data
iris = datasets.load_iris()

# Simple setup in the data
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target


class SimplePipeline:
    def __init__(self):
        self.frame = None
        # Each value is None when we instantiate the class
        self.X_train, self.X_test, self.y_train, self.Y_test = None, None, None, None
        self.model = None
        self.load_dataset()

    def load_dataset(self):
        """Loading the dataset, and make the train, test, split."""
        dataset = datasets.load_iris()

        # Removing the units (cm) from the headers
        self.feature_names = [fn[:-5] for fn in dataset.feature_names]
        self.frame = pd.DataFrame(dataset.data, columns=self.feature_names)
        self.frame['target'] = dataset.target

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.frame[self.feature_names], self.frame.target, test_size=0.65, random_state=42)

    def train(self, algorithm=LogisticRegression):

        self.model = algorithm(solver='lbfgs', multi_class='auto')
        self.model.fit(self.X_train, self.y_train)

    def predict(self, input_data):
        return self.model.predict(input_data)

    def get_accuracy(self):
        return self.model.score(X=self.X_test, y=self.y_test)

    def run_pipeline(self):
        """Execution method for running the pipeline several times."""
        self.load_dataset()
        self.train()


class PipelineWithFeatureEngineering(SimplePipeline):
    def __init__(self):
        # Calling the inherit method SimplePipeline __init__ first.
        super().__init__()

        # Standardizing the variables in the dataset.
        self.scaler = StandardScaler()

    def apply_scaler(self):
        # Fitting the scaler on the training data and then applying it to both training and testing data.
        self.scaler.fit(self.X_train)
        self.X_train = self.scaler.transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def predict(self, input_data):
        # Applying the scaler before making the predictions.
        scaled_input_data = self.scaler.transform(input_data)
        return self.model.predict(scaled_input_data)

    def run_pipeline(self):
        self.load_dataset()
        self.apply_scaler()
        self.train()


@pytest.fixture
def pipelines():
    pipeline_v1 = SimplePipeline()
    pipeline_v2 = PipelineWithFeatureEngineering()
    pipeline_v1.run_pipeline()
    pipeline_v2.run_pipeline()
    return pipeline_v1, pipeline_v2


def test_accuracy_higher_than_benchmark(pipelines):
    pipeline_v1, _ = pipelines

    # Initial Benchmark
    benchmark_predictions = [1.0] * len(pipeline_v1.y_test)
    benchmark_accuracy = accuracy_score(y_true=pipeline_v1.y_test, y_pred=benchmark_predictions)

    # Getting the accuracy of the model
    predictions = pipeline_v1.predict(pipeline_v1.X_test)
    actual_accuracy = accuracy_score(y_true=pipeline_v1.y_test, y_pred=predictions)

    print(f'Accuracy of model 1: {actual_accuracy}, Accuracy of Benchmark: {benchmark_accuracy}')

    # Comparing the accuracy of the first model against the benchmark
    assert actual_accuracy > benchmark_accuracy


def test_accuracy_compared_to_previous_version(pipelines):
    pipeline_v1, pipeline_v2 = pipelines

    # Getting the accuracy of each version
    v1_accuracy = pipeline_v1.get_accuracy()
    v2_accuracy = pipeline_v2.get_accuracy()

    print(f'Accuracy of model 1: {v1_accuracy}')
    print(f'Accuracy of model 2: {v2_accuracy}')

    # Comparing the accuracy of the second model against the first one
    assert v2_accuracy >= v1_accuracy
