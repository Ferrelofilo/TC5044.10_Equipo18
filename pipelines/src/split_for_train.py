from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from preprocess import DataPreprocessor
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from utils.logger_setup import setup_logger

logger = setup_logger(__name__)


# Pytorch Dataset for Flare Data
class FlareDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[0][idx], self.y[1][idx], self.y[2][idx]


@dataclass
class DataSplitter:
    test_size: float = 0.2
    random_state: int = 42
    return_type: Literal["numpy", "dataloaders"] = "numpy"
    batch_size: int = 32

    def _split_target(self, data):
        """Split the target columns for flares."""
        if self.return_type == "numpy":
            return {
                "common flares": np.array(data.pop("common flares")),
                "moderate flares": np.array(data.pop("moderate flares")),
                "severe flares": np.array(data.pop("severe flares")),
            }
        if self.return_type == "dataloaders":
            return [
                np.array(data.pop("common flares")),
                np.array(data.pop("moderate flares")),
                np.array(data.pop("severe flares")),
            ]

    def split_data(self, data_df):
        """Split the data into train and test sets, returning based on specified return type."""
        logger.info("Splitting data into train and test sets...")

        X_train, X_test = train_test_split(
            data_df, test_size=self.test_size, random_state=self.random_state
        )

        # Separate the target columns
        y_train = self._split_target(X_train)
        y_test = self._split_target(X_test)

        if self.return_type == "numpy":
            return X_train, X_test, y_train, y_test
        elif self.return_type == "dataloaders":
            return self._return_as_dataloaders(X_train, X_test, y_train, y_test)
        else:
            raise ValueError(
                "Invalid return_type. Choose either 'numpy' or 'dataloaders'."
            )

    def _return_as_dataloaders(self, X_train, X_test, y_train, y_test):
        """Return the data as PyTorch DataLoaders."""
        logger.info("Returning data as PyTorch DataLoaders...")

        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        y_train_tensors = [
            torch.tensor(target, dtype=torch.float32).unsqueeze(1) for target in y_train
        ]
        y_test_tensors = [
            torch.tensor(target, dtype=torch.float32).unsqueeze(1) for target in y_test
        ]

        # Create datasets and dataloaders
        train_dataset = FlareDataset(X_train_tensor, y_train_tensors)
        test_dataset = FlareDataset(X_test_tensor, y_test_tensors)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )

        return train_loader, test_loader


if __name__ == "__main__":
    preprocess_df = DataPreprocessor().main(save_local_data=False)
    splitter_dataloaders = DataSplitter(return_type="dataloaders")
    splitter_numpy = DataSplitter(return_type="numpy")
    # Execution test
    train_loader, test_loader = splitter_dataloaders.split_data(preprocess_df)
    print("DataLoaders: ", train_loader, test_loader)
    X_train, X_test, y_train, y_test = splitter_numpy.split_data(preprocess_df)
    print("Numpy split: ", X_train.shape, X_test.shape, len(y_train), len(y_test))
