from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from preprocess import DataPreprocessor
from split_for_train import DataSplitter


# Model definition
@dataclass
class FlareModel:
    input_len: int = 10
    model: nn.Module = None
    optimizer: optim.Optimizer = None
    criterion: nn.Module = None
    rmse_metric: torchmetrics.Metric = None

    def __post_init__(self):
        self.model = SimpleMonnFlare3(self.input_len)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
        self.rmse_metric = torchmetrics.MeanSquaredError(squared=False)

    def train(self, dataloader, epochs=10):
        """Train the model with the provided dataloader."""
        self.model.train()

        for epoch in range(epochs):
            running_loss = 0.0

            for inputs, y1, y2, y3 in dataloader:
                self.optimizer.zero_grad()

                outputs_y1, outputs_y2, outputs_y3 = self.model(inputs)

                loss_y1 = self.criterion(outputs_y1, y1)
                loss_y2 = self.criterion(outputs_y2, y2)
                loss_y3 = self.criterion(outputs_y3, y3)

                loss = loss_y1 + loss_y2 + loss_y3

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            print(
                f"Epoch [{epoch+1}/{epochs}] - Average Loss: {running_loss/len(dataloader):.4f}"
            )


# Define the SimpleMonnFlare3 model
class SimpleMonnFlare3(nn.Module):
    def __init__(self, input_len=10):
        super(SimpleMonnFlare3, self).__init__()
        self.first_dense = nn.Linear(input_len, 64)
        self.second_dense = nn.Linear(64, 32)
        self.y1_output = nn.Linear(32, 1)
        self.y2_output = nn.Linear(32, 1)
        self.y3_output = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.first_dense(x))
        x = torch.relu(self.second_dense(x))
        y1 = self.y1_output(x)  # common flares
        y2 = self.y2_output(x)  # moderate flares
        y3 = self.y3_output(x)  # severe flares
        return y1, y2, y3


if __name__ == "__main__":
    # Example Usage
    preprocess_df = DataPreprocessor().main(save_local_data=False)

    # Split data using DataSplitter
    splitter = DataSplitter(return_type="dataloaders")
    train_loader, test_loader = splitter.split_data(preprocess_df)

    # Initialize and train the model using FlareModel
    flare_model = FlareModel(input_len=preprocess_df.shape[1] - 3)
    flare_model.train(train_loader)
