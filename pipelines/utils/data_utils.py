
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

def target_output(data):
    return [np.array(data.pop("common flares")), np.array(data.pop("moderate flares")),
            np.array(data.pop("severe flares"))]


def split_data(data_df, test_size=0.2, random_state=42):
    X_train, X_test = train_test_split(data_df, test_size=test_size, random_state=random_state)

    y_train = target_output(X_train)
    y_test = target_output(X_test)

    # PyTorch tensors
    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    y_train = [torch.tensor(target, dtype=torch.float32).unsqueeze(1) for target in y_train]
    y_test = [torch.tensor(target, dtype=torch.float32).unsqueeze(1) for target in y_test]

    return X_train, X_test, y_train, y_test


class FlareDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[0][idx], self.y[1][idx], self.y[2][idx]


def create_dataloader(x, y, shuffle=True, batch_size=32):
    flare_dataset = FlareDataset(x, y)

    dataloader = DataLoader(flare_dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader