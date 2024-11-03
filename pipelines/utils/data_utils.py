import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


def target_output(data):
    """
    Extrae las salidas objetivo del conjunto de datos, separando las etiquetas de flares comunes,
    moderados y severos.
    Args:
        data (pd.DataFrame): Conjunto de datos que contiene las columnas de flares.
    Returns:
        list: Lista de arrays de NumPy con las etiquetas para flares comunes, moderados y severos.
    """
    return [np.array(data.pop("common flares")),
            np.array(data.pop("moderate flares")),
            np.array(data.pop("severe flares"))]


def split_data(data_df, test_size=0.2, random_state=42):
    """
    Divide el conjunto de datos en entrenamiento y prueba, y convierte las salidas en tensores de PyTorch.
    Args:
        data_df (pd.DataFrame): Conjunto de datos con características y etiquetas.
        test_size (float, opcional): Proporción del conjunto de prueba. Default es 0.2.
        random_state (int, opcional): Semilla para reproducibilidad. Default es 42.
    Returns:
        tuple: Tensores de PyTorch para X_train, X_test y listas de tensores para y_train y y_test.
    """
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
    """
    Dataset personalizado para datos de flares. Proporciona características y múltiples salidas objetivo.
    Args:
        X (torch.Tensor): Tensor de características.
        y (list of torch.Tensor): Lista de tensores objetivo para cada tipo de flare.
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[0][idx], self.y[1][idx], self.y[2][idx]


def create_dataloader(x, y, shuffle=True, batch_size=32):
    """
    Crea un DataLoader de PyTorch para el conjunto de datos de flares.
    Args:
        x (torch.Tensor): Tensor de características.
        y (list of torch.Tensor): Lista de tensores objetivo para cada tipo de flare.
        shuffle (bool, opcional): Si se deben mezclar los datos en cada época. Default es True.
        batch_size (int, opcional): Tamaño del lote. Default es 32.
    Returns:
        DataLoader: DataLoader configurado con el conjunto de datos de flares.
    """
    flare_dataset = FlareDataset(x, y)

    dataloader = DataLoader(flare_dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader
