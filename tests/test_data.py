from pipelines.src.load_data import load_data
from pipelines.src.preprocess_v2 import preprocess_data
from pipelines.utils import create_dataloader

import pandas as pd
import torch
import os

# Get the absolute path of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Navigate up to the project root directory
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../"))

DATASET_FILE_PATH = os.path.join(BASE_DIR, "data", "raw", "flare_data2_df.csv")


# Prueba para la función load_data usando un archivo real
def test_load_data(file_path=DATASET_FILE_PATH):
    # Llama a la función y realiza las verificaciones
    result = load_data(file_path)
    assert result is not None, "Se esperaba un resultado no nulo para load_data"
    assert isinstance(
        result, pd.DataFrame
    ), "Se esperaba que el resultado fuera un DataFrame"
    assert not result.empty, "Se esperaba que el DataFrame no estuviera vacío"


# Prueba para la función preprocess_data usando un archivo real
def test_preprocess_data(file_path=DATASET_FILE_PATH):
    # Llama a la función y realiza las verificaciones
    result = preprocess_data(file_path)
    assert result is not None, "Se esperaba un resultado no nulo para preprocess_data"
    assert len(result) == 4, "Se esperaba que el resultado contuviera 4 elementos"
    assert isinstance(
        result[0], torch.Tensor
    ), "Se esperaba que X_train fuera un torch.Tensor"
    assert isinstance(
        result[1], torch.Tensor
    ), "Se esperaba que X_test fuera un torch.Tensor"
    assert isinstance(
        result[2][0], torch.Tensor
    ), "Se esperaba que y_train fuera un torch.Tensor"
    assert isinstance(
        result[3][0], torch.Tensor
    ), "Se esperaba que y_test fuera un torch.Tensor"
    assert len(result[0]) > 0, "Se esperaba que X_train no estuviera vacío"
    assert len(result[1]) > 0, "Se esperaba que X_test no estuviera vacío"
    # Testing
    assert (
        len(result[2]) == 3
    ), "Se esperaba que y_train tenga 3 tensores por cada target"
    assert (
        len(result[3]) == 3
    ), "Se esperaba que y_test tenga 3 tensores por cada target"

    X_train, X_test, y_train, y_test = result
    data_loader = create_dataloader(x=X_train, y=y_train)
    assert (
        type(data_loader.dataset).__name__ == "FlareDataset"
    ), "Se espera que el tipo de dataset herede de FlareDataset"


if __name__ == "__main__":
    test_load_data()
    test_preprocess_data()
