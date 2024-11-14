from pipelines.transformers import get_flare_transformer
from pipelines.utils import split_data

import os
import sys
import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Initialize logger
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
# from utils.logger_setup import setup_logger

# logger = setup_logger(__name__)


def preprocess_data(data_path):
    """
    Realiza el preprocesamiento de los datos cargándolos desde un archivo CSV,
    aplicando transformaciones y dividiéndolos en conjuntos de entrenamiento y prueba.
    Args:
        data_path (str): Ruta al archivo CSV con los datos originales.
    Returns:
        tuple: Contiene cuatro elementos:
            X_train (pd.DataFrame): Características para el conjunto de entrenamiento.
            X_test (pd.DataFrame): Características para el conjunto de prueba.
            y_train (pd.Series o pd.DataFrame): Objetivo para el conjunto de entrenamiento.
            y_test (pd.Series o pd.DataFrame): Objetivo para el conjunto de prueba.
    """
    data_df = pd.read_csv(data_path)
    pipeline = get_flare_transformer()
    encoded = pipeline.fit_transform(data_df)
    data_df_processed = pd.DataFrame(encoded, index=data_df.index, columns=data_df.columns)

    X_train, X_test, y_train, y_test = split_data(data_df_processed)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

    # logger.debug("Initialize preprocesss")
    data_path = sys.argv[1]
    output_train_features = sys.argv[2]
    output_test_features = sys.argv[3]
    output_train_target = sys.argv[4]
    output_test_target = sys.argv[5]

    X_train, X_test, y_train, y_test = preprocess_data(data_path)
    # logger.debug("Saving X_train, X_test, y_train, y_test")
    torch.save(X_train, output_train_features)
    torch.save(X_test, output_test_features)
    torch.save(y_train, output_train_target)
    torch.save(y_test, output_test_target)
