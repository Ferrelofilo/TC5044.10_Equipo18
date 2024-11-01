import pandas as pd
import sys

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


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


def preprocess_data(data_path):
    data_df = pd.read_csv(data_path)
    columns_to_encode = ["modified Zurich class", "largest spot size", "spot distribution"]
    pipeline = Pipeline(
        [("encode", make_column_transformer((OrdinalEncoder(), columns_to_encode), remainder="passthrough")),
         # ('scale', MinMaxScaler()),  Podemos ir agregando más transformaciones de ser necesario.
         ]
    )
    encoded_scaled = pipeline.fit_transform(data_df)
    data_df_processed = pd.DataFrame(encoded_scaled, index=data_df.index, columns=data_df.columns)

    X_train, X_test, y_train, y_test = split_data(data_df_processed)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    data_path = sys.argv[1]
    output_train_features = sys.argv[2]
    output_test_features = sys.argv[3]
    output_train_target = sys.argv[4]
    output_test_target = sys.argv[5]

    X_train, X_test, y_train, y_test = preprocess_data(data_path)
    pd.DataFrame(X_train).to_csv(output_train_features, index=False)
    pd.DataFrame(X_test).to_csv(output_test_features, index=False)
    pd.DataFrame(y_train).to_csv(output_train_target, index=False)
    pd.DataFrame(y_test).to_csv(output_test_target, index=False)
