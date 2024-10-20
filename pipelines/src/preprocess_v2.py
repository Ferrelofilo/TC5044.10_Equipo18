import pandas as pd
import sys

import torch

from pipelines.transformers.flare_column_transformer import get_flare_transformer
from pipelines.utils.data_utils import split_data

def preprocess_data(data_path):
    data_df = pd.read_csv(data_path)
    pipeline = get_flare_transformer()
    encoded = pipeline.fit_transform(data_df)
    data_df_processed = pd.DataFrame(encoded, index=data_df.index, columns=data_df.columns)

    X_train, X_test, y_train, y_test = split_data(data_df_processed)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    data_path = sys.argv[1]
    output_train_features = sys.argv[2]
    output_test_features = sys.argv[3]
    output_train_target = sys.argv[4]
    output_test_target = sys.argv[5]

    X_train, X_test, y_train, y_test = preprocess_data(data_path)
    torch.save(X_train, output_train_features)
    torch.save(X_test, output_test_features)
    torch.save(y_train, output_train_target)
    torch.save(y_test, output_test_target)
