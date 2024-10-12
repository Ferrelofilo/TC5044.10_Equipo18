import os
from dataclasses import dataclass

import pandas as pd
from etl import CreateDF, DataIntoLocalFile
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

from utils.logger_setup import setup_logger

logger = setup_logger(__name__)

DEFAULT_COLUMN_NAMES = (
    "modified Zurich class",
    "largest spot size",
    "spot distribution",
    "activity",
    "evolution",
    "previous 24 hour flare activity",
    "historically-complex",
    "became complex on this pass",
    "area",
    "common flares",
    "moderate flares",
    "severe flares",
)


@dataclass
class DataPreprocessor:
    save_folder_path: str = "../data/preprocess"
    column_names: tuple[str] = DEFAULT_COLUMN_NAMES

    def load_data(self):
        """Load the raw data from a file."""
        logger.info(f"Loading data")
        try:
            data_local = DataIntoLocalFile()
            saved_path = data_local.download_and_extract()

            df_class = CreateDF(saved_path)
            df = df_class.create_dataframe()
            logger.info("Data loaded successfully.")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None

    def preprocess_data(self, data_df, columns_to_encode=None):
        """Preprocess the data using encoding and other transformations."""
        if columns_to_encode is None:
            columns_to_encode = [
                "modified Zurich class",
                "largest spot size",
                "spot distribution",
            ]
        pipeline = Pipeline(
            [
                (
                    "encode",
                    make_column_transformer(
                        (OrdinalEncoder(), columns_to_encode), remainder="passthrough"
                    ),
                ),
                # ('scale', MinMaxScaler()),  Podemos agregar más transformaciones si es necesario.
            ]
        )
        encoded_scaled = pipeline.fit_transform(data_df)
        data_df_processed = pd.DataFrame(
            encoded_scaled, index=data_df.index, columns=data_df.columns
        )
        return data_df_processed

    def clean_data(self, df):
        """Clean the data by removing duplicates, dropping unnecessary columns, and encoding."""
        logger.info("Cleaning data...")
        df.drop_duplicates(inplace=True)
        try:
            df.drop("area of largest spot", axis=1, inplace=True)
        except:
            pass

        # Preprocess the data
        df_processed = self.preprocess_data(df)

        logger.info("Data cleaned and preprocessed.")
        return df_processed

    def save_data(self, df):
        """Save the preprocessed data to a CSV file."""
        try:
            os.makedirs(self.save_folder_path, exist_ok=True)
            df.to_csv(self.save_folder_path + "preprocessed_data.csv", index=False)
            logger.info(f"Data saved to {self.save_folder_path}")
        except Exception as e:
            logger.error(f"Error saving data: {e}")

    def main(self, save_local_data):
        raw_df = self.load_data()
        preprocess_df = self.preprocess_data(raw_df)
        clean_df = self.clean_data(preprocess_df)
        if save_local_data:
            self.save_data(clean_df)
        return clean_df


if __name__ == "__main__":
    execute_preprocess = DataPreprocessor().main()
