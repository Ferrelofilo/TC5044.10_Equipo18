import os
import sys
from dataclasses import dataclass
from zipfile import ZipFile

import pandas as pd
import requests

# Initialize logger
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
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
    "area of largest spot",
    "common flares",
    "moderate flares",
    "severe flares",
)

DATA_FILE = "flare.data2"


DESTINATION_RAW_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../data/raw"
)


@dataclass
class DataIntoLocalFile:
    data_download_url: str = (
        "https://archive.ics.uci.edu/static/public/89/solar+flare.zip"
    )
    destination_path: str = DESTINATION_RAW_PATH

    def _check_destination_path(self):
        """Check if the destination path exists, if not, create it."""
        if not os.path.exists(self.destination_path):
            logger.info(
                f"Destination path {self.destination_path} dxsoes not exist. Creating it."
            )
            os.makedirs(self.destination_path)

    def download_data(self):
        """Download the data from the given URL and save it as a zip file in the destination path."""
        self._check_destination_path()

        # Get the filename from the URL
        filename = os.path.join(
            self.destination_path, self.data_download_url.split("/")[-1]
        )

        try:
            logger.info(f"Downloading data from {self.data_download_url}")
            response = requests.get(self.data_download_url, stream=True)
            response.raise_for_status()  # Check for errors in the download

            # Save the content as a zip file
            with open(filename, "wb") as file:
                for chunk in response.iter_content(chunk_size=128):
                    file.write(chunk)

            logger.info(f"Downloaded file saved as {filename}")
            return filename  # Return the file path for extraction

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download data: {e}")
            return None

    def extract_data(self, zip_file_path):
        """Extract the zip file to the destination path."""
        try:
            logger.info(f"Extracting {zip_file_path} to {self.destination_path}")
            with ZipFile(zip_file_path, "r") as zip_ref:
                zip_ref.extractall(self.destination_path)
            logger.info(f"Data successfully extracted to {self.destination_path}")
        except Exception as e:
            logger.error(f"Failed to extract the zip file: {e}")

    def download_and_extract(self):
        """Main function to download and extract the data."""
        zip_file_path = self.download_data()
        if zip_file_path:
            self.extract_data(zip_file_path)

        return self.destination_path


@dataclass
class CreateDF:
    file_paths: str = DESTINATION_RAW_PATH
    column_names: tuple[str] = DEFAULT_COLUMN_NAMES
    data_file: str = DATA_FILE

    def _read_raw_data(self):
        """Check if file paths exist and store them."""
        valid_paths = []
        # for file_name in self.data_file:
        # Construct the full file path (assuming files are in the current working directory)
        file_path = os.path.join(self.file_paths, self.data_file)
        if os.path.exists(self.file_paths):
            valid_paths.append(file_path)
            logger.debug(f"Found data file to extract: {file_path}")
            return valid_paths
        else:
            print(f"File not exiting: {file_path}")

    def _cleanup_files(self):
        """Delete all files in self.file_paths except for .csv files."""
        for filename in os.listdir(self.file_paths):
            file_path = os.path.join(self.file_paths, filename)
            # Check if the file is not a .csv file and is a file
            if os.path.isfile(file_path) and not filename.endswith(".csv"):
                logger.info(f"Deleting file: {file_path}")
                os.remove(file_path)

    def _save_df(self, df):
        saving_path = os.path.join(self.file_paths, "flare_data2_df.csv")
        try:
            df.to_csv(saving_path, index=False)
            logger.debug(f"Saved CSV on: {saving_path}")
        except Exception as exc:
            logger.error("Error saving the CSV of the DF", exc_info=exc)

    def create_dataframe(self):
        """Create pandas DataFrames from the raw data file paths and concatenate them."""
        valid_paths = self._read_raw_data()
        dataframes = []

        for file_path in valid_paths:
            try:
                df = pd.read_csv(
                    file_path,
                    delim_whitespace=True,
                    skiprows=1,
                    header=None,
                    names=self.column_names,
                )
                dataframes.append(df)
                logger.debug(f"DataFrame created for file: {file_path}")
            except Exception as e:
                logger.error(f"Error reading {file_path}", exc_info=e)

        # Concatenate all DataFrames into a single DataFrame
        if dataframes:
            # concatenated_df = pd.concat(dataframes, ignore_index=True)
            # logger.debug("All DataFrames have been concatenated.")
            self._cleanup_files()
            data_df = dataframes.pop()
            dp = data_df[data_df.duplicated(keep=False)]
            logger.info(f"Duplicated rows dropped : {dp.duplicated().sum()}")
            data_df.drop_duplicates(inplace=True)
            data_df.drop(
                ["area of largest spot"], axis=1, inplace=True
            )  # solo tiene 1 valor
            self._save_df(data_df)
            logger.info("Raw data saved.")
            return data_df

        logger.error("No DataFrames were created.")
        return None


# Example of how to use the class
if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_local = DataIntoLocalFile(destination_path=sys.argv[1])
        saved_path = data_local.download_and_extract()
        df_class = CreateDF(saved_path)
        final_df = df_class.create_dataframe()

    else:
        data_local = DataIntoLocalFile()
        saved_path = data_local.download_and_extract()
        df_class = CreateDF(saved_path)
        final_df = df_class.create_dataframe()
