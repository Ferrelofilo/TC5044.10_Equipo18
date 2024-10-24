import os
import sys

import pandas as pd

# Initialize logger
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from utils.logger_setup import setup_logger

logger = setup_logger(__name__)


def load_data(filepath):
    return pd.read_csv(filepath)


if __name__ == "__main__":
    logger.debug("Loadding data")
    data_path = sys.argv[1]
    output_file = sys.argv[2]
    data = load_data(data_path)
    data.to_csv(output_file, index=False)
