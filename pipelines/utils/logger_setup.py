import logging
import os
import inspect


# Function to capture the calling file path and line number for logging
def get_caller_info():
    frame = inspect.stack()[2]
    filename = os.path.basename(frame.filename)
    line_number = frame.lineno
    return f"{filename}:{line_number}"


# Custom formatter to include the file and line number of the calling code
class CustomFormatter(logging.Formatter):
    def format(self, record):
        caller_info = get_caller_info()
        record.msg = f"{caller_info} - {record.msg}"
        return super().format(record)


# Function to set up and return a logger
def setup_logger(name, log_file="app.log", log_level=logging.DEBUG):
    logger = logging.getLogger(name)

    # Check if the logger already has handlers to avoid duplication
    if not logger.hasHandlers():
        logger.setLevel(log_level)

        # Create a file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)

        # Create a console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)

        # Define a common formatter
        formatter = CustomFormatter("%(asctime)s - [%(levelname)s] - %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add both handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
