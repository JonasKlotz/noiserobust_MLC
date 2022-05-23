import os
import sys
import logging
from datetime import datetime


def create_logger(log_dir_path, logging_level=10):
    """
    Creates a logger with a time stamp output folder in
    :param logging_level: levels range from 50 - critical, 40 error .. to 10 - debug
    :param log_dir_path: log directory
    :return: logger
    """
    tick = datetime.now()

    log_dir_path += f'/log_{tick.strftime("%d.%m-%H:%M")}'

    logger = logging.getLogger("pytorch_logger")
    logger.setLevel(logging_level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    os.makedirs(log_dir_path, exist_ok=True)
    output_file_handler = logging.FileHandler(os.path.join(log_dir_path, "run.log"))
    output_file_handler.setFormatter(formatter)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)

    logger.addHandler(output_file_handler)
    logger.addHandler(stdout_handler)
    return logger
