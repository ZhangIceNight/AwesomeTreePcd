# utils/logger_utils.py

import logging
import os

def setup_logger(log_file='training.log'):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger('PointCloudTraining')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file, mode='w')
    console_handler = logging.StreamHandler()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
