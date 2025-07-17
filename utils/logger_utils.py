import logging
import os

def setup_logger(log_path):
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)

    # 防止重复添加 handler
    if not logger.hasHandlers():
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
