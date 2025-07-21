import datetime
import shutil

import logging
import os
import sys

class LoggerWriter:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message.strip():
            self.logger.log(self.level, message.strip())

    def flush(self):
        pass


def setup_logger(log_file='training.log'):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # 创建 logger
    logger = logging.getLogger('PointCloudTraining')
    logger.setLevel(logging.INFO)

    # 避免重复添加 handler（应对 PyTorch Lightning 多次调用）
    if not logger.handlers:
        # 文件输出
        file_handler = logging.FileHandler(log_file, mode='w')
        # 控制台输出
        console_handler = logging.StreamHandler(sys.stdout)

        # 日志格式
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 添加 handler
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    # ✅ 将 print 输出和标准错误也写入日志文件
    sys.stdout = LoggerWriter(logger, logging.INFO)
    sys.stderr = LoggerWriter(logger, logging.ERROR)

    return logger





def create_experiment_dir(root_dir="Results", model_dir=None):
    # 创建时间戳目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    exp_dir = os.path.join(root_dir, model_dir, f"exp_{timestamp}")

    # 创建子目录
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    comet_dir = os.path.join(exp_dir, "comet_logs")
    log_dir = os.path.join(exp_dir, "logs")

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(comet_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    return exp_dir, ckpt_dir, comet_dir, log_dir
