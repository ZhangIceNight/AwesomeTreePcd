import logging
import yaml
import os
import argparse
from easydict import EasyDict as edict

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger

from models.pointnet_pl import PointCloudModel
from datasets.tree_dataset import TreeDataModule
from utils.logger_utils import setup_logger
from utils.seed_utils import seed_everything



def get_args():
    parser = argparse.ArgumentParser(description="Train PointNet with configurable args")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint for resuming")
    parser.add_argument("--wandb_name", type=str, help="Override wandb run name")
    parser.add_argument("--batch_size", type=int, help="Override data.batch_size")
    parser.add_argument("--epochs", type=int, help="Override trainer.max_epochs")
    parser.add_argument("--lr", type=float, help="Override model.learning_rate")
 
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return edict(config)

def override_config(config, args):
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.epochs:
        config.trainer.max_epochs = args.epochs
    if args.lr:
        config.model.learning_rate = args.lr
    if args.wandb_name:
        config.wandb.name = args.wandb_name
    return config

def train():
    args = get_args()

    config = load_config(args.config)
    config = override_config(config, args)
    
    # 初始化 logger
    log_file = "Results/training.log"
    logger = setup_logger(log_file)


    comet_logger = CometLogger(
        project=config.comet.project,
        name=config.comet.get("name"), 
        offline_directory=config.comet.offline_directory
    )

    # Setup Dataset Module
    data_module = TreeDataModule(**config.data)
    data_module.setup()

    # Setup Model
    model = PointCloudModel(config)

    # Setup Callbacks
    best_checkpoint_cb = ModelCheckpoint(
        dirpath="Results/checkpoints/",
        filename="pointnet-{epoch:02d}-val_acc{val_acc:.2f}",
        monitor="val_acc",
        save_top_k=1,
        mode="max",
        auto_insert_metric_name=False
    )

    latest_checkpoint_cb = ModelCheckpoint(
        dirpath="Results/checkpoints/",
        filename="pointnet-latest",
        save_top_k=1,
        every_n_epochs=1,
        save_last=True,
    )

    trainer = pl.Trainer(
        logger=comet_logger,
        callbacks=[best_checkpoint_cb, latest_checkpoint_cb],        
        **config['trainer']
    )

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.fit(model, datamodule=data_module, ckpt_path=args.resume)
    else:
        print("Training from scratch")
        trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    seed_everything(98)
    train()