import logging
import yaml
import os
import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from models.pointnet_pl import PointCloudModel
from datasets.tree_dataset import TreeDataModule
from utils.logger_utils import setup_logger

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Path to .ckpt file")
    return parser.parse_args()


def train(args):
    config_path = "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 初始化 logger
    log_file = "Results/training.log"
    logger = setup_logger(log_file)

    # init wandb logger
    wandb_logger = WandbLogger(
        project="TreeSpecies_cls",
        name="pointnetv1",
        group="baselines",
        log_model="all",
        config=config
    )
    # Setup Module
    data_module = TreeDataModule(**config['data'])
    data_module.setup()

    # Setup Model
    model = PointCloudModel(model_hparams=config['model'], experiment_params=config)

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
        logger=wandb_logger,
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
    args = get_args()
    train(args)