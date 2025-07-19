import logging
import os

from omegaconf import DictConfig, OmegaConf
import hydra

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger

from models.pointnet_pl import PointCloudModel
from datasets.tree_dataset import TreeDataModule
from utils.logger_utils import setup_logger
from utils.seed_utils import seed_everything

@hydra.main(config_path="configs", config_name="PLU_AUT_pointnet_lr1e-3_bs32", version_base=None)
def train(config: DictConfig):
    # 初始化 logger
    log_file = "Results/training.log"
    logger = setup_logger(log_file)

    print(OmegaConf.to_yaml(config))
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

    if config.model.get("resume"):
        print(f"Resuming from checkpoint: {config.model.resume}")
        trainer.fit(model, datamodule=data_module, ckpt_path=config.model.resume)
    else:
        print("Training from scratch")
        trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    seed_everything(42)
    train()