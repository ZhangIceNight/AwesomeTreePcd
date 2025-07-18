import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from models.pointnet_pl import PointCloudModel
from datasets.tree_dataset import TreeDataModule
from utils.logger_utils import setup_logger



if __name__ == "__main__":
    config_path = "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 初始化 logger
    log_file = "Results/training.log"
    logger = setup_logger(log_file)

    # Setup Module
    data_module = TreeDataModule(**config['data'])
    data_module.setup()

    # Setup Model
    model = PointCloudModel(model_hparams=config['model'], experiment_params=config)

    # Setup Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath="Results/checkpoints/",
        filename="pointnet-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        save_top_k=1,
        mode="min"
    )


    trainer = pl.Trainer(
        logger=False,  # 禁用 pytorch_lightning 默认的 logger（如 TensorBoard）
        callbacks=[checkpoint_cb],        
   
        **config['trainer']
    )

    # 添加训练开始日志
    logger.info("Training started")
    trainer.fit(model, datamodule=data_module)

    # 可以把最佳模型等再保存一次
    logger.info("Training finished. Best model saved.")
