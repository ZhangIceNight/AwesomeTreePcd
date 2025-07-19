import pytorch_lightning as pl
import torch
from torch import optim, nn
from .pointnet import PointNetCls
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

class PointCloudModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model_hparams = config.model
        self.opt_hparams = config.optimizer
        self.model = PointNetCls(num_classes=self.model_hparams['num_classes'])
        self.loss_fn = nn.CrossEntropyLoss()

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        points, labels = batch
        logits = self.model(points)
        loss = self.loss_fn(logits, labels.squeeze())
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        points, labels = batch
        logits = self.model(points)
        loss = self.loss_fn(logits, labels.squeeze())
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == labels).float().mean()

        # 记录验证损失 & 准确率
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_acc", accuracy, prog_bar=True, logger=True)

        return {
            "val_loss": loss,
            "val_acc": accuracy
        }

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.opt_hparams["learning_rate"], weight_decay=self.opt_hparams["weight_decay"])
 
        total_epochs = self.opt_hparams["max_epochs"]  # 例如：50
        warmup_epochs = self.opt_hparams["warmup_epochs"]

        scheduler_warmup = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs
        )
    
        # Cosine 退火阶段：从 max lr 衰减到接近 0
        scheduler_cosine = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,  
            eta_min=self.opt_hparams["eta_min"]
        )
    
        # 合并两个调度器为一个阶段式调度器
        combined_scheduler = SequentialLR(
            optimizer,
            schedulers=[scheduler_warmup, scheduler_cosine],
            milestones=[warmup_epochs]  # 第 5 个 epoch 结束后切换到 cosine
        )
 
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': combined_scheduler,
                'interval': 'epoch', 
                'frequency': 1
            }
        }
 
   

 
