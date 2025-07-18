import pytorch_lightning as pl
import torch
from torch import optim, nn
from .pointnet import PointNetCls

class PointCloudModel(pl.LightningModule):
    def __init__(self, model_hparams, experiment_params):
        super().__init__()
        self.save_hyperparameters()
        self.model_hparams = model_hparams
        self.model = PointNetCls(num_classes=self.model_hparams['num_classes'])
        self.loss_fn = nn.CrossEntropyLoss()
        print(f"LR Type: {type(self.model_hparams['learning_rate'])}")
        print(f"LR Value: {self.model_hparams['learning_rate']}")

    def training_step(self, batch, batch_idx):
        points, labels = batch
        logits = self.model(points)
        loss = self.loss_fn(logits, labels.squeeze())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        points, labels = batch
        logits = self.model(points)
        loss = self.loss_fn(logits, labels.squeeze())
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == labels).float().mean()

        # 记录验证损失 & 准确率
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", accuracy, prog_bar=True)

        return {
            "val_loss": loss,
            "val_acc": accuracy
        }

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.model_hparams["learning_rate"])


 
