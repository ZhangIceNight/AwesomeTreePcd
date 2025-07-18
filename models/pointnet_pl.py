import pytorch_lightning as pl
import torch
from torch import optim, nn
from models.pct import NaivePCT
from models.pointnet import PointNetCls

class PointCloudModel(pl.LightningModule):
    def __init__(self, model_hparams, experiment_params):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = model_hparams["num_classes"]
        self.model = PointNetCls(num_classes=self.num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

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
        return optim.Adam(self.parameters(), lr=self.hparams.model_hparams["learning_rate"])


 
if __name__ == "__main__":
    from datasets.tree_dataset import TreeDataModule
    data_module = TreeDataModule(
        hdf5_path="data/your_pointcloud_data.h5",
        batch_size=8,
        num_workers=0,
        val_split=0.1
    )
 
    data_module.setup()
 
    # 测试 data_loader
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))
    points, labels = batch
 
    print("Point cloud batch shape: ", points.shape)   # Should be [B, N, 3]
    print("Label batch shape: ", labels.shape)         # Should be [B, ]
    
    model_hparams = {"num_classes":4}
    model = PointCloudModel()
    logits = model(points)
    print("Model outputs shape: ", logits.shape) # Should be [B, num_classes] → [4, 4]