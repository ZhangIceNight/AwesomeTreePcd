# trainers/cls_trainer.py

import os
import torch
import numpy as np
from tqdm import tqdm

from datasets.pls_dataset import PLSDataset
from models import pointnet


class ClassificationTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')

        self.train_dataset = PLSDataset(
            root=cfg['data_root'],
            split='train',
            npoints=cfg['num_points'],
            use_uniform_sample=cfg['use_uniform_sample'],
            use_normals=cfg['use_normals']
        )
        self.test_dataset = PLSDataset(
            root=cfg['data_root'],
            split='test',
            npoints=cfg['num_points'],
            use_uniform_sample=cfg['use_uniform_sample'],
            use_normals=cfg['use_normals']
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'])
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'])

        self.num_classes = len(self.train_dataset.classes)
        self.model = pointnet.get_model(self.num_classes, normal_channel=cfg['use_normals']).to(self.device)
        self.criterion = pointnet.get_loss

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg['learning_rate'],
            weight_decay=cfg['weight_decay']
        )

        self.best_acc = 0.0
        os.makedirs(os.path.dirname(cfg['save_path']), exist_ok=True)

    def train(self):
        for epoch in range(self.cfg['num_epochs']):
            self.model.train()
            train_acc = []

            for points, labels in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.cfg['num_epochs']}"):
                points, labels = points.to(self.device), labels.to(self.device)
                points = points.transpose(2, 1)

                self.optimizer.zero_grad()
                preds, _ = self.model(points)
                loss = self.criterion(preds, labels)
                loss.backward()
                self.optimizer.step()

                pred_labels = preds.max(1)[1]
                acc = pred_labels.eq(labels).float().mean().item()
                train_acc.append(acc)

            avg_train_acc = np.mean(train_acc)
            print(f"[Epoch {epoch+1}] Train Acc: {avg_train_acc:.4f}")

            self.evaluate(epoch)

    def evaluate(self, epoch):
        self.model.eval()
        all_acc = []

        with torch.no_grad():
            for points, labels in self.test_loader:
                points, labels = points.to(self.device), labels.to(self.device)
                points = points.transpose(2, 1)
                preds, _ = self.model(points)
                pred_labels = preds.max(1)[1]
                acc = pred_labels.eq(labels).float().mean().item()
                all_acc.append(acc)

        avg_test_acc = np.mean(all_acc)
        print(f"[Epoch {epoch+1}] Test Acc: {avg_test_acc:.4f}")

        if avg_test_acc > self.best_acc:
            self.best_acc = avg_test_acc
            torch.save(self.model.state_dict(), self.cfg['save_path'])
            print(f"[✓] Saved best model with acc {avg_test_acc:.4f}")
