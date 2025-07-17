import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule

class TreeSpeciesDataset(Dataset):
    def __init__(self, hdf5_path):
        self.data = h5py.File(hdf5_path, 'r')
        self.points = self.data['points'][:]  # shape: (N, 1024, 3)
        self.labels = self.data['labels'][:]  # shape: (N, )

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        point_cloud = self.points[idx]
        label = self.labels[idx]
        return torch.as_tensor(point_cloud), torch.as_tensor(label).long()

class TreeDataModule(LightningDataModule):
    def __init__(self, hdf5_path, batch_size=32, num_workers=4, val_split=0.1):
        super().__init__()
        self.hdf5_path = hdf5_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split

    def setup(self, stage=None):
        dataset = TreeSpeciesDataset(self.hdf5_path)
        val_len = int(len(dataset) * self.val_split)
        train_len = len(dataset) - val_len
        self.train_ds, self.val_ds = random_split(dataset, [train_len, val_len])

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
