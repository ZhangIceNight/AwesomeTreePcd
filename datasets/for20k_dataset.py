import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule
from utils import augmentations

class TreeSpeciesDataset(Dataset):
    def __init__(self, hdf5_path, split='train', fold_idx=0, use_normalization=False, augmentations_list=[]):
        self.split = split
        self.fold_idx = fold_idx
        self.use_normalization = use_normalization
        self.augmentations = augmentations_list
 
        self.data = h5py.File(hdf5_path, 'r')
        if self.split == 'train':
            self.points = self.data[f'train_{fold_idx}'][:]  # shape: (N, 1024, 3)
            self.labels = self.data[f'label_train_{fold_idx}'][:]  # shape: (N, )
        else:
            self.points = self.data[f'val_{fold_idx}'][:]  # shape: (N, 1024, 3)
            self.labels = self.data[f'label_val_{fold_idx}'][:]  # shape: (N, )
 
    def __len__(self):
        return len(self.points)
 
    def __getitem__(self, idx):
        point_cloud = self.points[idx]
        label = self.labels[idx]
 
        # 数据增强（仅对训练集）
        if self.split == 'train':
            if 'rotate' in self.augmentations:
                point_cloud = augmentations.random_rotate_point_cloud_y_axis(point_cloud)
            if 'scale' in self.augmentations:
                point_cloud = augmentations.random_scale_point_cloud(point_cloud)
            if 'shift' in self.augmentations:
                point_cloud = augmentations.random_shift_point_cloud(point_cloud)
            if 'dropout' in self.augmentations:
                point_cloud = augmentations.random_sample_dropout(point_cloud)
 
        # 归一化
        if self.use_normalization:
            point_cloud = augmentations.pc_normalize(point_cloud)
 
        return torch.as_tensor(point_cloud).float(), torch.as_tensor(label).long()

class TreeDataModule(LightningDataModule):
    def __init__(self, data_dir, batch_size=32, fold_idx=0, num_workers=4, use_normalization=False, augmentations_list=None, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.fold_idx = fold_idx
        self.num_workers = num_workers
        self.use_normalization = use_normalization
        self.augmentations_list = augmentations_list
    
    def setup(self, stage=None):
        self.train_ds = TreeSpeciesDataset(self.data_dir, split='train', fold_idx=self.fold_idx, use_normalization=self.use_normalization, augmentations_list=self.augmentations_list)
        self.val_ds = TreeSpeciesDataset(self.data_dir, split='val', fold_idx=self.fold_idx, use_normalization=self.use_normalization)
    
    
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

if __name__ == "__main__":
    data_module = TreeDataModule(
        hdf5_path="/public/wjzhang/datasets/for20k_sample1024_xyzlbs.h5",
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