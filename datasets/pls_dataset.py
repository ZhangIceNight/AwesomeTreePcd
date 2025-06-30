# datasets/pls_dataset.py

import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pyntcloud import PyntCloud

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    scale = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    return pc / scale

def farthest_point_sample(points, npoint):
    N, D = points.shape
    xyz = points[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    return points[centroids.astype(np.int32)]

class PLSDataset(Dataset):
    def __init__(self, root, npoints=1024, split='train', use_uniform_sample=True, use_normals=False):
        super().__init__()
        self.root = root
        self.npoints = npoints
        self.use_uniform_sample = use_uniform_sample
        self.use_normals = use_normals
        self.split = split

        self.classes = sorted(os.listdir(root))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.datapath = []

        for cls in self.classes:
            folder = os.path.join(root, cls, split)
            if not os.path.exists(folder): continue
            for fname in glob.glob(os.path.join(folder, '*.ply')):
                self.datapath.append((cls, fname))

        print(f"[PLSDataset] Loaded {len(self.datapath)} {split} samples from {len(self.classes)} classes.")

    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, index):
        cls, filepath = self.datapath[index]
        label = self.class_to_idx[cls]

        cloud = PyntCloud.from_file(filepath)
        pc = cloud.points[['x', 'y', 'z']].values.astype(np.float32)
        if len(pc) < self.npoints:
            repeat = self.npoints // len(pc) + 1
            pc = np.tile(pc, (repeat, 1))[:self.npoints, :]
        else:
            if self.use_uniform_sample:
                pc = farthest_point_sample(pc, self.npoints)
            else:
                pc = pc[:self.npoints]

        pc = pc_normalize(pc)

        return torch.tensor(pc, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
