import h5py
import numpy as np
from sklearn.model_selection import KFold

def create_5fold_hdf5(input_path, output_path, n_splits=5):
    """
    input_path: 原始 hdf5 文件路径（含 points 和 labels）
    output_path: 输出文件路径，存储 5 fold 的 train/val 数据
    """
    # 加载数据
    with h5py.File(input_path, 'r') as hf:
        points = hf['points'][:]
        labels = hf['labels'][:]

    print(f"Total samples: {len(points)}")
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    with h5py.File(output_path, 'w') as out_hf:
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(points)):
            # 获取 train/val 数据集
            train_points, val_points = points[train_idx], points[val_idx]
            train_labels, val_labels = labels[train_idx], labels[val_idx]

            # 保存到 HDF5 中
            out_hf.create_dataset(f'train_{fold_idx}', data=train_points)
            out_hf.create_dataset(f'label_train_{fold_idx}', data=train_labels)
            out_hf.create_dataset(f'val_{fold_idx}', data=val_points)
            out_hf.create_dataset(f'label_val_{fold_idx}', data=val_labels)

            print(f"Fold {fold_idx}: train {len(train_points)}, val {len(val_points)}")

    print(f"Saved 5-fold dataset to {output_path}")

# 示例调用
if __name__ == "__main__":
    input_hdf5 = '/public/wjzhang/datasets/PLU_AUT_sample1024_xyzlbs.h5'
    output_hdf5 = '/public/wjzhang/datasets/PLU_AUT_sample1024_xyzlbs_5fold.h5'
    create_5fold_hdf5(input_hdf5, output_hdf5)
