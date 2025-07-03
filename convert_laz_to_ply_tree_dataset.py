import os
import pandas as pd
import numpy as np
import laspy
from pyntcloud import PyntCloud

def read_laz(filepath):
    with laspy.open(filepath) as f:
        las = f.read()
        return np.vstack((las.x, las.y, las.z)).T

def normalize_pc(pc):
    pc = pc - np.mean(pc, axis=0)
    scale = np.max(np.linalg.norm(pc, axis=1))
    return pc / scale

def save_ply(pc, save_path):
    df = pd.DataFrame(pc, columns=['x', 'y', 'z'])
    cloud = PyntCloud(df)
    cloud.to_file(save_path)

def process_dataset(meta_csv, input_dir, output_dir):
    df = pd.read_csv(meta_csv, sep='\t')  # 修改分隔符为你的真实情况
    for idx, row in df.iterrows():
        tree_id = row['id']
        species = row['species']
        is_train = row['train']
        split = 'train' if is_train else 'test'
        laz_path = os.path.join(input_dir, f"{tree_id}.laz")
        
        if not os.path.exists(laz_path):
            print(f"Missing: {laz_path}")
            continue
        
        try:
            pc = read_laz(laz_path)
            pc = normalize_pc(pc)
            save_dir = os.path.join(output_dir, species, split)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{tree_id}.ply")
            save_ply(pc, save_path)
        except Exception as e:
            print(f"Error processing {tree_id}: {e}")

# 运行入口（你可以根据自己的实际路径修改）
if __name__ == '__main__':
    meta_csv = 'tree_metadata.csv'             # 元信息路径
    input_dir = '/home/wjzhang/workspace/datasets/PLS_AUT_Species'           # .laz 文件所在文件夹
    output_dir = '/home/wjzhang/workspace/datasets/PLS_ModelNetStyle'                     # 输出保存目录
    process_dataset(meta_csv, input_dir, output_dir)

