import os
import numpy as np
import h5py
from pyntcloud import PyntCloud
from tqdm import tqdm
import json

def read_ply(filepath):
    try:
        cloud = PyntCloud.from_file(filepath)
        points = cloud.points.values  # (x, y, z)
        return points
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def farthest_point_sampling(point_cloud, num_samples):
    if point_cloud.shape[0] < num_samples:
        padding = np.zeros((num_samples - point_cloud.shape[0], 3))
        return np.vstack([point_cloud, padding])
    indices = [np.random.randint(point_cloud.shape[0])]  # 随机选第一个点
    distances = np.zeros(point_cloud.shape[0])  # 各点到已选点的距离
    for _ in range(num_samples - 1):
        dists = np.linalg.norm(point_cloud - point_cloud[indices[-1]], axis=1)
        distances = np.minimum(distances, dists)
        indices.append(np.argmax(distances))  # 选择最远点
    return point_cloud[indices]

def read_species_file(species_path):
    file_to_label = {}
    label_to_id = {}
    current_id = 0

    with open(species_path, 'r') as f:
        next(f) # skip the title line
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue  # 跳过不完整的行
            filename_id = parts[6]  # 第7列（index 6）
            category = parts[4]     # 第5列（index 4）

            if category not in label_to_id:
                label_to_id[category] = current_id
                current_id += 1

            file_to_label[filename_id] = label_to_id[category]

    return file_to_label, label_to_id


def save_ply_with_labels_to_hdf5(ply_dir, species_path, hdf5_path, num_samples=1024, log_file='process_log.txt'):
    # 读取 Species 文件
    file_to_label, label_to_id = read_species_file(species_path)
 
    # 收集所有 ply 文件
    ply_files = [f for f in os.listdir(ply_dir) if f.lower().endswith('.ply')]
    total_files = len(ply_files)
 
    all_points = []
    all_labels = []
 
    with open(log_file, 'a') as log_f:  # 创建日志文件
        for filename in tqdm(ply_files, desc="Processing .ply files", total=total_files):
            try:
                file_id = os.path.splitext(filename)[0]  # 去掉后缀
                label = file_to_label.get(file_id)

                # 读取并采样点云数据
                filepath = os.path.join(ply_dir, filename)
                points = read_ply(filepath)
                if points is not None:
                    sampled = farthest_point_sampling(points, num_samples)
                    all_points.append(sampled)
                    all_labels.append(label)
                # 尝试保存
                log_f.write(f"[SUCCESS] {filename}\n")
            except Exception as e:
                log_f.write(f"[FAILED] {filename} -> {str(e)}\n")
 
    # 转换为 NumPy 数组
    all_points = np.array(all_points, dtype=np.float32)  # (N, 1024, 3)
    all_labels = np.array(all_labels, dtype=np.int32)     # (N,)
 
    # 写入 HDF5
    with h5py.File(hdf5_path, 'w') as f:
        f.create_dataset('points', data=all_points)
        f.create_dataset('labels', data=all_labels)
        f.attrs['num_samples'] = num_samples
        f.attrs['num_classes'] = len(set(all_labels))
        f.attrs['label_map'] = json.dumps(label_to_id, ensure_ascii=False)  # 将 label_to_id 字典转换为 JSON 字符串，存入 HDF5 文件属性中


    print(f"Saved {len(all_points)} point clouds to {hdf5_path}")
    print(f"Species map is: {label_to_id}")

if __name__ == '__main__':
    ply_dir = "/home/wjzhang/workspace/datasets/PLS_AUT_ply_normalized_tiny"       # 替换为你的 PLY 文件夹路径
    species_path = "/home/wjzhang/workspace/datasets/PLS_AUT_Species/species.txt"      # 替换为你的 species.txt 路径
    hdf5_path = "/home/wjzhang/workspace/datasets/PLU_AUT_sample1024_xyzlbs.h5"

    save_ply_with_labels_to_hdf5(ply_dir, species_path, hdf5_path)
