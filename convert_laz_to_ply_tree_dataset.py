import os
import numpy as np
import laspy
import open3d as o3d
from tqdm import tqdm

def read_laz(filepath):
    try:
        with laspy.open(filepath) as f:
            las = f.read()
            # 提取 x, y, z 三个坐标
            points = np.vstack((las.x, las.y, las.z)).T
            return points
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def normalize_pc(pc):
    if pc is None or len(pc) == 0:
        return None
    pc = pc - np.mean(pc, axis=0)
    scale = np.max(np.linalg.norm(pc, axis=1))
    return pc / scale if scale != 0 else pc

def save_ply(pc, save_path):
    if pc is None or len(pc) == 0:
        return
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    o3d.io.write_point_cloud(save_path)

def process_dataset(input_dir, output_dir, log_file='process_log.txt'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 打开日志文件（追加模式）
    with open(log_file, 'a') as log:
        # 获取所有 .laz 文件
        laz_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.laz')]
        total_files = len(laz_files)
        
        for idx, filename in enumerate(tqdm(laz_files, desc="Processing Files", total=total_files)):
            filepath = os.path.join(input_dir, filename)
            base_name = os.path.splitext(filename)[0]
            save_path = os.path.join(output_dir, base_name + ".ply")

            point_cloud = read_laz(filepath)
            normalized_pc = normalize_pc(point_cloud)

            # 尝试保存
            try:
                save_ply(normalized_pc, save_path)
                log.write(f"[SUCCESS] {filename} -> {save_path}\n")
            except Exception as e:
                log.write(f"[FAILED] {filename} -> {str(e)}\n")

if __name__ == '__main__':
    input_dir = '/home/wjzhang/workspace/datasets/PLS_AUT_Species'           # .laz 文件所在文件夹
    output_dir = '/home/wjzhang/workspace/datasets/PLS_AUT_ply_normalized'     # 输出保存目录
    process_dataset(input_dir, output_dir)
