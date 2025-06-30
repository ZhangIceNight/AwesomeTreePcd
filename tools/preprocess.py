import os
import laspy
import open3d as o3d
import numpy as np

def read_single_las(file_path):
    """读取单个 LAS 文件为 Open3D 点云对象"""
    if not os.path.exists(file_path):
        print(f"❌ 文件未找到：{file_path}")
        return None
    try:
        las = laspy.read(file_path)
        xyz = np.vstack((las.x, las.y, las.z)).T
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        return pcd
    except Exception as e:
        print(f"⚠️ 读取失败 {file_path}：{e}")
        return None


def split_and_save_tiles(file_path, output_dir, tile_size=1.0, min_points=100):
    """将点云按 tile_size 网格切割并保存为 .npy"""
    os.makedirs(output_dir, exist_ok=True)
    pcd = read_single_las(file_path)
    if pcd is None:
        return
    xyz = np.asarray(pcd.points)
    min_x, min_y = xyz[:, 0].min(), xyz[:, 1].min()
    tiles = {}
    for point in xyz:
        i = int((point[0] - min_x) // tile_size)
        j = int((point[1] - min_y) // tile_size)
        key = (i, j)
        tiles.setdefault(key, []).append(point)
    print(f"✅ 切割完成，共生成 {len(tiles)} 个格子")
    count = 0
    for key, points in tiles.items():
        if len(points) < min_points:
            continue
        points = np.array(points)
        out_path = os.path.join(output_dir, f"tile_{key[0]}_{key[1]}.npy")
        np.save(out_path, points)
        count += 1
    print(f"✅ 有效 tile 保存完成，共保存 {count} 个")


def check_las_coordinate_unit(file_path, verbose=True):
    """
    检查 .las 文件坐标单位是否为“米”，并打印坐标范围与推测单位。
    :param file_path: str
    :param verbose: bool
    :return: str, 单位猜测结果（"meter", "centimeter", "millimeter", "unknown"）
    """
    try:
        las = laspy.read(file_path)
        scale = las.header.scales
        offset = las.header.offsets
        x, y, z = las.x, las.y, las.z
        dx, dy, dz = x.max() - x.min(), y.max() - y.min(), z.max() - z.min()

        if verbose:
            print("📦 Header Info:")
            print(f"  Scale:  {scale}")
            print(f"  Offset: {offset}")
            print("📏 坐标范围差值:")
            print(f"  x: {dx:.2f} m")
            print(f"  y: {dy:.2f} m")
            print(f"  z: {dz:.2f} m")

        unit_guess = "unknown"
        if scale[0] >= 1.0 or max(dx, dy) > 10:
            unit_guess = "meter"
        elif scale[0] >= 0.01 and max(dx, dy) > 100:
            unit_guess = "centimeter"
        elif scale[0] >= 0.001 and max(dx, dy) > 1000:
            unit_guess = "millimeter"

        if verbose:
            print(f"🧠 推测单位：{unit_guess}")

        return unit_guess

    except Exception as e:
        print(f"读取失败：{e}")
        return "error"
