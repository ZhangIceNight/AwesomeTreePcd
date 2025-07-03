from tools import split_and_save_tiles
from tools import check_las_coordinate_unit
from tools import read_single_las
import numpy as np


#-----------------------------------------------#
# check_las_coordinate_unit("data/Plot_5.las")
#-----------------------------------------------#

#-----------------------------------------------#
# pcd = read_single_las("data/Plot_5.las")
# xyz = np.asarray(pcd.points)

# # 打印前 5 个点
# for i in range(5):
#     print(f"Point {i}: x={xyz[i,0]:.3f}, y={xyz[i,1]:.3f}, z={xyz[i,2]:.3f}")
#-----------------------------------------------#

#-----------------------------------------------#
split_and_save_tiles(
    file_path="data/Plot_5.las",
    output_dir="data/tiles",
    tile_size=1.0,
    min_points=100
)
