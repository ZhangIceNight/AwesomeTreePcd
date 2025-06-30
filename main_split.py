from tools import split_and_save_tiles
from tools import check_las_coordinate_unit



check_las_coordinate_unit("your_data/plot_big.las")



# split_and_save_tiles(
#     file_path="your_data/plot_big.las",
#     output_dir="your_data/tiles",
#     tile_size=1.0,
#     min_points=100
# )
