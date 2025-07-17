from tools.preprocess import split_and_save_tiles_with_labels

for site in ["Plot_1", "Plot_3", "Plot_5"]:
    split_and_save_tiles_with_labels(
        file_path=f"data/ForestSemantic/{site}.las",
        output_dir=f"data/ForestSemantic/tiles_{site}/",
        tile_size=1.0,
        min_points=100
    )
