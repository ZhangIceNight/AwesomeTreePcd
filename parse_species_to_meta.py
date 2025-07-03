# parse_species_to_meta.py

import pandas as pd

# 修改为你的 txt 文件路径
txt_path = '/home/wjzhang/workspace/datasets/PLS_AUT_Species/species.txt'
out_path = '/home/wjzhang/workspace/datasets/PLS_AUT_Species/meta.csv'

species_list = []
with open(txt_path, 'r') as f:
    for line in f:
        parts = line.strip().split()
        tree_class = parts[3].strip('"')     # "decid" or "conifer"
        species = parts[4].strip('"')        # OA, SP, etc.
        plot = parts[5].strip('"')           # e.g., A
        tree_id = parts[6].strip('"')        # e.g., A0001
        is_train = parts[7].strip()          # TRUE or FALSE

        filename = f'{tree_id}.laz'
        split = 'train' if is_train == 'TRUE' else 'test'

        species_list.append({
            'filename': filename,
            'species': species,
            'class': tree_class,
            'split': split
        })

df = pd.DataFrame(species_list)
df.to_csv(out_path, index=False)
print(f'[✓] meta.csv saved to: {out_path}')

