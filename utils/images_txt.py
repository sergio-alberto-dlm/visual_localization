import os
from typing import List
import glob

import pandas as pd


def imgtxt_to_df(path: os.PathLike) -> pd.DataFrame:
    return pd.read_csv(path, skipinitialspace=True)


class ImagesTxt:
    def __init__(self, path: os.PathLike, dataset_path: os.PathLike = ''):
        if not os.path.exists(path):
            raise ValueError(f"input path: {path} does not exist.")
        abs_path = os.path.abspath(path)
        rel_path = os.path.relpath(path, start=dataset_path)
        dataset_to_rawdata_path = os.path.join(os.path.dirname(rel_path), 'raw_data')
        self.df = imgtxt_to_df(abs_path)
        ds_paths_f = lambda p: os.path.join(dataset_to_rawdata_path, p.strip())
        self.df['dataset_to_image_path'] = self.df['image_path'].apply(ds_paths_f)

        self.rel_path = rel_path
        self.abs_path = abs_path

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> object:
        return self.df.iloc[idx]

    def __repr__(self):
        return f"ImagesTxt({self.rel_path})"


def get_all_imagestxt(dataset_path: os.PathLike) -> List[ImagesTxt]:
    pattern = os.path.join(dataset_path, '**', 'images.txt')
    matching_files = glob.glob(pattern, recursive=True)

    relative_paths = [os.path.relpath(path, start=dataset_path) for path in matching_files]
    return {
        os.path.dirname(rel_path): ImagesTxt(matching_path, dataset_path)
        for rel_path, matching_path in zip(relative_paths, matching_files)
    }
