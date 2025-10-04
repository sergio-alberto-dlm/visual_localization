import os

from tqdm import tqdm
from PIL import Image
import pandas as pd
import numpy as np

from .extractor import Extractor


def extract_and_save(
        data: pd.DataFrame,
        dataset_path: os.PathLike,
        root_dump_path: os.PathLike
    ) -> pd.DataFrame:

    dataset_path = os.path.abspath(dataset_path)
    root_dump_path = os.path.abspath(root_dump_path)

    dataset_img_paths = [
        row['dataset_to_image_path']
        for row in data
    ]

    abs_img_paths = [
        os.path.join(dataset_path, path)
        for path in dataset_img_paths
    ]

    
    superpoint = Extractor()
    number_of_images = len(dataset_img_paths)
    for i in tqdm(range(number_of_images)):
        rel_dump_path = dataset_img_paths[i]
        abs_img_path = abs_img_paths[i]
        img = [Image.open(abs_img_path)]
        feats = superpoint.run(img)[0]

        dumproot_to_npz_dirname = os.path.dirname(rel_dump_path)
        dumproot_to_npz_path = os.path.join(dumproot_to_npz_dirname, 'local_feats.npz')

        npz_apspath = os.path.join(root_dump_path, dumproot_to_npz_path)
        os.makedirs(os.path.dirname(npz_apspath), exist_ok=True)
        np.savez(npz_apspath, **feats)

    dumproot_npz_paths = [
        os.path.join(os.path.dirname(ds_img_path), 'local_feats.npz')
        for ds_img_path in dataset_img_paths
    ]

    img_to_npz_dict_map = {
        'dataset_to_image_path': dataset_img_paths,
        'outputdir_to_feats_path': dumproot_npz_paths
    }

    return pd.DataFrame(img_to_npz_dict_map)
