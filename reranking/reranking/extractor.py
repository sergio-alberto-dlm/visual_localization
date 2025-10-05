import os
from typing import List

from PIL import Image
import numpy as np
import torch
import cv2
import pandas as pd

from lightglue import SuperPoint
from lightglue.utils import rbd

from . import utils as rr_utils


# Helper to convert PIL to torch.Tensor usable by LightGlue / SuperPoint
def pil_to_torch_rgb(
    pil_img: Image.Image,
    device: torch.device = 'cuda',
    resize_max: int = 2048
) -> torch.Tensor:
    """
    Convert PIL Image to torch.Tensor (3,H,W), normalized [0,1], optionally resize so max edge <= resize_max.
    """
    img = pil_img.convert('RGB')
    img_np = np.array(img).astype(np.float32) / 255.0  # H x W x 3
    # maybe resize
    if resize_max is not None:
        h, w = img_np.shape[:2]
        scale = resize_max / max(h, w)
        if scale < 1.0:
            new_size = (int(w * scale), int(h * scale))
            img_np = cv2.resize(img_np, new_size, interpolation=cv2.INTER_LINEAR)
    # to torch and permute
    img_t = torch.from_numpy(img_np).permute(2,0,1).unsqueeze(0).to(device)  # shape (1,3,H,W)
    return img_t


class Extractor:
    def __init__(self, max_num_keypoints: int=2048):
        self.superpoint = None
        self.max_num_keypoints = max_num_keypoints
        self.wakeup()

    def wakeup(self) -> None:
        self.superpoint = SuperPoint(max_num_keypoints=self.max_num_keypoints)
        self.superpoint = self.superpoint.eval().to('cuda')

    def run(self, image_list: List[Image.Image]) -> List[dict]:
        features_list = []
        for img in image_list:
            img_tensor = pil_to_torch_rgb(img)

            with torch.no_grad():
                feats = self.superpoint.extract(img_tensor)

            feats = {
                k: v.cpu() for k, v in feats.items()
            }

            features_list.append(feats)

        return features_list


class SessionDumpedFeatures:
    def __init__(self, dump_dir: os.PathLike, location: str, session: str) -> None:
        self.dump_dir = dump_dir
        self.location = location.upper()
        self.session = session
        self.session_path = os.path.join(
            self.dump_dir, self.location, 'sessions', self.session
        )
        csv_path = os.path.join(self.session_path, 'local_feats.csv')
        self.df = pd.read_csv(csv_path)

    def __getitem__(self, dataset_to_img_path: os.PathLike) -> dict:
        row = self.df[self.df['dataset_to_image_path'] == dataset_to_img_path]
        if len(row) == 0:
            raise ValueError(f"{dataset_to_img_path} not found in {self.session_path}")
        
        npz_rel_path = row['outputdir_to_feats_path'].values[0]
        npz_abs_path = os.path.join(self.dump_dir,  npz_rel_path)
        feats = np.load(npz_abs_path)

        return feats

    def __repr__(self) -> str:
        return f"SessionDumpedFeatures({self.location}/sessions/{self.session})"


class DumpedFeatures:
    def __init__(self, dump_dir: os.PathLike) -> None:
        self.dump_dir = dump_dir
        all_combos = [
            (location, session)
            for location in rr_utils.LOCATIONS
            for session in rr_utils.SESSIONS
        ]

        self.sessions_features = {
            DumpedFeatures.key(location, session):\
                SessionDumpedFeatures(self.dump_dir, location, session)
            for location, session in all_combos
        }

    @staticmethod
    def key(location: str, session: str):
        return f"{location.upper()}/sessions/{session}"

    def __getitem__(self, dataset_to_img_path: os.PathLike) -> dict:
        location, session = rr_utils.get_location_and_session(dataset_to_img_path)
        key = DumpedFeatures.key(location, session)
        features_map = self.sessions_features[key]

        return features_map[dataset_to_img_path]
