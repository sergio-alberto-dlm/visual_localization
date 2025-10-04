import os
from typing import List

from PIL import Image
import numpy as np
import torch
import cv2

from lightglue import SuperPoint
from lightglue.utils import rbd



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
    def __init__(self, max_num_keypoints: int=4096):
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
