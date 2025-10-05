from typing import Any

import numpy as np
import torch
import cv2
from lightglue import LightGlue
from lightglue.utils import rbd


def compute_homography_inlier_count(
    pts0: np.ndarray,
    pts1: np.ndarray,
    ransac_thresh: float = 4.0
) -> int:
    """
    Estimate homography by RANSAC, return inlier count.
    """
    if len(pts0) < 4:
        return 0

    H, mask = cv2.findHomography(pts0, pts1, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
    if mask is None:
        return 0
    inlier_count = int(mask.sum())
    return inlier_count


class Matcher:
    def __init__(self, ransac_thresh: float = 4.0):
        self.lightglue = LightGlue(features='superpoint').eval().to('cuda')
        self.ransac_thresh = ransac_thresh

    def get_keypts(self, feats0: Any, feats1: Any) -> tuple:
        feats0 = {
            k: torch.Tensor(v).to('cuda')
            for k, v in feats0.items()
        }
        feats1 = {
            k: torch.Tensor(v).to('cuda')
            for k, v in feats1.items()
        }

        #Match
        matches01 = self.lightglue({'image0': feats0, 'image1': feats1})
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
        matches = matches01['matches']  # shape (M,2) indices

        # Get keypoints
        keypts0 = feats0['keypoints'][matches[:,0]].cpu().numpy()  # shape (M,2)
        keypts1 = feats1['keypoints'][matches[:,1]].cpu().numpy()

        return keypts0, keypts1

    def get_inliers_count(self, feats0: Any, feats1: Any) -> int:
        keypts0, keypts1 = self.get_keypts(feats0, feats1)
        inliers_cnt = compute_homography_inlier_count(
            keypts0,
            keypts1,
            ransac_thresh = self.ransac_thresh
        )

        return inliers_cnt
