import torch
import numpy as np
import cv2
from PIL import Image
from typing import List, Tuple

# LightGlue & SuperPoint imports
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd

# Helper to convert PIL to torch.Tensor usable by LightGlue / SuperPoint
def pil_to_torch_rgb(pil_img: Image.Image, device: torch.device, resize_max: int = None) -> torch.Tensor:
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

def extract_matches_sp_lg(img0: torch.Tensor, img1: torch.Tensor,
                          extractor: SuperPoint, matcher: LightGlue
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns keypoint arrays pts0, pts1 of matched points between img0 and img1 using SuperPoint + LightGlue.

    pts0, pts1 are numpy arrays of shape (M,2) for M matched points.
    """
    # Extract features
    feats0 = extractor.extract(img0)
    feats1 = extractor.extract(img1)

    # Match
    matches01 = matcher({'image0': feats0, 'image1': feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
    matches = matches01['matches']  # shape (M,2) indices

    # Get keypoints
    keypts0 = feats0['keypoints'][matches[:,0]].cpu().numpy()  # shape (M,2)
    keypts1 = feats1['keypoints'][matches[:,1]].cpu().numpy()

    return keypts0, keypts1

def compute_homography_inlier_count(pts0: np.ndarray, pts1: np.ndarray,
                                   ransac_thresh: float = 4.0) -> int:
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

def rerank_with_sp_lg_ransac(query_img: Image.Image,
                             ref_imgs: List[Image.Image],
                             device: torch.device = 'cuda',
                             max_num_keypoints: int = 2048,
                             resize_max: int = 2048,
                             ransac_thresh: float = 4.0
                            ) -> Tuple[List[int], List[int]]:
    """
    Given one query PIL Image and list of reference PIL Images,
    returns re-ranked indices (ordered references) and their inlier counts.

    Returns:
      ranked_indices: list of indices into ref_imgs (0..len-1), ordered best → worst
      inlier_counts: list of inlier counts, same length as ref_imgs
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load extractor & matcher
    extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(device)
    matcher = LightGlue(features='superpoint').eval().to(device)

    # Convert query image
    q_img_t = pil_to_torch_rgb(query_img, device, resize_max=resize_max)

    inlier_counts = []

    for ref_img in ref_imgs:
        r_img_t = pil_to_torch_rgb(ref_img, device, resize_max=resize_max)

        # Extract matches
        pts_q, pts_r = extract_matches_sp_lg(q_img_t, r_img_t, extractor, matcher)

        # Compute inliers
        inliers = compute_homography_inlier_count(pts_q, pts_r, ransac_thresh=ransac_thresh)
        inlier_counts.append(inliers)

    # Rank by inlier counts descending
    ranked_indices = sorted(range(len(ref_imgs)), key=lambda i: inlier_counts[i], reverse=True)

    return ranked_indices, inlier_counts

