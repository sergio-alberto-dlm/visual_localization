import os
from typing import List

import numpy as np

from .inference import CudaInference, ApiInference
from .align import get_best_subset_alignment, apply_similarity_transform


def full_vggt_extrinsics(vgg_extrinsics: np.array) -> np.array:
    full_extrinsics = np.array([
        np.eye(4)
        for _ in range(len(vgg_extrinsics))
    ])
    full_extrinsics[:, :3, :] = vgg_extrinsics
    return full_extrinsics


class VggtQueryPoseEstimation:
    def __init__(self):
        self.vggt = CudaInference()

    def get_query_pose(
        self,
        query_img_path: os.PathLike,
        ref_img_paths: List[os.PathLike],
        ref_img_poses: List|np.array,
        dataset_path: os.PathLike,
        sample_size: int = 4
    ) -> dict:
        query_abspath = os.path.join(dataset_path, query_img_path)
        ref_abspaths = [
            os.path.join(dataset_path, path)
            for path in ref_img_paths
        ]
        ref_img_poses = np.array(ref_img_poses)

        vggt_input_path_list = [query_abspath] + ref_abspaths
        vggt_predictions = self.vggt.run(vggt_input_path_list)
        vggt_predictions['Twc'] = np.linalg.inv(
            full_vggt_extrinsics(vggt_predictions['extrinsic'])
        )


        query_vggt_predictions = {
            k: v[0] for k, v in vggt_predictions.items()
        }
        ref_vggt_predictions = {
            k: v[1:] for k, v in vggt_predictions.items()
        }

        n_ref_images = len(ref_img_paths)
        alignment_results = get_best_subset_alignment(
            ref_vggt_predictions['Twc'],
            ref_img_poses,
            min(sample_size, n_ref_images)
        )

        R, t, s = alignment_results['transform']
        est_query_pose = apply_similarity_transform(
            query_vggt_predictions['Twc'],
            R, t, s
        )

        return {
            'error': alignment_results['error'],
            'pose': est_query_pose 
        }
