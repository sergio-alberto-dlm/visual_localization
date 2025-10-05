import copy
import random
from typing import List
import itertools

import numpy as np
from evo.core.trajectory import PosePath3D
import open3d as o3d


def compute_ate(traj_ref_se3: np.array, traj_est_se3: np.array) -> dict:
    # Extract translation vectors from 4x4 matrices
    positions_ref = np.array(traj_ref_se3)[:, :3, 3]
    positions_est = np.array(traj_est_se3)[:, :3, 3]

    # Euclidean distances
    errors = np.linalg.norm(positions_ref - positions_est, axis=1)
    
    # Compute metrics
    return {
        "rmse": np.sqrt(np.mean(errors**2)),
        "mean": np.mean(errors),
        "median": np.median(errors),
        "std": np.std(errors),
        "max": np.max(errors)
    }


def optimal_rotation(R_ref_list, R_src_list):
    assert len(R_ref_list) == len(R_src_list), "Mismatched list lengths"
    
    # Compute the covariance matrix
    M = np.zeros((3, 3))
    for R_ref, R_src in zip(R_ref_list, R_src_list):
        M += R_ref @ R_src.T

    # SVD of the covariance matrix
    U, _, Vt = np.linalg.svd(M)
    
    # Ensure a proper rotation (det = +1)
    T = U @ Vt
    if np.linalg.det(T) < 0:
        # Reflection detected, fix it
        U[:, -1] *= -1
        T = U @ Vt

    return T


def umeyama_align(src_poses: np.array, ref_poses: np.array) -> dict:
    traj_ref = PosePath3D(poses_se3=ref_poses)
    traj_est = PosePath3D(poses_se3=src_poses)
    traj_est_aligned = copy.deepcopy(traj_est)
    R, t, s = traj_est_aligned.align(traj_ref, correct_scale=True)

    ate = compute_ate(traj_ref.poses_se3, traj_est_aligned.poses_se3)

    return {
        'aligned_src_poses': np.array(traj_est_aligned.poses_se3),
        'error': ate['rmse'],
        'transform': (R, t, s)
    }


def full_align(src_poses: np.array, ref_poses: np.array) -> dict:
    umeyama_results = umeyama_align(src_poses, ref_poses)
    aligned_poses = umeyama_results['aligned_src_poses']
    R, t, s = umeyama_results['transform']

    ref_rots = ref_poses[:, :3, :3]
    aligned_rots = aligned_poses[:, :3, :3]
    T = optimal_rotation(ref_rots, aligned_rots)

    new_aligned_poses = np.array([
        apply_rotation_to_transform(pose, T)
        for pose in aligned_poses
    ])

    return {
        'aligned_src_poses': new_aligned_poses,
        'error': umeyama_results['error'],
        'transform': (T@R, t, s),
    }


def visualize_camera_trajectories(trajectories: List[np.array], scale: float=0.2) -> None:
    """
    trajectories: List of trajectories
                  Each trajectory is a list of 4x4 pose matrices (numpy arrays)
    scale: Size of the coordinate frame (camera model)
    """
    vis_geometries = []

    for i, traj in enumerate(trajectories):
        color = [random.random(), random.random(), random.random()]

        for pose in traj:
            # Create a coordinate frame
            cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
            cam.transform(pose)  # Place it using the 4x4 pose
            vis_geometries.append(cam)

        # Optionally connect camera centers with a line
        points = [pose[:3, 3] for pose in traj]
        lines = [[j, j+1] for j in range(len(points)-1)]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([color for _ in lines])

        vis_geometries.append(line_set)

    o3d.visualization.draw_geometries(vis_geometries)


def apply_similarity_transform(
    pose: np.array,
    R: np.array,
    t: np.array,
    s: float
) -> np.array:
    """
    pose: 4x4 numpy array
    R: 3x3 rotation
    t: 3x1 translation
    s: scalar scale
    """
    new_pose = np.eye(4)
    new_pose[:3, :3] = R @ pose[:3, :3]
    new_pose[:3, 3] = s * (R @ pose[:3, 3]) + t
    return new_pose


def apply_rotation_to_transform(M, T):
    assert M.shape == (4, 4), "M must be a 4x4 matrix"
    assert T.shape == (3, 3), "T must be a 3x3 rotation matrix"

    R = M[:3, :3]           # Extract rotation
    t = M[:3, 3]            # Extract translation

    R_new = T @ R           # Left-multiply rotation

    M_new = np.eye(4)
    M_new[:3, :3] = R_new
    M_new[:3, 3] = t        # Keep translation the same

    return M_new


def get_best_subset_alignment(src_poses: np.array, ref_poses: np.array, sample_size: int=4) -> dict:
    n = len(ref_poses)

    if sample_size < 3 or sample_size > n:
        raise ValueError("sample_size must be > 2 and less than len(poses)")
    
    index_combinations = list(itertools.combinations(range(n), sample_size))
    index_combinations = [list(combo) for combo in index_combinations]

    min_error = 1000
    best_results = None
    best_combo = None
    for combo in index_combinations:
        sub_src_poses = src_poses[combo]
        sub_ref_poses = ref_poses[combo]
        #align_results = umeyama_align(sub_src_poses, sub_ref_poses)
        align_results = full_align(sub_src_poses, sub_ref_poses)

        if align_results['error'] < min_error:
            min_error = align_results['error']
            best_results = align_results
            best_combo = combo

    return {
        **best_results,
        'idcs': best_combo,
        'ref_poses': ref_poses[best_combo],
    }






