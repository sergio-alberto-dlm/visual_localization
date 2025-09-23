import numpy as np
import open3d as o3d


def get_conf_mask(predictions: dict, conf_thres: float=10.0) -> np.ndarray:
    pred_world_points = predictions["world_points_from_depth"]
    pred_world_points_conf = predictions.get("depth_conf", np.ones_like(pred_world_points[..., 0]))
    conf = pred_world_points_conf.reshape(-1) #Flatten confidence matrix
    conf_threshold = np.percentile(conf, conf_thres) #determine the conf_thres'th percentile.
    conf_mask = (conf >= conf_threshold) & (conf > 1e-5) #Idcs where conf is greater than the conf_thres'th percentile.

    return conf_mask


def to_point_cloud(predictions: dict, conf_thres: float = 80.0) -> object:
    point_cloud = o3d.geometry.PointCloud()

    conf_mask = get_conf_mask(predictions, conf_thres)
    points = predictions["world_points_from_depth"].reshape(-1, 3)
    colors = predictions["images"].transpose(0, 2, 3, 1).reshape(-1, 3)

    point_cloud.points = o3d.utility.Vector3dVector(points[conf_mask])
    point_cloud.colors = o3d.utility.Vector3dVector(colors[conf_mask])

    return point_cloud
