import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
import matplotlib


def get_conf_mask(predictions: dict, conf_thres: float=10.0) -> np.ndarray:
    pred_world_points = predictions["world_points_from_depth"]
    pred_world_points_conf = predictions.get("depth_conf", np.ones_like(pred_world_points[..., 0]))
    conf = pred_world_points_conf.reshape(-1) #Flatten confidence matrix
    conf_threshold = np.percentile(conf, conf_thres) #determine the conf_thres'th percentile.
    conf_mask = (conf >= conf_threshold) & (conf > 1e-5) #Idcs where conf is greater than the conf_thres'th percentile.

    return conf_mask


def to_point_cloud(predictions: dict, conf_thres: float = 80.0, show_cam: bool = False) -> object:
    point_cloud = o3d.geometry.PointCloud()

    conf_mask = get_conf_mask(predictions, conf_thres)
    points = predictions["world_points_from_depth"].reshape(-1, 3)
    colors = predictions["images"].transpose(0, 2, 3, 1).reshape(-1, 3)

    point_cloud.points = o3d.utility.Vector3dVector(points[conf_mask])
    point_cloud.colors = o3d.utility.Vector3dVector(colors[conf_mask])

    pcds = [point_cloud]

    
    if show_cam:
        scene_scale = calc_scene_scale(predictions, conf_thres)

        colormap = matplotlib.colormaps.get_cmap("gist_rainbow")
        

        camera_matrices = predictions["extrinsic"]
        num_cameras = len(camera_matrices)
        extrinsics_matrices = np.zeros((num_cameras, 4, 4))
        extrinsics_matrices[:, :3, :4] = camera_matrices
        extrinsics_matrices[:, 3, 3] = 1

        colors = generate_colormap_colors(num_cameras, 'copper')

        cams_pcd = []
        for i in range(num_cameras):
            world_to_camera = extrinsics_matrices[i]
            camera_to_world = np.linalg.inv(world_to_camera)
            rgba_color = colormap(i / num_cameras)
            current_color = tuple(int(255 * x) for x in rgba_color[:3])
            current_color = colors[i]

            cams_pcd.append(integrate_camera_as_open3d_vector(camera_to_world, current_color, scene_scale))

        pcds += cams_pcd

    return pcds


def calc_scene_scale(predictions: dict, conf_thres: float):
    if "world_points_from_depth" not in predictions or "depth_conf" not in predictions:
        raise ValueError("Missing required prediction keys.")

    pred_world_points = predictions["world_points_from_depth"]
    pred_world_points_conf = predictions["depth_conf"]

    # Flatten confidence and points
    conf = pred_world_points_conf.reshape(-1)
    world_points = pred_world_points.reshape(-1, 3)

    # Apply confidence threshold
    conf_threshold = np.percentile(conf, conf_thres)
    min_valid_conf = max(conf_threshold, 1e-5)
    conf_mask = conf >= min_valid_conf
    valid_points = world_points[conf_mask]

    if valid_points.shape[0] == 0:
        raise ValueError("No points passed the confidence threshold.")

    # Percentile-based bounding box
    lower = np.percentile(valid_points, 5, axis=0)
    upper = np.percentile(valid_points, 95, axis=0)

    # Diagonal length = scene scale
    scene_scale = np.linalg.norm(upper - lower)
    return scene_scale


def get_opengl_conversion_matrix():
    """
    Returns a 4x4 matrix that converts from standard coordinate system to OpenGL-style camera system.
    """
    flip_z = np.eye(4)
    flip_z[2, 2] = -1
    return flip_z


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Applies a 4x4 transformation to a Nx3 array of points.
    """
    num_points = points.shape[0]
    points_hom = np.hstack((points, np.ones((num_points, 1))))
    transformed = (transform @ points_hom.T).T
    return transformed[:, :3]


def apply_transform_to_mesh(mesh: o3d.geometry.TriangleMesh, transform: np.ndarray):
    """
    Applies a 4x4 transformation to an Open3D mesh in-place.
    """
    mesh.transform(transform)


def integrate_camera_as_open3d_vector(transform: np.ndarray, face_colors: tuple, scene_scale: float) -> o3d.utility.Vector3dVector:
    """
    Returns an Open3D Vector3dVector of a fake camera mesh's transformed vertices.

    Args:
        transform (np.ndarray): 4x4 transformation matrix for camera positioning.
        face_colors (tuple): Color of the camera face (unused here).
        scene_scale (float): Scale of the scene.

    Returns:
        o3d.utility.Vector3dVector: Transformed vertices as a Vector3dVector.
    """
    def create_cone_mesh(radius, height, sections=4):
        """
        Manually creates a low-poly cone as an Open3D TriangleMesh.
        The cone points along -Z axis. Returns o3d.geometry.TriangleMesh.
        """
        angles = np.linspace(0, 2 * np.pi, sections, endpoint=False)
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        z = np.zeros_like(x)

        # Base circle points
        base = np.stack([x, y, z], axis=1)

        # Tip of the cone
        tip = np.array([[0, 0, -height]])

        # All vertices
        vertices = np.vstack((base, tip))
        tip_index = len(vertices) - 1

        # Build faces (base to tip)
        triangles = []
        for i in range(sections):
            next_i = (i + 1) % sections
            triangles.append([i, next_i, tip_index])  # Side triangle

        # Base face (optional)
        center_index = len(vertices)
        vertices = np.vstack([vertices, [[0, 0, 0]]])  # base center
        for i in range(sections):
            next_i = (i + 1) % sections
            triangles.append([i, center_index, next_i])  # Base triangle

        # Convert to Open3D TriangleMesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.compute_vertex_normals()
        return mesh

    cam_width = scene_scale * 0.05
    cam_height = scene_scale * 0.1

    # Create cone mesh
    base_mesh = create_cone_mesh(cam_width, cam_height, sections=4)

    # Apply rotations and transformations
    rot_45 = np.eye(4)
    rot_45[:3, :3] = Rotation.from_euler("z", 45, degrees=True).as_matrix()
    rot_45[2, 3] = -cam_height  # move the cone tip backwards

    opengl_transform = get_opengl_conversion_matrix()
    #complete_transform = transform @ opengl_transform @ rot_45
    complete_transform = transform @ rot_45

    apply_transform_to_mesh(base_mesh, complete_transform)

    # Assign vertex colors
    color_normalized = np.array(face_colors) / 255.0
    colors = np.tile(color_normalized, (np.asarray(base_mesh.vertices).shape[0], 1))
    base_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    return base_mesh


def generate_colormap_colors(n, colormap='viridis'):
    """
    Generates N RGB colors from a matplotlib colormap.
    Output: list of RGB tuples in 0–255 range.
    """
    cmap = matplotlib.pyplot.get_cmap(colormap)
    return [tuple((np.array(cmap(i / max(n - 1, 1)))[:3] * 255).astype(int)) for i in range(n)]
