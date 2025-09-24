import numpy as np
import pycolmap


def estimate_depthmap(extrinsic: np.ndarray, pointmap: np.ndarray) -> np.array:
    h, w, c = pointmap.shape
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    Xw = pointmap.reshape(-1, 3)  # shape (H*W, 3)
    valid = ~np.isnan(Xw).any(axis=1) & (Xw != 0).any(axis=1)

    Xc = (R @ Xw[valid].T).T + t  # shape (N_valid, 3)
    Z = np.full((h * w,), np.nan, dtype=np.float32)
    Z[valid] = Xc[:, 2]

    depth_map = Z.reshape(h, w, 1)
    return depth_map


def bundle_adjustment(predictions: dict, *, max_num_iterations: int=100, step: int=2) -> dict:
    img_names = predictions['image_names']
    if len(img_names) != len(set(img_names)):
        raise ValueError("Repeated image")

    reconstruction = pycolmap.Reconstruction()
    n, c, h, w = predictions["images"].shape

    for i in range(n):
        intrinsic = predictions["intrinsic"][i]

        camera = pycolmap.Camera(
            model = 'PINHOLE',
            width = w,
            height = h,
            params = [intrinsic[0,0], intrinsic[1,1], intrinsic[0,2], intrinsic[1,2]],
            camera_id = i + 1
        )

        reconstruction.add_camera(camera)
    
    for i, extrinsic in enumerate(predictions["extrinsic"]):
        full_extrinsic = np.eye(4)
        full_extrinsic[:3, :] = extrinsic

        # set image
        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(full_extrinsic[:3, :3]),
            full_extrinsic[:3, 3]
        )  # Rot and Trans

        image = pycolmap.Image(
            id = i + 1,
            name = predictions["image_names"][i],
            camera_id = i + 1,
            cam_from_world = cam_from_world
        )
        reconstruction.add_image(image)
    
    point3D_dict = {}
    for i, point_map in enumerate(predictions["world_points_from_depth"]):
        color_img = predictions["images"][i]
        color_img = color_img.transpose(1, 2, 0)
        points2D = []
        for u in range(0, h, step):
            for v in range(0, w, step):
                point_world = point_map[u, v]
                key = tuple(round(coord, 5) for coord in point_world)

                if key in point3D_dict:
                    point3D_id = point3D_dict[key]
                else:
                    rgb = tuple((255*color_img[u, v]).astype(np.uint8))
                    point3D_id = reconstruction.add_point3D(
                        point_world,
                        pycolmap.Track(),
                        rgb
                    )
                    point3D_dict[key] = point3D_id
                
                point2d = pycolmap.Point2D([v, u], point3D_id)
                points2D.append(point2d)
        reconstruction.images[i + 1].points2D = points2D

    ba_options = pycolmap.BundleAdjustmentOptions()
    ba_options.solver_options.max_num_iterations = max_num_iterations
    pycolmap.bundle_adjustment(reconstruction, ba_options)

    ba_predictions = {}

    ba_predictions['extrinsic'] = np.array([
        image.cam_from_world.matrix()[:3]
        for image in list(reconstruction.images.values())[::-1]
    ])
    ba_predictions['intrinsic'] = []
    for camera in list(reconstruction.cameras.values())[::-1]:
        fx, fy, cx, cy = camera.params
        intrinsic = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]
        ])
        ba_predictions['intrinsic'].append(intrinsic)
    ba_predictions['intrinsic'] = np.asarray(ba_predictions['intrinsic'])
    ba_predictions['image_names'] = predictions['image_names']

    return ba_predictions
