import os

import open3d as o3d

from vggt_ba import CudaInference as Vggt, to_point_cloud, bundle_adjustment


img_paths = [
    'images/frame_000071.jpg',
    'images/frame_000087.jpg'
]
if img_paths == []:
    RuntimeWarning("Include image paths")
for path in img_paths:
    if not os.path.exists(path):
        RuntimeWarning("Images don't exist")


def main():
    print("Loading model")
    vggt = Vggt()
    print("Running inference")
    predictions = vggt.run(img_paths)
    print("Rinning bundle adjsutment to enhance camera properties estimation")
    ba_results = bundle_adjustment(predictions)

    print(ba_results)

    #Optional, show point clouds
    pcd = to_point_cloud(predictions)
    o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    main()
