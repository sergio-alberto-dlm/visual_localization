# scripts/vis_crocodl_rerun.py
# Visualize CrocoDL/LAMAR-style iOS sessions with Rerun.
# - Per-subsesseion image streams
# - Camera poses (if present)
# - Trajectory line strip (if trajectories.txt is present)
#
# Usage examples:
#   python scripts/vis_crocodl_rerun.py --scene /DATA/capture/HYDRO --split map --subsession ios_2023-10-27_10.20.23_000
#   python scripts/vis_crocodl_rerun.py --scene /DATA/capture/HYDRO --split map --subsession all --every 5
#   python scripts/vis_crocodl_rerun.py --scene /DATA/capture/HYDRO --split query --keyframes-only

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import rerun as rr
from dataset import make_transform

# Your dataset module from the project (the one we made earlier):
from dataset import IOSMapDataset, IOSQueryDataset

def _ts_to_seconds(ts: int) -> float:
    """
    Best-effort timestamp scaling for viewer timelines.
    CrocoDL/LAMAR iOS timestamps are often in *nanoseconds* relative-to-session.
    We default to ns -> s. If your time looks off, change scale here.
    """
    return float(ts) * 1e-9

def log_subsession_map(scene_dir: Path, subsession: str, every: int):
    tfm = make_transform()
    ds = IOSMapDataset(scene_dir, transform=tfm, subsession=subsession, return_image=True)
    # Group frames for this subsession (already filtered by ctor):
    frames = ds.samples
    frames.sort(key=lambda fr: fr.timestamp)

    # Build a trajectory if poses exist:
    traj_points = []
    for fr in frames:
        if fr.Twc is not None:
            p = fr.Twc[:3, 3].numpy()
            traj_points.append(p.tolist())

    # Log trajectory (one polyline per subsession)
    if len(traj_points) >= 2:
        rr.log(f"world/{subsession}/trajectory", rr.LineStrips3D([traj_points]))  # 3D lines. :contentReference[oaicite:1]{index=1}

    # Per-sensor intrinsics cache so we only log once (static)
    pinhole_done = set()

    for idx, fr in enumerate(frames):
        if idx % every:
            continue

        item = ds[idx]  # returns dict with image/K/Twc/Tcw/path/...
        ts = item["timestamp"]
        sensor_id = item["sensor_id"]
        img = (item["image"] * 255).permute(1, 2, 0).numpy()
        W, H, _ = img.shape
        K = item["K"].numpy()  # 3x3

        # Timelines: useful to scrub in the viewer
        # rr.set_time_seq("frame_idx", idx)                     # integer frame index  :contentReference[oaicite:2]{index=2}
        # rr.set_time_seconds("time", _ts_to_seconds(ts))            # seconds timeline     :contentReference[oaicite:3]{index=3}
        rr.set_time("stable_time", duration=_ts_to_seconds(ts))

        base = f"world/{subsession}/{sensor_id}"

        # Pose: camera (child) -> world (parent). Twc fits rr.Transform3D logging.  :contentReference[oaicite:4]{index=4}
        if item.get("Twc") is not None:
            Twc = item["Twc"].numpy()
            R = Twc[:3, :3]
            t = Twc[:3, 3]
            rr.log(base, rr.Transform3D(mat3x3=R, translation=t))

        # Intrinsics (Pinhole) logged once per sensor as static data.
        # We pass the camera matrix as a PinholeProjection and resolution.  :contentReference[oaicite:5]{index=5}
        if sensor_id not in pinhole_done:
            rr.log(f"{base}/image", rr.Pinhole(image_from_camera=K, resolution=[W, H]), static=True)
            pinhole_done.add(sensor_id)

        # RGB image
        rr.log(f"world/{subsession}/image", rr.Image(np.array(img)))           # RR image archetype  :contentReference[oaicite:6]{index=6}

def log_subsession_query(scene_dir: Path, subsession: str, every: int, keyframes_only: bool):
    tfm = make_transform()
    ds = IOSQueryDataset(scene_dir, transform=tfm, subsession=subsession, return_image=True, keyframes_only=keyframes_only)
    frames = ds.samples
    frames.sort(key=lambda fr: fr.timestamp)

    pinhole_done = set()

    for idx, fr in enumerate(frames):
        if idx % every:
            continue

        item = ds[idx]
        ts = item["timestamp"]
        sensor_id = item["sensor_id"]
        img: Image.Image = item["image"]
        W, H = img.size
        K = item["K"].numpy()

        rr.set_time_sequence("frame_idx", idx)                     # :contentReference[oaicite:7]{index=7}
        rr.set_time_seconds("time", _ts_to_seconds(ts))            # :contentReference[oaicite:8]{index=8}

        base = f"world/{subsession}/{sensor_id}"

        if sensor_id not in pinhole_done:
            rr.log(f"{base}/image", rr.Pinhole(image_from_camera=K, resolution=[W, H]), static=True)  # :contentReference[oaicite:9]{index=9}
            pinhole_done.add(sensor_id)

        rr.log(f"{base}/image", rr.Image(np.array(img)))           # :contentReference[oaicite:10]{index=10}

def main():
    ap = argparse.ArgumentParser(description="Rerun viewer for CrocoDL/LAMAR-style iOS captures.")
    ap.add_argument("--scene", type=Path, required=True,
                    help="Path to a single capture/scene directory containing ios_map and/or ios_query.")
    ap.add_argument("--split", choices=["map", "query"], default="map",
                    help="Which split to visualize.")
    ap.add_argument("--subsession", default="all",
                    help="Subsession to visualize (e.g., ios_2023-10-27_10.20.23_000). Use 'all' to iterate all.")
    ap.add_argument("--every", type=int, default=1,
                    help="Log every Nth frame (for speed).")
    ap.add_argument("--keyframes-only", action="store_true",
                    help="(query only) visualize only pruned keyframes.")
    ap.add_argument("--spawn", action="store_true",
                    help="Spawn the Rerun viewer automatically.")
    args = ap.parse_args()

    rr.init(f"CrocoDL Rerun • {args.split}", spawn=args.spawn)

    # Discover subsessions:
    if args.split == "map":
        idx = IOSMapDataset(args.scene).idx
    else:
        idx = IOSQueryDataset(args.scene, keyframes_only=args.keyframes_only).idx

    subs = idx.subsessions if args.subsession == "all" else [args.subsession]
    missing = [s for s in subs if s not in idx.subsessions]
    if missing:
        raise SystemExit(f"Unknown subsession(s): {missing}. Available: {idx.subsessions}")

    for s in subs:
        rr.log(f"world/{s}", rr.TextLog(f"Subsession: {s}"))
        if args.split == "map":
            log_subsession_map(args.scene, s, args.every)
        else:
            log_subsession_query(args.scene, s, args.every, args.keyframes_only)

    # Optional: keep the script alive if you didn't use --spawn
    # rr.spin()

if __name__ == "__main__":
    main()
