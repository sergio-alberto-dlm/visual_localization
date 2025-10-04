#!/usr/bin/env python3
# utils/vis_crocodl.py
# Visualize CrocoDL/LAMAR-style multi-device sessions (iOS / HoloLens / Spot) with Rerun.
#
# Highlights:
# - Per-subsesseion, per-device, per-sensor image streams
# - Camera poses & trajectory line strip (for *map* splits if trajectories exist)
# - Optional: visualize "retrieve queries" as pinhole cameras from a trajectories.txt
#
# Usage examples:
#   python scripts/vis_crocodl_rerun.py --scene /DATA/capture/HYDRO --split map --device ios --subsession ios_2023-10-27_10.20.23_000
#   python scripts/vis_crocodl_rerun.py --scene /DATA/capture/HYDRO --split map --device hl  --subsession all --every 5
#   python scripts/vis_crocodl_rerun.py --scene /DATA/capture/HYDRO --split query --device ios --keyframes-only
#   python scripts/vis_crocodl_rerun.py --scene /DATA/capture/HYDRO --split map --device ios --vis-retrieve-queries --queries-pose /path/to/queries_trajectories.txt
#
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torchvision.transforms as T
import numpy as np
import rerun as rr
from PIL import Image

# --- dataset module -------------------------------------

from dataset import IOSMapDataset, IOSQueryDataset
from dataset import HLMapDataset, HLQueryDataset
from dataset import SpotMapDataset, SpotQueryDataset
try:
    from pose_utils import parse_trajectories_txt, _q_to_R  # user-provided module
except Exception:
    parse_trajectories_txt = None
    _q_to_R = None

# -----------------------------------------------------------------------------

def _ts_to_seconds(ts: int) -> float:
    """
    Best-effort timestamp scaling for viewer timelines.
    KITTI/CrocoDL/LAMAR timestamps are often in nanoseconds relative-to-session.
    """
    return float(ts) * 1e-9

def _device_datasets(device: str, split: str):
    """
    Return (MapDatasetClass, QueryDatasetClass) for a given device type.
    """
    device = device.lower()
    assert device in ("ios", "hl", "spot"), f"Unsupported device: {device}"

    if device == "ios":
        return IOSMapDataset, IOSQueryDataset
    if device == "hl":
        return HLMapDataset, HLQueryDataset
    if device == "spot":
        return SpotMapDataset, SpotQueryDataset

def _auto_pick_device(scene_dir: Path, split: str) -> str:
    """
    Heuristic: pick device by existing split folder names.
    """
    candidates = []
    if (scene_dir / f"ios_{split}").exists(): candidates.append("ios")
    if (scene_dir / f"hl_{split}").exists():  candidates.append("hl")
    if (scene_dir / f"spot_{split}").exists(): candidates.append("spot")

    if not candidates:
        raise SystemExit(
            f"Could not auto-detect device under {scene_dir}. "
            f"Looked for ios_{split}, hl_{split}, spot_{split}."
        )
    if len(candidates) > 1:
        # If you have multiple present, force the user to specify.
        raise SystemExit(
            f"Multiple devices present for split '{split}': {candidates}. "
            f"Please pass --device {{ios|hl|spot}}."
        )
    return candidates[0]

def _iter_subsessions(idx, subsession_arg: str) -> List[str]:
    """
    Normalize subsession selection.
    """
    subs = idx.subsessions if subsession_arg == "all" else [subsession_arg]
    missing = [s for s in subs if s not in idx.subsessions]
    if missing:
        raise SystemExit(f"Unknown subsession(s): {missing}. Available: {idx.subsessions}")
    return subs

# ---------- Logging helpers ---------------------------------------------------

def _log_trajectory_polyline(subsession: str, devices: List[dict]):
    traj = []
    for dev in devices:
        Twc = dev.get("Twc")
        if Twc is not None:
            p = Twc[:3, 3].cpu().numpy() if hasattr(Twc, "cpu") else Twc[:3, 3].numpy()
            traj.append(p.tolist())

    if len(traj) >= 2:
        rr.log(f"world/{subsession}/trajectory", rr.LineStrips3D([traj]))

def _ensure_numpy_image(tensor_img) -> np.ndarray:
    """
    dataset._base_item() packs images as CHW float tensor [0,1].
    Convert to HxWxC uint8 for Rerun.
    """
    if hasattr(tensor_img, "permute"):
        img = (tensor_img * 255.0).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        return img
    if isinstance(tensor_img, Image.Image):
        return np.array(tensor_img)
    if isinstance(tensor_img, np.ndarray):
        return tensor_img
    raise TypeError(f"Unsupported image type: {type(tensor_img)}")

def _log_device_sensors(
    subsession: str,
    item: dict,
    pinhole_logged: set,
    frame_idx: int,
    device: str,
):
    """
    Log all sensor images + intrinsics for a single device item
    """
    ts: int = item["timestamp"]
    dev_id: str = item["dev_id"]
    img_sensors: Dict[str, object] = item.get("img_sensor", {})

    # Use both a scrub-able sequence + a wall-clock-ish seconds axis
    rr.set_time("frame_idx", sequence=frame_idx)
    # rr.set_time_seconds("time", _ts_to_seconds(ts))

    base_dev = f"world/{subsession}/{dev_id}"

    # Pose (if available). Twc (world <- camera/device) aligns with Transform3D
    Twc = item.get("Twc")
    if Twc is not None:
        Twc_np = Twc.cpu().numpy() if hasattr(Twc, "cpu") else Twc.numpy()
        R = Twc_np[:3, :3]
        t = Twc_np[:3, 3]
        rr.log(base_dev, rr.Transform3D(mat3x3=R, translation=t))

    # Multi-sensor logging
    for sensor_id, s in img_sensors.items():
        # Image
        img = _ensure_numpy_image(s.img)
        Tws = s.Tws
        H, W = img.shape[:2]
        # Intrinsics
        K = s.K.cpu().numpy() if hasattr(s.K, "cpu") else (s.K.numpy() if hasattr(s.K, "numpy") else np.asarray(s.K))

        # base_sensor = f"{base_dev}/{sensor_id}"
        rr.log(f"{base_dev}", rr.Points3D(
            Tws[:3, 3].cpu().numpy() if hasattr(Tws, "cpu") else Tws[:3, 3].numpy(), radii=50.0
        ))
        # Pinhole logged once per (device_id, sensor_id)
        key = (dev_id, sensor_id)
        if key not in pinhole_logged:
            rr.log(f"{base_dev}", rr.Pinhole(image_from_camera=K, resolution=[W, H]), static=True)
            pinhole_logged.add(key)
            
        if device == "ios":
            rr.log(f"world/{subsession}/image", rr.Image(img))
        else:
            rr.log(f"world/{subsession}/{sensor_id}", rr.Image(img))

def _log_retrieve_queries(
    subsession: str,
    queries_pose_path: Path,
    default_resolution: Tuple[int, int],
    default_K: np.ndarray,
):
    """
    Log "retrieve queries" as pinhole cameras with poses from a trajectories.txt
    """
    if parse_trajectories_txt is None or _q_to_R is None:
        rr.log(
            f"world/{subsession}/retrieve_queries",
            rr.TextLog("Cannot log retrieve queries: pose_utils helpers not found."),
        )
        return

    q_poses = parse_trajectories_txt(queries_pose_path)

    for q in q_poses:
        dev_id = q.device_id  # expected "subsession/.../..."
        base = f"world/{subsession}/retrieve_queries/{dev_id}"

        R = _q_to_R(q.q)  # 3x3
        t = q.t           # (3,)
        rr.log(base, rr.Transform3D(mat3x3=R, translation=t))
        W, H = default_resolution
        rr.log(base, rr.Pinhole(image_from_camera=default_K, resolution=[W, H]), static=True)

# ---------- Per-split drivers -------------------------------------------------

def log_subsession_map(
    scene_dir: Path,
    subsession: str,
    every: int,
    device: str,
    vis_retrieve_queries: bool,
    queries_pose: Optional[Path],
):
    tfm = T.Compose([T.Resize((480, 640)), T.ToTensor()])
    MapDS, _ = _device_datasets(device, "map")
    ds = MapDS(scene_dir, transform=tfm, subsession=subsession, return_image=True)

    # Sorted by timestamp already by your indexer; sorting again for safety:
    ds.samples.sort(key=lambda d: d.timestamp)

    # Build a quick "aligned view" of items to:
    #  - plot a trajectory polyline
    #  - feed the sensor logger
    items = [ds[i] for i in range(len(ds))]
    _log_trajectory_polyline(subsession, items)

    pinhole_logged: set = set()
    for i, item in enumerate(items):
        if every > 1 and (i % every != 0):
            continue
        _log_device_sensors(subsession, item, pinhole_logged, frame_idx=i, device=device)

    # If we should log retrieve queries, pick a sane default K/resolution from any sensor seen above.
    if vis_retrieve_queries and queries_pose is not None:
        # Try to reuse the latest sensor's intrinsics/resolution as default:
        fallback_K = None
        fallback_res = None
        for it in items:
            sensors = it.get("img_sensor", {})
            if sensors:
                sid, s = next(iter(sensors.items()))
                img = _ensure_numpy_image(s.img)
                H, W = img.shape[:2]
                K = s.K.cpu().numpy() if hasattr(s.K, "cpu") else (s.K.numpy() if hasattr(s.K, "numpy") else np.asarray(s.K))
                fallback_res = (W, H)
                fallback_K = K
                break
        if fallback_K is None:
            # Last resort:
            fallback_K = np.array([[500., 0., 320.],
                                   [0., 500., 240.],
                                   [0.,   0.,   1.]], dtype=float)
            fallback_res = (640, 480)
        _log_retrieve_queries(subsession, queries_pose, fallback_res, fallback_K)

def log_subsession_query(
    scene_dir: Path,
    subsession: str,
    every: int,
    keyframes_only: bool,
    device: str,
):
    tfm = make_transform()
    _, QueryDS = _device_datasets(device, "query")
    ds = QueryDS(scene_dir, transform=tfm, subsession=subsession, return_image=True, keyframes_only=keyframes_only)
    ds.samples.sort(key=lambda d: d.timestamp)

    pinhole_logged: set = set()
    for i in range(len(ds)):
        if every > 1 and (i % every != 0):
            continue
        item = ds[i]  # has the same img_sensor structure as map
        _log_device_sensors(subsession, item, pinhole_logged, frame_idx=i)

# ---------- Main --------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Rerun viewer for multi-device CrocoDL/LAMAR-style captures.")
    ap.add_argument("--scene", type=Path, required=True,
                    help="Path to a single capture/scene containing *_{map,query} subfolders per device.")
    ap.add_argument("--split", choices=["map", "query"], default="map",
                    help="Split to visualize.")
    ap.add_argument("--device", choices=["ios", "hl", "spot", "auto"], default="auto",
                    help="Device type (auto-detect if unique).")
    ap.add_argument("--subsession", default="all",
                    help="Subsession to visualize. Use 'all' to iterate all.")
    ap.add_argument("--every", type=int, default=1,
                    help="Log every Nth frame (>=1).")
    ap.add_argument("--keyframes-only", action="store_true",
                    help="(query only) visualize only pruned keyframes.")
    ap.add_argument("--spawn", action="store_true",
                    help="Spawn the Rerun viewer automatically.")
    ap.add_argument("--vis-retrieve-queries", action="store_true",
                    help="If provided, visualize retrieve queries (poses as pinhole cameras).")
    ap.add_argument("--queries-pose", type=Path, default=None,
                    help="Path to trajectories.txt of retrieve queries.")
    args = ap.parse_args()

    if args.every < 1:
        raise SystemExit("--every must be >= 1")

    dev = args.device if args.device != "auto" else _auto_pick_device(args.scene, args.split)

    rr.init(f"CrocoDL Rerun • {args.split} • {dev}", spawn=args.spawn)

    # Build an indexer from the right dataset to discover subsessions
    if args.split == "map":
        MapDS, _ = _device_datasets(dev, "map")
        idx = MapDS(args.scene).idx
    else:
        _, QueryDS = _device_datasets(dev, "query")
        idx = QueryDS(args.scene, keyframes_only=args.keyframes_only).idx

    subs = _iter_subsessions(idx, args.subsession)

    for s in subs:
        rr.log(f"world/{s}", rr.TextLog(f"Subsession: {s}"))
        if args.split == "map":
            log_subsession_map(
                scene_dir=args.scene,
                subsession=s,
                every=args.every,
                device=dev,
                vis_retrieve_queries=args.vis_retrieve_queries,
                queries_pose=args.queries_pose,
            )
        else:
            log_subsession_query(
                scene_dir=args.scene,
                subsession=s,
                every=args.every,
                keyframes_only=args.keyframes_only,
                device=dev,
            )

    # Keep alive if not spawning:
    # rr.spin()

if __name__ == "__main__":
    main()
