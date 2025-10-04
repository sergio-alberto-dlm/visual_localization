#!/usr/bin/env python3
from __future__ import annotations
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import csv
import numpy as np
import torch
from tqdm import tqdm

from utils.dataset import make_transform, IOSMapDataset, HLMapDataset, SpotMapDataset

# ------------------------------ Utilities ------------------------------------

def pick_map_dataset(device_type: str):
    d = device_type.lower()
    if d == "ios":  return IOSMapDataset
    if d == "hl":   return HLMapDataset
    if d == "spot": return SpotMapDataset
    raise ValueError(f"Unknown device type: {device_type}. Use ios|hl|spot|auto")

def auto_detect_device(scene_dir: Path) -> str:
    cands = []
    if (scene_dir / "ios_map").exists():  cands.append("ios")
    if (scene_dir / "hl_map").exists():   cands.append("hl")
    if (scene_dir / "spot_map").exists(): cands.append("spot")
    if not cands:
        raise SystemExit(f"No *map* split under {scene_dir} (expected ios_map/ hl_map/ spot_map/).")
    if len(cands) > 1:
        raise SystemExit(f"Multiple devices found {cands}. Please pass --device ios|hl|spot.")
    return cands[0]

def default_num_workers() -> int:
    try:
        import multiprocessing as mp
        return max(1, mp.cpu_count() // 2)
    except Exception:
        return 4

# ------------------------------ Core logic -----------------------------------

@torch.inference_mode()
def extract_one_subsession(
    MapDS,
    sessions_root: Path,
    subsession: str,
    model,
    device: torch.device,
    batch_size: int,
    normalize: bool,
    save_dtype: str,
) -> Tuple[np.ndarray, List[Tuple[str, str, int, str]]]:
    """
    Returns:
      features: (N, D)
      manifest_rows: list of (device_id, sensor_id, timestamp, abs_path)
    """
    ds = MapDS(sessions_root, transform=make_transform(), subsession=subsession, return_image=True)
    # stable order
    ds.samples.sort(key=lambda d: (d.timestamp, d.device_id))

    # Gather rows = per-sensor images across ALL device records in this subsession
    rows: List[Tuple[str, str, int, str, torch.Tensor]] = []
    seen: set = set()  # dedup guard on (device_id,sensor_id,timestamp,abs_path)

    for i in range(len(ds)):
        item = ds[i]
        dev_id = item["dev_id"]
        ts     = int(item["timestamp"])
        sensors = item.get("img_sensor", {})
        for sensor_id, s in sensors.items():
            abs_path = str(s.abs_path)
            key = (dev_id, sensor_id, ts, abs_path)
            if key in seen:
                continue
            seen.add(key)
            rows.append((dev_id, sensor_id, ts, abs_path, s.img))  # CHW float tensor [0,1]

    if not rows:
        # empty subsession
        return np.empty((0, 0), dtype=np.float32), []

    # Forward in batches
    feats_chunks: List[np.ndarray] = []
    manifest_rows: List[Tuple[str, str, int, str]] = []

    use_amp = (device.type == "cuda")
    amp_dtype = torch.float16

    for start in tqdm(range(0, len(rows), batch_size), desc=f"{subsession}", ncols=100):
        batch = rows[start:start + batch_size]
        imgs = torch.stack([r[4] for r in batch], dim=0).to(device, non_blocking=True)

        if use_amp:
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                f = model(imgs)  # (b, D)
        else:
            f = model(imgs)

        if normalize:
            f = torch.nn.functional.normalize(f, dim=-1)

        if save_dtype == "float16":
            f = f.to(torch.float16)

        feats_np = f.detach().cpu().numpy()
        feats_chunks.append(feats_np)

        for (dev_id, sensor_id, ts, abs_path, _img) in batch:
            manifest_rows.append((dev_id, sensor_id, ts, abs_path))

    feats = np.concatenate(feats_chunks, axis=0)

    # Ensure on-disk dtype is as requested
    if save_dtype == "float16" and feats.dtype != np.float16:
        feats = feats.astype(np.float16, copy=False)
    elif save_dtype == "float32" and feats.dtype != np.float32:
        feats = feats.astype(np.float32, copy=False)

    return feats, manifest_rows

def save_subsession(out_dir: Path, subsession: str, feats: np.ndarray, manifest_rows: List[Tuple[str, str, int, str]]):
    ss_dir = out_dir / subsession
    ss_dir.mkdir(parents=True, exist_ok=True)
    np.save(ss_dir / f"{subsession}.npy", feats)

    with open(ss_dir / "manifest.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_index", "device_id", "sensor_id", "timestamp", "abs_path"])
        for i, (dev_id, sensor_id, ts, abs_path) in enumerate(manifest_rows):
            w.writerow([i, dev_id, sensor_id, ts, abs_path])

# ------------------------------ Main -----------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Extract MegaLoc embeddings per *subsession* (map split). Rows are (device_id, sensor_id, timestamp, abs_path)."
    )
    ap.add_argument("--sessions_root", required=True, help="Scene root containing <device>_map/")
    ap.add_argument("--out_dir", required=True, help="Output root; will create <subsession>/<subsession>.npy and manifest.csv")
    ap.add_argument("--device", choices=["ios", "hl", "spot", "auto"], default="auto", help="Dataset device type")
    ap.add_argument("--subsession", default=None, help="If set, process only this subsession")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--compute_device", default="cuda:0", help="e.g., cuda:0 or cpu")
    ap.add_argument("--no_amp", action="store_true", help="(kept for compatibility; AMP toggled automatically on CUDA)")
    ap.add_argument("--normalize", action="store_true", help="L2-normalize embeddings")
    ap.add_argument("--dtype", choices=["float32", "float16"], default="float32", help="On-disk dtype")
    ap.add_argument("--skip_existing", action="store_true", help="Skip subsessions whose output already exists")
    args = ap.parse_args()

    sessions_root = Path(args.sessions_root)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Select dataset type
    dev_type = auto_detect_device(sessions_root) if args.device == "auto" else args.device
    MapDS = pick_map_dataset(dev_type)

    # Build a tiny dataset just to get the indexer & subsession list
    idx = MapDS(sessions_root).idx
    subs = idx.subsessions if args.subsession is None else [args.subsession]
    missing = [s for s in subs if s not in idx.subsessions]
    if missing:
        raise SystemExit(f"Unknown subsession(s): {missing}. Available: {idx.subsessions}")

    # Model
    compute_device = torch.device(args.compute_device if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    model = torch.hub.load("gmberton/MegaLoc", "get_trained_model").to(compute_device)
    model.eval()

    # Iterate subsessions
    for ss in subs:
        npy_path = out_root / ss / f"{ss}.npy"
        man_path = out_root / ss / "manifest.csv"
        if args.skip_existing and npy_path.exists() and man_path.exists():
            print(f"[SKIP] {ss} already exists.")
            continue

        feats, manifest_rows = extract_one_subsession(
            MapDS=MapDS,
            sessions_root=sessions_root,
            subsession=ss,
            model=model,
            device=compute_device,
            batch_size=args.batch_size,
            normalize=args.normalize,
            save_dtype=args.dtype,
        )
        save_subsession(out_root, ss, feats, manifest_rows)
        print(f"[OK] {ss}: saved {feats.shape} → {npy_path}")

    print(f"\n[DONE] Wrote per-subsession embeddings under: {out_root}")

if __name__ == "__main__":
    main()
