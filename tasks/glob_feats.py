#!/usr/bin/env python3
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset import make_transform

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def default_num_workers() -> int:
    try:
        import multiprocessing as mp
        return max(1, mp.cpu_count() // 2)
    except Exception:
        return 4

def to_list(x):
    # Turn tensor/list/tuple/scalar into a python list (keeps strings)
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]

def get_field(batch: Dict[str, Any], key: str, B: int) -> List[Any]:
    """Safely fetch a field from the collated batch and return list length B."""
    v = batch.get(key, None)
    if v is None:
        return ["" for _ in range(B)]
    lst = to_list(v)
    if len(lst) == B:
        return lst
    # Some collate setups pack strings/paths as list already; ensure length B
    if isinstance(v, list) and len(v) == B:
        return v
    # Fallback
    return [v for _ in range(B)]

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Extract DINOv3 (ViT-B/16) features from ios_map using IOSMapDataset, saving ONE .npy per subsession."
    )
    p.add_argument("--repo_dir", default="submodules/dinov3",
                   help="Local dinov3 repo for torch.hub.load")
    p.add_argument("--weights", 
                   default="/media/jbhayet/Data/datasets/croco-experiments/checkpoints/dinov3_vitb16.pth",
                   help="Path to dinov3_vitb16.pth")
    p.add_argument("--sessions_root", required=True,
                   help="Path to sessions/ (contains ios_map/ and ios_query/)")
    p.add_argument("--out_dir", required=True,
                   help="Output directory (one folder per subsession)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=default_num_workers())
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no_amp", action="store_true", help="Disable AMP")
    p.add_argument("--normalize", action="store_true", help="L2-normalize embeddings")
    p.add_argument("--dtype", choices=["float32", "float16"], default="float32",
                   help="Data type used when saving embeddings")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip subsessions whose output .npy already exists")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Perf knobs
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    # ---- Import your dataset pieces from your project ----
    try:
        from utils.dataset import IOSMapDataset, collate_map  # ensure PYTHONPATH if needed
    except Exception as e:
        print("[ERROR] Could not import IOSMapDataset/collate_map from dataset.py")
        print("Make sure this script runs from your project root or set PYTHONPATH.")
        print("Original error:", repr(e))
        raise

    # ---- Dataset / DataLoader (ios_map split) ----
    sessions_root = Path(args.sessions_root)
    ios_map_root = sessions_root / "ios_map"
    if not ios_map_root.exists():
        raise FileNotFoundError(f"ios_map folder not found under {sessions_root}")

    tfm = make_transform()
    ds = IOSMapDataset(str(sessions_root), transform=tfm)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        collate_fn=collate_map
    )

    # ---- Model ----
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    #model = torch.hub.load(
    #    args.repo_dir, "dinov3_vitb16", source="local", weights=args.weights
    #).to(device)
    model = torch.hub.load("gmberton/MegaLoc", "get_trained_model").to(device)
    model.eval()

    use_amp = (device.type == "cuda") and (not args.no_amp)
    amp_dtype = torch.float16

    # ---- Accumulators per subsession ----
    # subsession -> list[np.ndarray], and aligned metadata lists
    feat_accum: Dict[str, List[np.ndarray]] = {}
    meta_accum: Dict[str, Dict[str, List[Any]]] = {}  # {ss: {"timestamp":[], "sensor_id":[], "image_path":[]}}

    # If skipping existing, mark the subsessions to ignore
    skip_ss = set()
    if args.skip_existing:
        for ss_dir in (p for p in ios_map_root.joinpath("raw_data").glob("*") if p.is_dir()):
            out_ss_dir = out_dir / ss_dir.name
            npy_path = out_ss_dir / f"{ss_dir.name}.npy"
            if npy_path.exists():
                skip_ss.add(ss_dir.name)

    # ---- Pass over the dataset once ----
    total_batches = None
    if hasattr(ds, "__len__"):
        total_batches = max(1, (len(ds) + args.batch_size - 1) // args.batch_size)

    pbar = tqdm(loader, total=total_batches, ncols=100, desc="Extracting DINOv3")
    with torch.inference_mode():
        for batch in pbar:
            imgs = batch["image"].to(device, non_blocking=True)  # (B,3,H,W)
            B = imgs.shape[0]

            subsessions = get_field(batch, "subsession", B)
            timestamps  = get_field(batch, "timestamp", B)
            sensor_ids  = get_field(batch, "sensor_id", B)
            image_paths = get_field(batch, "path", B)

            # Forward
            if use_amp:
                with torch.cuda.amp.autocast(dtype=amp_dtype):
                    feats = model(imgs)  # (B, D)
            else:
                feats = model(imgs)

            if args.normalize:
                feats = torch.nn.functional.normalize(feats, dim=-1)

            feats = feats.detach().cpu()
            if args.dtype == "float16":
                feats = feats.to(torch.float16)

            feats_np = feats.numpy()  # (B, D)

            # Split rows by subsession
            for i in range(B):
                ss = str(subsessions[i]) if subsessions[i] is not None else "unknown"
                if ss in skip_ss:
                    continue

                if ss not in feat_accum:
                    feat_accum[ss] = []
                    meta_accum[ss] = {"timestamp": [], "sensor_id": [], "image_path": []}

                feat_accum[ss].append(feats_np[i:i+1])  # keep row shape for cheap concat
                meta_accum[ss]["timestamp"].append(str(timestamps[i]))
                meta_accum[ss]["sensor_id"].append(str(sensor_ids[i]))
                meta_accum[ss]["image_path"].append(str(image_paths[i]))

    # ---- Write one .npy + manifest.csv per subsession ----
    for ss, chunks in feat_accum.items():
        ss_dir = out_dir / ss
        ss_dir.mkdir(parents=True, exist_ok=True)
        npy_path = ss_dir / f"{ss}.npy"
        manifest_path = ss_dir / "manifest.csv"

        features = np.concatenate(chunks, axis=0) if len(chunks) > 0 else np.empty((0, 0), dtype=np.float32)
        np.save(npy_path, features)

        # Manifest aligned with row order in .npy
        with open(manifest_path, "w", encoding="utf-8", newline="") as f:
            import csv
            w = csv.writer(f)
            w.writerow(["row_index", "timestamp", "sensor_id", "image_path"])
            for i, (ts, sid, ip) in enumerate(zip(
                meta_accum[ss]["timestamp"],
                meta_accum[ss]["sensor_id"],
                meta_accum[ss]["image_path"]
            )):
                w.writerow([i, ts, sid, ip])

        print(f"[OK] {ss}: saved {features.shape} -> {npy_path}")

    print(f"\n[DONE] Wrote per-subsession features under: {out_dir}")

if __name__ == "__main__":
    main()
