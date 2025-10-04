#!/usr/bin/env python3
from __future__ import annotations
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
from tqdm import tqdm

from utils.dataset import (
    make_transform,
    IOSQueryDataset, HLQueryDataset, SpotQueryDataset,
    IOSMapDataset,   HLMapDataset,   SpotMapDataset,
)

# --------------------------- Helpers ---------------------------

def pick_query_dataset(dev: str):
    dev = dev.lower()
    if dev == "ios":  return IOSQueryDataset
    if dev == "hl":   return HLQueryDataset
    if dev == "spot": return SpotQueryDataset
    raise ValueError(f"Unknown query device: {dev}")

def pick_map_dataset(dev: str):
    dev = dev.lower()
    if dev == "ios":  return IOSMapDataset
    if dev == "hl":   return HLMapDataset
    if dev == "spot": return SpotMapDataset
    raise ValueError(f"Unknown map device: {dev}")

def default_num_workers() -> int:
    try:
        import multiprocessing as mp
        return max(1, mp.cpu_count() // 2)
    except Exception:
        return 4

def l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n

def read_manifest(manifest_path: Path) -> List[Dict[str, str]]:
    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def build_faiss_ip_index(X: np.ndarray, use_gpu: bool = False):
    try:
        import faiss  # type: ignore
    except Exception:
        return None, None
    d = X.shape[1]
    index = faiss.IndexFlatIP(d)
    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(X.astype(np.float32, copy=False))
    return index, faiss

def cosine_topk_chunked(
    Q: np.ndarray, X: np.ndarray, topk: int, device: str = "cuda:0", chunk: int = 100_000
) -> Tuple[np.ndarray, np.ndarray]:
    """Chunked cosine/IP search without FAISS. Assumes l2-normalized inputs."""
    use_cuda = torch.cuda.is_available() and device.startswith("cuda")
    dev = torch.device(device if use_cuda else "cpu")
    Q_t = torch.from_numpy(Q.astype(np.float32, copy=False)).to(dev, non_blocking=True)
    N = X.shape[0]
    B = Q_t.shape[0]
    scores_all = torch.full((B, topk), -1e9, device=dev)
    idx_all = torch.full((B, topk), -1, dtype=torch.int64, device=dev)

    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        Xc = torch.from_numpy(X[start:end].astype(np.float32, copy=False)).to(dev, non_blocking=True)
        sims = Q_t @ Xc.t()
        vals, inds = torch.topk(sims, k=min(topk, end - start), dim=1, largest=True, sorted=False)
        inds = inds + start
        cat_scores = torch.cat([scores_all, vals], dim=1)
        cat_idx = torch.cat([idx_all, inds], dim=1)
        vals2, inds2 = torch.topk(cat_scores, k=topk, dim=1, largest=True, sorted=True)
        idx2 = torch.gather(cat_idx, 1, inds2)
        scores_all, idx_all = vals2, idx2
        del Xc, sims, vals, inds, cat_scores, cat_idx, vals2, inds2, idx2
        if use_cuda:
            torch.cuda.synchronize()
    return scores_all.detach().cpu().numpy(), idx_all.detach().cpu().numpy()

# ------------------ Load MAP embeddings + meta ------------------

def load_map_features_all(
    map_feats_root: Path,
    normalize: bool = True,
    dtype: str = "float32",
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Expects per-subsession files produced by your extractor:
      <root>/<subsession>/<subsession>.npy
      <root>/<subsession>/manifest.csv  (row_index,device_id,sensor_id,timestamp,abs_path)
    Returns:
      X_map: (N, D) embeddings
      meta:  list of dicts aligned to rows, each with:
             {'subsession','device_id','sensor_id','timestamp','abs_path'}
    """
    all_feats = []
    all_meta: List[Dict[str, Any]] = []
    dtype_np = np.float16 if dtype == "float16" else np.float32

    subdirs = sorted([p for p in map_feats_root.iterdir() if p.is_dir()])
    D = None
    for ss_dir in subdirs:
        ss = ss_dir.name
        npy_path = ss_dir / f"{ss}.npy"
        man_path = ss_dir / "manifest.csv"
        if not npy_path.exists() or not man_path.exists():
            continue
        feats = np.load(npy_path, mmap_mode="r")
        if feats.ndim != 2:
            print(f"[WARN] {npy_path} not 2D; skip.")
            continue
        if D is None:
            D = feats.shape[1]
        elif feats.shape[1] != D:
            print(f"[WARN] Dim mismatch in {npy_path}; got {feats.shape[1]} vs {D}. Skip.")
            continue
        rows = read_manifest(man_path)
        n = min(len(rows), feats.shape[0])

        feats = np.asarray(feats[:n], dtype=dtype_np)
        if normalize:
            feats = l2_normalize_np(feats)

        for k in range(n):
            r = rows[k]
            all_meta.append({
                "subsession": ss,
                "device_id": r.get("device_id", ""),
                "sensor_id": r.get("sensor_id", ""),
                "timestamp": r.get("timestamp", ""),
                "abs_path": r.get("abs_path", r.get("image_path", "")),
            })
        all_feats.append(feats)
        print(f"[MAP] Loaded {ss}: {feats.shape}")

    if not all_feats:
        raise RuntimeError(f"No map features under {map_feats_root}")
    X_map = np.concatenate(all_feats, axis=0)
    if len(all_meta) != X_map.shape[0]:
        all_meta = all_meta[:X_map.shape[0]]
    print(f"[MAP] Total: {X_map.shape}")
    return X_map, all_meta

# ---------------- Build Tws(world←sensor) for MAP ----------------

def build_Tws_lookup_for_map(
    sessions_root: Path,
    map_device: str,
) -> Dict[Tuple[str, str, int, str], List[List[float]]]:
    """
    (subsession, sensor_id, timestamp, abs_path) -> Tws (4x4 list).
    Tws = Twc @ Trs per sensor, read from MAP dataset (no image loading).
    """
    MapDS = pick_map_dataset(map_device)
    ds = MapDS(sessions_root, transform=None, return_image=False)
    ds.samples.sort(key=lambda d: (d.subsession, d.timestamp, d.device_id))
    Tws: Dict[Tuple[str, str, int, str], List[List[float]]] = {}
    for dev in ds.samples:
        if dev.Twc is None:
            continue
        ss = dev.subsession
        ts = int(dev.timestamp)
        Twc = dev.Twc
        for sid, srow in dev.sensor_items.items():
            Trs = srow.Trs
            Tws_mat = (Twc @ Trs).cpu().numpy() if hasattr(Twc, "cpu") else (Twc @ Trs).numpy()
            key = (ss, sid, ts, str(srow.abs_path))
            Tws[key] = Tws_mat.tolist()
    print(f"[MAP] Tws entries: {len(Tws)}")
    return Tws

# ------------------------------- Main -------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Cross-device multi-sensor retrieval with MegaLoc (cosine)."
    )
    ap.add_argument("--sessions_root", required=True,
                    help="Scene root containing <device>_{map,query} for ios/hl/spot.")
    ap.add_argument("--map_feats_root", required=True,
                    help="Root of MAP embeddings for the selected MAP device "
                         "(e.g., .../embeddings/megaloc/ios or hl or spot).")
    ap.add_argument("--map_device",   choices=["ios", "hl", "spot"], required=True,
                    help="Device type for MAP split.")
    ap.add_argument("--query_device", choices=["ios", "hl", "spot"], required=True,
                    help="Device type for QUERY split.")
    ap.add_argument("--out_json", default=None,
                    help="Optional explicit output json path; default is {scene}_map_{MAP}_query_{Q}.json")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--normalize", action="store_true",
                    help="L2-normalize both map and query embeddings (recommended).")
    ap.add_argument("--use_faiss", action="store_true",
                    help="Use FAISS IndexFlatIP if available.")
    ap.add_argument("--faiss_gpu", action="store_true",
                    help="If using FAISS, put index on GPU 0.")
    ap.add_argument("--compute_device", default="cuda:0")
    args = ap.parse_args()

    sessions_root = Path(args.sessions_root)
    map_feats_root = Path(args.map_feats_root)

    scene_name = args.sessions_root.split("/")[-2] #sessions_root.name
    if args.out_json is None:
        out_name = f"{scene_name}_map_{args.map_device}_query_{args.query_device}.json"
        out_path = Path(out_name)
    else:
        out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ----- Load MAP features + metadata -----
    X_map, map_meta = load_map_features_all(
        map_feats_root=map_feats_root,
        normalize=args.normalize,
        dtype="float32",
    )

    # ----- Tws(world←sensor) lookup from MAP -----
    Tws_lookup = build_Tws_lookup_for_map(sessions_root, args.map_device)

    # ----- Build FAISS / fallback -----
    index = None
    if args.use_faiss:
        index, _ = build_faiss_ip_index(X_map, use_gpu=args.faiss_gpu)
        if index is None:
            print("[INFO] FAISS not available; falling back to chunked cosine.")

    # ----- QUERY dataset + MegaLoc model -----
    QueryDS = pick_query_dataset(args.query_device)
    tfm = make_transform()
    qds = QueryDS(sessions_root, transform=tfm, keyframes_only=True, return_image=True)  # keyframes subsampled
    qds.samples.sort(key=lambda d: (d.subsession, d.timestamp, d.device_id))

    device = torch.device(args.compute_device if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    megaloc = torch.hub.load("gmberton/MegaLoc", "get_trained_model").to(device)
    megaloc.eval()
    use_amp = (device.type == "cuda")
    amp_dtype = torch.float16

    # ----- Retrieval loop (multi-sensor per query device record) -----
    results: List[Dict[str, Any]] = []

    for qi in tqdm(range(len(qds)), desc=f"Queries • {args.query_device}→{args.map_device}", ncols=100):
        qitem = qds[qi]
        q_ss = qitem["subsession"]
        q_dev = qitem["dev_id"]
        q_ts = int(qitem["timestamp"])
        sensors: Dict[str, Any] = qitem.get("img_sensor", {})

        for q_sid, s in sensors.items():
            q_img_t: torch.Tensor = s.img      # CHW float [0,1]
            q_abs   = str(s.abs_path)
            q_Trs   = s.Trs.cpu().numpy().tolist() if hasattr(s.Trs, "cpu") else s.Trs.numpy().tolist()

            # 1× forward
            imgs = q_img_t.unsqueeze(0).to(device, non_blocking=True)
            if use_amp:
                with torch.cuda.amp.autocast(dtype=amp_dtype):
                    q_feat = megaloc(imgs)
            else:
                q_feat = megaloc(imgs)
            if args.normalize:
                q_feat = torch.nn.functional.normalize(q_feat, dim=-1)
            q_np = q_feat.detach().cpu().to(torch.float32).numpy()  # (1, D)

            # search
            if index is not None:
                Dists, Idxs = index.search(q_np.astype(np.float32, copy=False), args.topk)
                scores = Dists[0]
                inds   = Idxs[0]
            else:
                scores, inds = cosine_topk_chunked(q_np, X_map, topk=args.topk, device=args.compute_device)
                scores = scores[0]; inds = inds[0]

            # Build topk list (include Tws from MAP lookup)
            hits: List[Dict[str, Any]] = []
            for rank, jj in enumerate(inds, start=1):
                if jj < 0 or jj >= len(map_meta):
                    continue
                m = map_meta[jj]
                key = (m["subsession"], m["sensor_id"], int(m["timestamp"]), m["abs_path"])
                Tws = Tws_lookup.get(key, None)
                hits.append({
                    "rank": rank,
                    "score": float(scores[rank-1]),
                    "subsession": m["subsession"],
                    "timestamp": str(m["timestamp"]),
                    "sensor_id": m["sensor_id"],
                    "path": m["abs_path"],
                    "Tws": Tws,   # 4x4 or null
                })

            results.append({
                "path": q_abs,
                "subsession": q_ss,
                "device_id": q_dev,
                "sensor_id": q_sid,
                "timestamp": str(q_ts),
                "Trs": q_Trs,         # 4x4 (device/rig←sensor) for the QUERY item
                "topk": hits,
            })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[DONE] Wrote: {out_path}  ({len(results)} query-sensor entries)")

if __name__ == "__main__":
    main()
