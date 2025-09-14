import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv 
import json 
import argparse
from pathlib import Path 
from typing import List, Dict, Tuple, Any 

import numpy as np
import torch 
from torch.utils.data import DataLoader

from utils.geo_utils import l2_normalize
from utils.system_utils import try_import_faiss, default_num_workers
from utils.dataset import make_transform

# -------------------------
# Utilities
# -------------------------

def read_manifest(manifest_path: Path) -> List[Dict[str, str]]:
    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows 

def load_map_features_all(
    map_feats_root: Path,  
    normalize: bool = True, 
    dtype: str = "float32", 
) -> Tuple[np.ndarray, list[Dict[str, Any]]]:
    f"""
        Load every subsession's embeddings (.npy) + manifest.csv and concatenates 
        into a single (N, D) array with aligned metadata list. 
        Returns: (X_map, metas) where metas[i] has {'subsession', 'abs_path', 'timestamp', 'sensor_id', 'rel_image_path'}
    """
    all_feats = []
    all_meta = []
    dtype_np = np.float16 if dtype == "float16" else np.float32

    # map_feats_root/
    #   subsessionA/subsessionA.npy
    #   subsessionA/manifest.csv
    #   subsessionB/subsessionB.npy
    #   ...
    subdirs = sorted([p for p in map_feats_root.iterdir() if p.is_dir()])
    total = 0 
    D = None 

    for ss_dir in subdirs:
        ss = ss_dir.name 
        npy_path = ss_dir / f"{ss}.npy"
        man_path = ss_dir / "manifest.csv"
        if not npy_path.exists() or not man_path.exists():
            continue

        #  load embeddings (allow big sets via memmap)
        feats = np.load(npy_path, mmap_mode="r")
        if D is None:
            D = feats.shape[1] if feats.ndim == 2 else 0
        elif feats.ndim == 2 and feats.shape[1] != D:
            print(f"[WARN] Skipping {npy_path} due to dim mismatch.")
            continue 

        # read manifest lines (row_index must align to rows)
        rows = read_manifest(man_path)
        if len(rows) != feats.shape[0]:
            print(f"[WARN] Rows mismatch in {ss} ({len(rows)} vs {feats.shape[0]}). Still attempting alignment.")

        for row in rows:
            all_meta.append({
                "subsession": ss, 
                "timestamp": row.get("timestamp", ""), 
                "sensor_id": row.get("sensor_id", ""),
                "abs_path": row.get("image_path", "")
            })

        # force materialize to desired dtype (keeps memory reasonable)
        feats = np.asarray(feats, dtype=dtype_np)
        if normalize:
            feats = l2_normalize(feats)

        all_feats.append(feats)
        total += feats.shape[0]
        print(f"[MAP] Loaded {ss}: {feats.shape}")

    if len(all_feats) == 0:
        raise RuntimeError(f"No map features found under {map_feats_root}")

    X_map = np.concatenate(all_feats, axis=0)
    if len(all_meta) != X_map.shape[0]:
        # if manifest were missing/misaligned, truncate metadat to match features 
        all_meta = all_meta[:X_map.shape[0]]
    print(f"[MAP] Total embeddings: {X_map.shape}")
    
    return X_map, all_meta

def build_faiss_ip_index(X: np.ndarray, use_gpu: bool = False):
    """
        Build an inner-product FAISS index (cosine if inputs are l2-normalized).
    """
    faiss = try_import_faiss()
    if faiss is None:
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
    """
        Fallback when FAISS not available. Assumes Q and X are L2-normalized.
        Computes IP = cosine with chunked matmul on GPU (if available).
        Returns (scores[B, K], idx[B, K]) as numpy float32/int64.
    """

    use_cuda = torch.cuda.is_available() and device.startswith("cuda")
    dev = torch.device(device if use_cuda else "cpu")

    Q_t = torch.from_numpy(Q.astype(np.float32, copy=False).to(dev, non_blocking=True))
    N = X.shape[0]
    scores_all = torch.full((Q_t.shape[0], topk), -1e9, device=dev)
    idx_all = torch.full((Q_t.shape[0], topk), -1, dtype=torch.int64, device=dev)

    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        X_chunk = torch.from_numpy(X[start:end].astype(np.float32, copy=False)).to(dev, non_blocking=True)
        # sim = Q @ X^T (B, D) -> (B, n)
        sims = Q_t @ X_chunk.t()

        # take topk for this chunk 
        vals, inds = torch.topk(sims, k=min(topk, end-start), dim=1, largest=True, sorted=False)
        inds += start # offset to global index 

        # merge with running topk 
        cat_scores = torch.cat([scores_all, vals], dim=1)
        cat_idx = torch.cat([idx_all, inds], dim=1)
        # final topk 
        vals2, inds2 = torch.topk(cat_scores, k=topk, dim=1, largest=True, sorted=True)
        idx2 = torch.gather(cat_idx, 1, inds2)
        scores_all, idx_all = vals2, idx2

        del X_chunk, sims, vals, inds, cat_scores, cat_idx, vals2, inds2, idx2
        if use_cuda:
            torch.cuda.synchronize()

        return scores_all.detach().cpu().numpy(), idx_all.detach().cpu().numpy()

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Retrieve top-K map images for each query image using cosine similarity.")
    ap.add_argument("--sessions_root", required=True,
                    help="Path to sessions/ containing ios_map/ and ios_query/")
    ap.add_argument("--map_feats_root", required=True,
                    help="Root with per-subsession features (the output of your extractor).")
    ap.add_argument("--repo_dir", default="submodules/dinov3",
                    help="Local DINOv3 repo for torch.hub.load")
    ap.add_argument("--weights", 
                    default="/media/jbhayet/Data/datasets/croco-experiments/checkpoints/dinov3_vitb16.pth",
                    help="Path to dinov3_vitb16.pth")
    ap.add_argument("--out_jsonl", required=True,
                    help="Path to write retrieval results (.jsonl)")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=default_num_workers())
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--normalize", action="store_true",
                    help="L2-normalize both map and query embeddings (recommended).")
    ap.add_argument("--use_faiss", action="store_true",
                    help="Use FAISS IndexFlatIP if available for speed.")
    ap.add_argument("--faiss_gpu", action="store_true",
                    help="If using FAISS, move index to GPU 0.")
    ap.add_argument("--dtype", choices=["float32", "float16"], default="float32",
                    help="Working dtype for map embeddings in RAM.")
    ap.add_argument("--keyframes_only", action="store_true",
                    help="Only evaluate query keyframes (dataset supports it).")
    args = ap.parse_args()

    sessions_root = Path(args.sessions_root)
    map_feats_root = Path(args.map_feats_root)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # -------------------
    # Load map embeddings
    # -------------------
    X_map, map_meta = load_map_features_all(
        map_feats_root=map_feats_root, 
        normalize=args.normalize, 
        dtype=args.dtype, 
    )

    # -------------------
    # Build FAISS index (optional)
    # -------------------
    index = None 
    faiss = None 
    if args.use_faiss:
        index, faiss = build_faiss_ip_index(X_map, use_gpu=args.faiss_gpu)
        if index is None:
            print("[INFO] FAISS not available; will use chunked cosine fallback.")

    # -------------------
    # Load query dataset / loader
    # -------------------
    try:
        from utils.dataset import IOSQueryDataset, collate_query
        from utils.dataset import make_transform
    except Exception as e:
        print("[ERROR] Could not import dataset pieces. Ensure dataset.py is importable.")
        raise 

    tfm = make_transform()

    qds = IOSQueryDataset(str(sessions_root), transform=tfm, keyframes_only=args.keyframes_only)
    qloader = DataLoader(
        qds, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.num_workers, pin_memory=True, 
        persistent_workers=(args.num_workers > 0),
        collate_fn=collate_query
    )

    # -------------------
    # Load DINOv3
    # -------------------
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dinov3 = torch.hub.load(
        args.repo_dir, "dinov3_vitb16", source="local", weights=args.weights
    ).to(device)
    dinov3.eval()

    use_amp = (device.type == "cuda")
    amp_dtype = torch.float16

    # -------------------
    # Retrieval loop
    # -------------------
    with open(out_path, "w", encoding="utf-8") as fout, torch.inference_mode():
        for batch in qloader:
            imgs: torch.Tensor = batch["image"].to(device, non_blocking=True) # (B, 3, H, W)
            B = imgs.shape[0]
            paths: List[str] = batch.get("path", [""] * B)
            timestamps = batch.get("timestamp", [""] * B)
            subsessions = batch.get("subsession", [""] * B)

            # extract query features 
            if use_amp:
                with torch.cuda.amp.autocast(dtype=amp_dtype):
                    q = dinov3(imgs)

            else:
                q = dinov3(imgs)

            if args.normalize:
                q = torch.nn.functional.normalize(q, dim=-1)

            q_np = q.detach().cpu().to(torch.float32).numpy()

            # search 
            if index is not None:
                Dists, Idxs = index.search(q_np.astype(np.float32, copy=False), args.topk)
                # FAISS returns inner product; with normalization that's cosine similarity 
                scores = Dists
                inds = Idxs
            else:
                scores, inds = cosine_topk_chunked(
                    Q=q_np, X=X_map, topk=args.topk, device=args.device, chunk=100_000
                )

            # write results (one JSON object per query)
            for i in range(B):
                hits = []
                for j in range(args.topk):
                    jj = int(inds[i, j])
                    if jj < 0 or jj >= len(map_meta):
                        continue 
                    m = map_meta[jj]
                    hits.append({
                        "rank": j + 1, 
                        "score": float(scores[i, j]),
                        "subsession": m["subsession"],
                        "timestamp": m["timestamp"], 
                        "sensor_id": m["sensor_id"], 
                        "abs_path": m["abs_path"], 
                        "map_index": jj, 
                    })

                rec = {
                    "query_path": paths[i] if isinstance(paths, list) else str(paths[i]), 
                    "query_subsession": subsessions[i] if isinstance(subsessions, list) else str(subsessions[i]), 
                    "query_timestamp": str(timestamps[i] if isinstance(timestamps, list) else timestamps[i]),
                    "topk": hits,
                }
                fout.write(json.dumps(rec) + "\n")

    print(f"[DONE] Wrote retrieval to {out_path}")
    print("Tip: `jq -r '.query_path, (.topk[]|.rank,.score,.abs_path)'", " < ", str(out_path))

if __name__ == "__main__":
    main()