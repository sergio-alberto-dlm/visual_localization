from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable
import numpy as np 
import torchvision.transforms as transforms 

import torch
from torch.utils.data import Dataset
from PIL import Image

from utils.pose_utils import quat_to_R, invert_se3

# --------------------------- standard image transf ---------------------------
def make_transform(resize_size: int = 224):
    to_tensor = transforms.ToTensor()
    resize = transforms.Resize((resize_size, resize_size), antialias=True)
    normalize = transforms.Normalize(
        mean=(0.430, 0.411, 0.296),
        std=(0.213, 0.156, 0.143),
    )
    return transforms.Compose([to_tensor, resize, normalize])


# --------------------------- small parsing helpers ---------------------------

def _read_rows(txt: Path) -> List[List[str]]:
    rows = []
    if not txt.exists():
        return rows
    for line in txt.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        rows.append(parts)
    return rows

# ---------------------------------- models -----------------------------------

@dataclass(frozen=True)
class CameraModel:
    sensor_id: str
    name: str
    sensor_type: str
    model: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def K(self) -> torch.Tensor:
        K = torch.eye(3, dtype=torch.float64)
        K[0, 0] = self.fx; K[1, 1] = self.fy
        K[0, 2] = self.cx; K[1, 2] = self.cy
        return K

@dataclass
class FrameRec:
    split: str                 # "map" or "query"
    subsession: str            # ios_2023-..._000
    timestamp: int
    sensor_id: str
    rel_image: str
    abs_image: Path
    Twc: Optional[torch.Tensor] = None  # 4x4
    Tcw: Optional[torch.Tensor] = None  # 4x4
    cov6x6: Optional[torch.Tensor] = None

# ------------------------------- base indexer --------------------------------

class _SplitIndex:
    def __init__(self, split_root: Path, split_name: str):
        self.root = split_root
        self.split = split_name                       # "map" or "query"
        self.proc = split_root / "proc"
        self.raw = split_root / "raw_data"

        self.subsessions: List[str] = []
        self.sensors: Dict[str, CameraModel] = {}
        self.frames_by_ts: Dict[int, FrameRec] = {}
        self.frames_by_sub: Dict[str, List[FrameRec]] = {}
        self.keyframe_ts: Optional[set[int]] = None   # query only

        self._parse_all()

    def _parse_all(self):
        # subsessions
        self.subsessions = [r[0] for r in _read_rows(self.proc / "subsessions.txt") if r]
        # keyframes (query)
        if self.split == "query":
            rows = _read_rows(self.proc / "keyframes_pruned_subsampled.txt")
            if rows:
                self.keyframe_ts = {int(r[0]) for r in rows if r}

        # sensors
        for r in _read_rows(self.root / "sensors.txt"):
            if len(r) < 10: continue
            cam = CameraModel(
                sensor_id=r[0], name=r[1], sensor_type=r[2], model=r[3],
                width=int(r[4]), height=int(r[5]),
                fx=float(r[6]), fy=float(r[7]), cx=float(r[8]), cy=float(r[9]),
            )
            self.sensors[cam.sensor_id] = cam

        # images
        for r in _read_rows(self.root / "images.txt"):
            if len(r) < 3: continue
            ts = int(r[0]); sensor_id = r[1]; rel = r[2]
            subsession = Path(rel).parts[0]
            abs_path = self.raw / rel
            fr = FrameRec(split=self.split, subsession=subsession, timestamp=ts,
                          sensor_id=sensor_id, rel_image=rel, abs_image=abs_path)
            self.frames_by_ts[ts] = fr
            self.frames_by_sub.setdefault(subsession, []).append(fr)

        for sub in self.frames_by_sub:
            self.frames_by_sub[sub].sort(key=lambda f: f.timestamp)

        # trajectories (map only)
        if (self.root / "trajectories.txt").exists():
            for r in _read_rows(self.root / "trajectories.txt"):
                if len(r) < 9: continue
                ts = int(r[0])
                qw, qx, qy, qz = map(float, r[2:6])
                tx, ty, tz = map(float, r[6:9])
                R = quat_to_R(qw, qx, qy, qz)
                Twc = torch.eye(4, dtype=torch.float64)
                Twc[:3, :3] = R; Twc[:3, 3] = torch.tensor([tx, ty, tz], dtype=torch.float64)
                Tcw = invert_se3(Twc)
                cov = None
                if len(r) >= 9 + 36:
                    cov = torch.tensor(list(map(float, r[9:9+36])),
                                       dtype=torch.float64).reshape(6, 6)
                if ts in self.frames_by_ts:
                    fr = self.frames_by_ts[ts]
                    fr.Twc = Twc; fr.Tcw = Tcw; fr.cov6x6 = cov

    # queries
    def frames(self, subsession: Optional[str] = None, keyframes_only=False) -> List[FrameRec]:
        if subsession is not None:
            frs = list(self.frames_by_sub.get(subsession, []))
        else:
            frs = [f for sub in self.subsessions for f in self.frames_by_sub.get(sub, [])]
        if keyframes_only and self.keyframe_ts is not None:
            frs = [f for f in frs if f.timestamp in self.keyframe_ts]
        return frs

# ------------------------------ user-facing API -------------------------------

class IOSBase(Dataset):
    """Shared utilities for map/query datasets."""
    def __init__(self, root: str | Path, split: str,
                transform=None, return_image=True, keyframes_only=False):
        root = Path(root)
        assert split in ("map", "query")
        self.root = root
        self.split = split
        self.transform = transform
        self.return_image = return_image

        self.idx = _SplitIndex(root / f"ios_{split}", split)
        self.samples: List[FrameRec] = self.idx.frames(keyframes_only=keyframes_only)

    # --- common getters ---
    def __len__(self): return len(self.samples)

    def _load_image(self, path: Path):
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            return self.transform(img)
        # default: CHW float tensor in [0,1]
        return torch.from_numpy(np.array(img)).permute(2,0,1).float()/255.0

    def _cam_K(self, sensor_id: str) -> torch.Tensor:
        return self.idx.sensors[sensor_id].K()

    def _base_item(self, fr: FrameRec) -> dict:
        item = {
            "split": fr.split,
            "subsession": fr.subsession,
            "timestamp": fr.timestamp,
            "sensor_id": fr.sensor_id,
            "path": str(fr.abs_image),
            "K": self._cam_K(fr.sensor_id),   # 3x3 float64
        }
        if self.return_image:
            item["image"] = self._load_image(fr.abs_image)
        return item

class IOSMapDataset(IOSBase):
    """Yields frames with poses (for mapping / BA / refinement)."""
    def __init__(self, root: str | Path, transform=None, return_image=True,
                 subsession: Optional[str]=None):
        super().__init__(root, "map", transform, return_image, keyframes_only=False)
        if subsession is not None:
            self.samples = self.idx.frames(subsession=subsession)

    def __getitem__(self, i: int) -> dict:
        fr = self.samples[i]
        d = self._base_item(fr)
        d["Twc"] = None if fr.Twc is None else fr.Twc.clone()
        d["Tcw"] = None if fr.Tcw is None else fr.Tcw.clone()
        d["cov6x6"] = None if fr.cov6x6 is None else fr.cov6x6.clone()
        return d

class IOSQueryDataset(IOSBase):
    """Yields query frames; `keyframes_only=True` to match your list."""
    def __init__(self, root: str | Path, transform=None, return_image=True,
                 keyframes_only=True, subsession: Optional[str]=None):
        super().__init__(root, "query", transform, return_image, keyframes_only)
        if subsession is not None:
            self.samples = self.idx.frames(subsession=subsession, keyframes_only=keyframes_only)

    def __getitem__(self, i: int) -> dict:
        fr = self.samples[i]
        return self._base_item(fr)

# --------------------------- sequential pair dataset --------------------------

class IOSNeighborPairs(Dataset):
    """
    Produces (i, j) pairs from a split, restricted to same subsession,
    with a positive stride (e.g., (t, t+stride)). Useful for VO / relative pose.
    """
    def __init__(self, map_ds: IOSMapDataset, stride: int = 1):
        assert isinstance(map_ds, IOSMapDataset)
        self.map_ds = map_ds
        self.stride = stride
        # Precompute indices grouped by subsession
        self.groups: Dict[str, List[int]] = {}
        for idx, fr in enumerate(map_ds.samples):
            self.groups.setdefault(fr.subsession, []).append(idx)
        # Build pair index list
        self.pairs: List[Tuple[int,int]] = []
        for g in self.groups.values():
            for k in range(len(g) - stride):
                i, j = g[k], g[k + stride]
                self.pairs.append((i, j))

    def __len__(self): return len(self.pairs)

    def __getitem__(self, p: int) -> dict:
        i, j = self.pairs[p]
        a = self.map_ds[i]
        b = self.map_ds[j]
        # relative pose T_ab = T_aw * T_wb
        Twc_a = a["Twc"]; Twc_b = b["Twc"]
        Tcw_a = a["Tcw"]; Tcw_b = b["Tcw"]
        Tab = None; Tba = None
        if Twc_a is not None and Twc_b is not None:
            # camera-to-world; we want T_ab (a->b) in camera frame a
            # T_ab = T_ac = Tcw_a @ Twc_b
            Tab = Tcw_a @ Twc_b
            Tba = Tcw_b @ Twc_a
        return {
            "a": a, "b": b,
            "Tab": Tab, "Tba": Tba,
            "stride": self.stride,
        }

# ------------------------------- collate helpers ------------------------------

def collate_map(batch: List[dict]) -> dict:
    """Stacks images and Ks; keeps SE(3) as tensors or None lists."""
    out = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch]
        if k == "image" and isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals, dim=0)
        elif k == "K" and isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals, dim=0)  # (B,3,3)
        elif k in ("Twc","Tcw","cov6x6"):
            out[k] = vals  # keep list of tensors/None
        else:
            out[k] = vals
    return out

def collate_query(batch: List[dict]) -> dict:
    out = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch]
        if k == "image" and isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals, dim=0)
        elif k == "K" and isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals, dim=0)
        else:
            out[k] = vals
    return out

# ------------------------------- quick usage ---------------------------------

def build_retrieval_loaders(
    root: str | Path,
    transform=None,
    batch_size: int = 8,
    num_workers: int = 4,
    keyframes_only: bool = True,
):
    """Returns (query_loader, map_loader) for feature extraction / retrieval."""
    from torch.utils.data import DataLoader
    qds = IOSQueryDataset(root, transform=transform, keyframes_only=keyframes_only)
    mds = IOSMapDataset(root, transform=transform)
    qloader = DataLoader(qds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, collate_fn=collate_query, pin_memory=True)
    mloader = DataLoader(mds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, collate_fn=collate_map, pin_memory=True)
    return qloader, mloader
