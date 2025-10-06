from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple, Iterable
import numpy as np 
import torchvision.transforms as transforms 

import torch
from torch.utils.data import Dataset
from PIL import Image

from utils.pose_utils import quat_to_R, invert_se3
from collections import namedtuple

# --------------------------- standard image transf ---------------------------
def make_transform(resize_size: int = 322):
    to_tensor = transforms.ToTensor()
    resize = transforms.Resize((resize_size, resize_size), antialias=True)
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
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

PoseRow    = namedtuple("PoseRow", ["timestamp", "device_id", "Twc", "Tcw", "cov"])
SensorRow  = namedtuple("SensorRow", ["Trs", "Tsr", "abs_path", "rel_path", "K"])

# @dataclass
# class SensorItem:
#     image: torch.Tensor
#     Trs: torch.Tensor 
#     Tsr: torch.Tensor 
#     Tws: torch.Tensor 
#     Tsw: torch.Tensor 
#     abs_path: str
#     K: torch.Tensor

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
class DeviceRec:
    split: str                 # "map" or "query"
    subsession: str            # ios_2023-..._000
    timestamp: int
    device_id: str
    sensor_items: Dict[str, SensorRow]       # sensor_id->SensorRow
    Twc: Optional[torch.Tensor] = None  # 4x4, T_world->dev
    Tcw: Optional[torch.Tensor] = None  # 4x4, T_dev->world 
    cov6x6: Optional[torch.Tensor] = None

# ------------------------------- base indexer --------------------------------

class _SplitIndex:
    def __init__(self, split_root: Path, split_name: str, device_type: str):
        self.root = split_root
        self.split = split_name                       # "map" or "query"
        self.device_type = device_type
        self.proc = split_root / "proc"
        self.raw = split_root / "raw_data"

        self.subsessions: List[str] = []
        self.sensors: Dict[str, CameraModel] = {}
        # self.devices_by_ts: Dict[int, List[DeviceRec]] = {}
        self.devices_by_sub: Dict[str, List[DeviceRec]: []] = {}
        self.devices: Dict[Tuple(int, str): DeviceRec] = {} # (timestamp, subsession)->Device
        self.keyframe_ts: Optional[set[int]] = None   # query only
        self.rigs: Dict[Tuple[str, str], torch.Tensor] = {}  # (rig_id, sensor_id) -> Trs (4x4)
        self.poses_by_ts: Dict[int, PoseRow] = {} 

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

        # rigs (optional)
        self._parse_rigs(self.root / "rigs.txt")

        # images
        for r in _read_rows(self.root / "images.txt"):
            if len(r) < 3: continue
            ts = int(r[0]); sensor_id = r[1]; rel = r[2]
            subsession = Path(rel).parts[0]
            abs_path = self.raw / rel
            if self.device_type == "spot":
                device_id = f"{subsession}/{ts}-body"
            elif self.device_type == "hl":
                device_id = f"{subsession}/hetrig_{ts}"
            elif self.device_type == "ios":
                device_id = f"{subsession}/cam_phone_{ts}"

            if (ts, subsession) in self.devices:
                # if device already exists, add corresponding sensor image 
                dev = self.devices[(ts, subsession)]
                if self.rigs:
                    Trs = self.rigs[(device_id, sensor_id)]
                    Tsr = invert_se3(Trs)
                else:
                    Trs, Tsr = torch.eye(4, dtype=torch.float64), torch.eye(4, dtype=torch.float64)
                dev.sensor_items[sensor_id] = SensorRow(Trs, Tsr, abs_path, rel, self.sensors[sensor_id].K())
            else:
                # else create new device 
                if self.rigs:
                    Trs = self.rigs[(device_id, sensor_id)]
                    Tsr = invert_se3(Trs)
                else: 
                    Trs, Tsr = torch.eye(4, dtype=torch.float64), torch.eye(4, dtype=torch.float64)
                sensor_items = {sensor_id: SensorRow(Trs, Tsr, abs_path, rel, self.sensors[sensor_id].K())}

                dev = DeviceRec(split=self.split, subsession=subsession, timestamp=ts,
                        device_id=device_id, sensor_items=sensor_items)
                self.devices[(ts, subsession)] = dev 
                
        # filter device by subsession 
        for item in self.devices:
            ts, subsession = item 
            dev = self.devices[(ts, subsession)]
            self.devices_by_sub.setdefault(subsession, []).append(dev)

        for sub in self.devices_by_sub:
            self.devices_by_sub[sub].sort(key=lambda d: d.timestamp)

        # trajectories (map only)
        if (self.root / "trajectories.txt").exists():
            row_traj = _read_rows(self.root / "trajectories.txt")
            for r in row_traj:
                if len(r) < 9: continue
                ts = int(r[0])
                dev_id = r[1]
                subsession = dev_id.split("/")[0]
                qw, qx, qy, qz = map(float, r[2:6])
                tx, ty, tz = map(float, r[6:9])
                R = quat_to_R(qw, qx, qy, qz)
                Twc = torch.eye(4, dtype=torch.float64)
                Twc[:3, :3] = R; Twc[:3, 3] = torch.tensor([tx, ty, tz], dtype=torch.float64)
                Tcw = invert_se3(Twc)
                cov6x6 = None
                if len(r) >= 9 + 36:
                    cov6x6 = torch.tensor(list(map(float, r[9:9+36])),
                                       dtype=torch.float64).reshape(6, 6)
                dev = self.devices[(ts, subsession)]
                # assert dev_id == dev.device_id, f"dev_id: {dev_id} of pose different from device_id {dev.device_id} record"
                dev.Twc = Twc
                dev.Tcw = Tcw 
                dev.cov6x6 = cov6x6

    def _parse_rigs(self, rigs_txt: Path):
        rows = _read_rows(rigs_txt)
        if len(rows) > 0:
            for r in rows:
                # # rig_id, sensor_id, qw, qx, qy, qz, tx, ty, tz
                if len(r) < 9:
                    continue
                rig_id, sensor_id = r[0], r[1]
                qw, qx, qy, qz = map(float, r[2:6])
                tx, ty, tz = map(float, r[6:9])
                R = quat_to_R(qw, qx, qy, qz)
                Trs = torch.eye(4, dtype=torch.float64)
                Trs[:3, :3] = R
                Trs[:3, 3] = torch.tensor([tx, ty, tz], dtype=torch.float64)
                self.rigs[(rig_id, sensor_id)] = Trs

    # queries
    def devices_fn(self, subsession: Optional[str] = None, keyframes_only=False) -> List[DeviceRec]:
        if subsession is not None:
            devs = list(self.devices_by_sub.get(subsession, []))
        else:
            devs = [f for sub in self.subsessions for f in self.devices_by_sub.get(sub, [])]
        if keyframes_only and self.keyframe_ts is not None:
            devs = [f for f in devs if f.timestamp in self.keyframe_ts]
        return devs 

# ------------------------------ user-facing API -------------------------------

# ------------------------------ IOS datasets ----------------------------------

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

        self.idx = _SplitIndex(root / f"ios_{split}", split, device_type="ios")
        self.samples: List[DeviceRec] = self.idx.devices_fn(keyframes_only=keyframes_only)

    # --- common getters ---
    def __len__(self): return len(self.samples)

    def _load_image(self, path: Path):
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            return self.transform(img)
        # default: CHW float tensor in [0,1]
        return torch.from_numpy(np.array(img)).permute(2,0,1).float()/255.0

    # def _cam_K(self, sensor_id: str) -> torch.Tensor:
    #     return self.idx.sensors[sensor_id].K()

    def _base_item(self, dev: DeviceRec) -> dict:
        item = {
            "split": dev.split,
            "subsession": dev.subsession,
            "timestamp": dev.timestamp,
            "dev_id": dev.device_id,
        }
        if self.return_image:
            images_sensors: Dict[str, SimpleNamespace] = {}
            for sensor_id in dev.sensor_items:
                sensor_item = dev.sensor_items[sensor_id]
                images_sensors[sensor_id] = SimpleNamespace(
                    img = self._load_image(sensor_item.abs_path),  
                    Trs = sensor_item.Trs, 
                    Tsr = sensor_item.Tsr,
                    abs_path = sensor_item.abs_path,
                    K = sensor_item.K,
                )
            item["img_sensor"] = images_sensors

        return item

class IOSMapDataset(IOSBase):
    """Yields frames with poses (for mapping / BA / refinement)."""
    def __init__(self, root: str | Path, transform=None, return_image=True,
                subsession: Optional[str]=None):
        super().__init__(root, "map", transform, return_image, keyframes_only=False)
        if subsession is not None:
            self.samples = self.idx.devices_fn(subsession=subsession)

    def __getitem__(self, i: int) -> dict:
        dev = self.samples[i]
        d = self._base_item(dev)
        d["Twc"] = None if dev.Twc is None else dev.Twc.clone()
        d["Tcw"] = None if dev.Tcw is None else dev.Tcw.clone()
        d["cov6x6"] = None if dev.cov6x6 is None else dev.cov6x6.clone()

        for sensor_id in d["img_sensor"]:
            sensor_item = d["img_sensor"][sensor_id]
            sensor_item.Tws = dev.Twc @ sensor_item.Trs
            sensor_item.Tsw = sensor_item.Tsr @ dev.Tcw 
        return d

class IOSQueryDataset(IOSBase):
    """Yields query frames; `keyframes_only=True` to match your list."""
    def __init__(self, root: str | Path, transform=None, return_image=True,
                keyframes_only=True, subsession: Optional[str]=None):
        super().__init__(root, "query", transform, return_image, keyframes_only)
        if subsession is not None:
            self.samples = self.idx.devices_fn(subsession=subsession, keyframes_only=keyframes_only)

    def __getitem__(self, i: int) -> dict:
        dev = self.samples[i]
        return self._base_item(dev)

# ------------------------------ HL datasets ----------------------------------

class HLBase(IOSBase):
    def __init__(self, root: str | Path, split: str,
                 transform=None, return_image=True, keyframes_only=False):
        root = Path(root)
        assert split in ("map", "query")
        self.root = root
        self.split = split
        self.transform = transform
        self.return_image = return_image

        self.idx = _SplitIndex(root / f"hl_{split}", split, device_type="hl")
        self.samples: List[DeviceRec] = self.idx.devices_fn(keyframes_only=keyframes_only)

class HLMapDataset(HLBase):
    def __init__(self, root, transform=None, return_image=True, subsession: Optional[str]=None):
        super().__init__(root, "map", transform, return_image, keyframes_only=False)
        if subsession is not None:
            self.samples = self.idx.devices_fn(subsession=subsession)

    def __getitem__(self, i: int) -> dict:
        dev = self.samples[i]
        d = self._base_item(dev)
        d["Twc"] = None if dev.Twc is None else dev.Twc.clone()
        d["Tcw"] = None if dev.Tcw is None else dev.Tcw.clone()
        d["cov6x6"] = None if dev.cov6x6 is None else dev.cov6x6.clone()

        for sensor_id in d["img_sensor"]:
            sensor_item = d["img_sensor"][sensor_id]
            sensor_item.Tws = dev.Twc @ sensor_item.Trs
            sensor_item.Tsw = sensor_item.Tsr @ dev.Tcw 
        return d

class HLQueryDataset(HLBase):
    def __init__(self, root, transform=None, return_image=True,
                 keyframes_only=True, subsession: Optional[str]=None):
        super().__init__(root, "query", transform, return_image, keyframes_only)
        if subsession is not None:
            self.samples = self.idx.devices_fn(subsession=subsession, keyframes_only=keyframes_only)

    def __getitem__(self, i: int) -> dict:
        return self._base_item(self.samples[i])


# ------------------------------ Spot datasets --------------------------------

class SpotBase(IOSBase):
    def __init__(self, root: str | Path, split: str,
                 transform=None, return_image=True, keyframes_only=False):
        root = Path(root)
        assert split in ("map", "query")
        self.root = root
        self.split = split
        self.transform = transform
        self.return_image = return_image

        self.idx = _SplitIndex(root / f"spot_{split}", split, device_type="spot")
        self.samples: List[DeviceRec] = self.idx.devices_fn(keyframes_only=keyframes_only)

class SpotMapDataset(SpotBase):
    def __init__(self, root, transform=None, return_image=True, subsession: Optional[str]=None):
        super().__init__(root, "map", transform, return_image, keyframes_only=False)
        if subsession is not None:
            self.samples = self.idx.devices_fn(subsession=subsession)

    def __getitem__(self, i: int) -> dict:
        dev = self.samples[i]
        d = self._base_item(dev)
        d["Twc"] = None if dev.Twc is None else dev.Twc.clone()
        d["Tcw"] = None if dev.Tcw is None else dev.Tcw.clone()
        d["cov6x6"] = None if dev.cov6x6 is None else dev.cov6x6.clone()

        for sensor_id in d["img_sensor"]:
            sensor_item = d["img_sensor"][sensor_id]
            sensor_item.Tws = dev.Twc @ sensor_item.Trs
            sensor_item.Tsw = sensor_item.Tsr @ dev.Tcw 
        return d

class SpotQueryDataset(SpotBase):
    def __init__(self, root, transform=None, return_image=True,
                 keyframes_only=True, subsession: Optional[str]=None):
        super().__init__(root, "query", transform, return_image, keyframes_only)
        if subsession is not None:
            self.samples = self.idx.devices_fn(subsession=subsession, keyframes_only=keyframes_only)

    def __getitem__(self, i: int) -> dict:
        return self._base_item(self.samples[i])

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
    device: str = "ios",   # "ios" | "hl" | "spot"
):
    from torch.utils.data import DataLoader
    ds_map = {"ios": IOSMapDataset, "hl": HLMapDataset, "spot": SpotMapDataset}[device]
    ds_qry = {"ios": IOSQueryDataset, "hl": HLQueryDataset, "spot": SpotQueryDataset}[device]

    qds = ds_qry(root, transform=transform, keyframes_only=keyframes_only)
    mds = ds_map(root, transform=transform)

    qloader = DataLoader(qds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, collate_fn=collate_query, pin_memory=True)
    mloader = DataLoader(mds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, collate_fn=collate_map, pin_memory=True)
    return qloader, mloader
