import os 
import csv 
import math 
import torch 
import numpy as np 
from collections import namedtuple

def quat_to_R(qw: float, qx: float, qy: float, qz: float) -> torch.Tensor:
    q = torch.tensor([qw, qx, qy, qz], dtype=torch.float64)
    q = q / torch.linalg.norm(q)
    w, x, y, z = q
    R = torch.tensor([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),   2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),   1 - 2*(x*x + y*y)],
    ], dtype=torch.float64)
    return R

def invert_se3(T: torch.Tensor) -> torch.Tensor:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = torch.eye(4, dtype=T.dtype).to(T.device)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti

def invert_se3_np(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=T.dtype)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti

# ------------------------------
# Small math/SE(3)/quaternion utils
# ------------------------------
def _q_normalize(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n == 0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n

def _q_conj(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=np.float64)

def _q_mul(q2, q1):
    # Hamilton product: q = q2 * q1
    w2, x2, y2, z2 = q2
    w1, x1, y1, z1 = q1
    return np.array([
        w2*w1 - x2*x1 - y2*y1 - z2*z1,
        w2*x1 + x2*w1 + y2*z1 - z2*y1,
        w2*y1 - x2*z1 + y2*w1 + z2*x1,
        w2*z1 + x2*y1 - y2*x1 + z2*w1
    ], dtype=np.float64)

def _q_from_R(R):
    # Robust-ish extraction
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2,1] - R[1,2]) / s
        y = (R[0,2] - R[2,0]) / s
        z = (R[1,0] - R[0,1]) / s
    else:
        i = int(np.argmax([R[0,0], R[1,1], R[2,2]]))
        if i == 0:
            s = math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
            w = (R[2,1] - R[1,2]) / s
            x = 0.25 * s
            y = (R[0,1] + R[1,0]) / s
            z = (R[0,2] + R[2,0]) / s
        elif i == 1:
            s = math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
            w = (R[0,2] - R[2,0]) / s
            x = (R[0,1] + R[1,0]) / s
            y = 0.25 * s
            z = (R[1,2] + R[2,1]) / s
        else:
            s = math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
            w = (R[1,0] - R[0,1]) / s
            x = (R[0,2] + R[2,0]) / s
            y = (R[1,2] + R[2,1]) / s
            z = 0.25 * s
    return _q_normalize([w,x,y,z])

def _q_to_R(q):
    w, x, y, z = _q_normalize(q)
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    R = np.array([
        [1-2*(yy+zz),   2*(xy-wz),     2*(xz+wy)],
        [2*(xy+wz),     1-2*(xx+zz),   2*(yz-wx)],
        [2*(xz-wy),     2*(yz+wx),     1-2*(xx+yy)]
    ], dtype=np.float64)
    return R

def _RT_to_T(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T

def _compose(Ta, Tb):
    return Ta @ Tb

def _angle_deg_between_quats(q1, q2):
    # relative rotation angle (0..180) in degrees
    q1 = _q_normalize(q1); q2 = _q_normalize(q2)
    q_rel = _q_mul(q2, _q_conj(q1))
    w = np.clip(abs(q_rel[0]), 0.0, 1.0)
    angle_rad = 2.0 * math.acos(w)
    return math.degrees(angle_rad)

# ------------------------------
# Parsers for capture-style files
# ------------------------------
ImageRow   = namedtuple("ImageRow",   ["timestamp", "sensor_id", "image_path"])
RigRow     = namedtuple("RigRow",     ["rig_id", "sensor_id", "q", "t"])
TrajRow    = namedtuple("TrajRow",    ["timestamp", "device_id", "q", "t"])

def _read_csv_like(path, expected_min_cols=1):
    rows = []
    with open(path, "r", newline="") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # allow both comma+spaces and comma
            parts = [p.strip() for p in next(csv.reader([line]))]
            if len(parts) < expected_min_cols:
                continue
            rows.append(parts)
    return rows

def parse_images_txt(path):
    # "# timestamp, sensor_id, image_path"
    rows = _read_csv_like(path, expected_min_cols=3)
    out = []
    for ts, sensor_id, img_path in rows:
        out.append(ImageRow(int(ts), sensor_id, img_path))
    return out

def parse_rigs_txt(path):
    # "# rig_id, sensor_id, qw, qx, qy, qz, tx, ty, tz"
    if not os.path.exists(path):
        return []
    rows = _read_csv_like(path, expected_min_cols=9)
    out = []
    for r in rows:
        rig_id, sensor_id = r[0], r[1]
        qw, qx, qy, qz = map(float, r[2:6])
        tx, ty, tz = map(float, r[6:9])
        out.append(RigRow(rig_id, sensor_id, _q_normalize([qw,qx,qy,qz]), np.array([tx,ty,tz], dtype=np.float64)))
    return out

def parse_trajectories_txt(path):
    # "# timestamp, device_id, qw, qx, qy, qz, tx, ty, tz, *covar"
    rows = _read_csv_like(path, expected_min_cols=9)
    out = []
    for r in rows:
        ts = int(r[0]); device_id = r[1]
        qw, qx, qy, qz = map(float, r[2:6])
        tx, ty, tz     = map(float, r[6:9])
        out.append(TrajRow(ts, device_id, _q_normalize([qw,qx,qy,qz]), np.array([tx,ty,tz], dtype=np.float64)))
    return out

def skew_sym_mat(x):
    device = x.device
    dtype = x.dtype
    ssm = torch.zeros(3, 3, device=device, dtype=dtype)
    ssm[0, 1] = -x[2]
    ssm[0, 2] = x[1]
    ssm[1, 0] = x[2]
    ssm[1, 2] = -x[0]
    ssm[2, 0] = -x[1]
    ssm[2, 1] = x[0]
    return ssm


def SO3_exp(theta):
    device = theta.device
    dtype = theta.dtype

    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    I = torch.eye(3, device=device, dtype=dtype)
    if angle < 1e-5:
        return I + W + 0.5 * W2
    else:
        return (
            I
            + (torch.sin(angle) / angle) * W
            + ((1 - torch.cos(angle)) / (angle**2)) * W2
        )


def V(theta):
    dtype = theta.dtype
    device = theta.device
    I = torch.eye(3, device=device, dtype=dtype)
    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    if angle < 1e-5:
        V = I + 0.5 * W + (1.0 / 6.0) * W2
    else:
        V = (
            I
            + W * ((1.0 - torch.cos(angle)) / (angle**2))
            + W2 * ((angle - torch.sin(angle)) / (angle**3))
        )
    return V


def SE3_exp(tau):
    dtype = tau.dtype
    device = tau.device

    rho = tau[:3]
    theta = tau[3:]
    R = SO3_exp(theta)
    t = V(theta) @ rho

    T = torch.eye(4, device=device, dtype=dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T