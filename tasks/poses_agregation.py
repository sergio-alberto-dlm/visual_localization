import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import json
import pandas as pd 
import numpy as np 
import torch 
import csv

from utils.pose_utils import invert_se3_np, _q_from_R

import torch
from typing import Literal, Tuple

# -------------------- SO(3)/SE(3) math (Torch, tiny-angle safe) --------------------

def _so3_hat(w: torch.Tensor) -> torch.Tensor:
    """(...,3) -> (...,3,3)"""
    wx, wy, wz = w.unbind(-1)
    O = torch.zeros_like(wx)
    return torch.stack([
        torch.stack([ O, -wz,  wy], dim=-1),
        torch.stack([ wz,  O, -wx], dim=-1),
        torch.stack([-wy, wx,  O], dim=-1)
    ], dim=-2)

def so3_exp(w: torch.Tensor) -> torch.Tensor:
    """(...,3) axis-angle -> (...,3,3) rotation."""
    th = torch.linalg.norm(w, dim=-1, keepdims=True)  # (...,1)
    small = (th.squeeze(-1) < 1e-8)[..., None, None]  # (...,1,1)
    k = torch.where(small, torch.zeros_like(w), w / th)
    K = _so3_hat(k)

    I = torch.eye(3, dtype=w.dtype, device=w.device).expand(*w.shape[:-1], 3, 3)
    sin_th = torch.sin(th)[..., None]
    cos_th = torch.cos(th)[..., None]

    R = I + sin_th * K + (1.0 - cos_th) * (K @ K)
    R = torch.where(small, I, R)
    return R

def so3_log(R: torch.Tensor) -> torch.Tensor:
    """(...,3,3) -> (...,3) axis-angle (tiny-angle safe)."""
    tr = (R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]).clamp(-1.0, 3.0)
    cos_th = (tr - 1.0) * 0.5
    th = torch.acos(cos_th.clamp(-1.0, 1.0))                 # (...,)

    w_skew = (R - R.transpose(-1, -2)) * 0.5                  # (...,3,3)
    v = torch.stack([w_skew[..., 2, 1], w_skew[..., 0, 2], w_skew[..., 1, 0]], dim=-1)  # (...,3)

    small = (th < 1e-8)[..., None]  # (...,1)
    s = torch.sin(th).clamp(min=1e-12)[..., None]             # (...,1)
    # general: w = th/(2 sin th) * vee(R - R^T)
    w = (th[..., None] / (2.0 * s)) * v                       # (...,3)

    # ensure exact shape (...,3) in both branches
    w = torch.where(small, v, w)
    return w


def se3_exp(xi: torch.Tensor) -> torch.Tensor:
    """(…,6) -> (…,4,4). xi = [v(3), w(3)]"""
    v, w = xi[..., :3], xi[..., 3:]
    th = torch.linalg.norm(w, dim=-1, keepdims=True)  # (...,1)
    R = so3_exp(w)  # (...,3,3)

    I = torch.eye(3, dtype=xi.dtype, device=xi.device).expand(*xi.shape[:-1], 3, 3)
    small = (th.squeeze(-1) < 1e-8)[..., None, None]
    k = torch.where(small, torch.zeros_like(w), w / th)
    K = _so3_hat(k)
    sin_th = torch.sin(th)[..., None]
    cos_th = torch.cos(th)[..., None]

    # V = I + (1 - cos)K + (th - sin)K^2
    V = I + (1.0 - cos_th) * K + (th - sin_th) * (K @ K)
    # small-angle Taylor: V ≈ I + 0.5K + 1/6 K^2
    V_small = I + 0.5 * K + (1.0/6.0) * (K @ K)
    V = torch.where(small, V_small, V)

    t = (V @ v[..., None]).squeeze(-1)  # (...,3)

    T = torch.eye(4, dtype=xi.dtype, device=xi.device).expand(*xi.shape[:-1], 4, 4).clone()
    T[..., :3, :3] = R
    T[..., :3,  3] = t
    return T

def se3_log(T: torch.Tensor) -> torch.Tensor:
    """(N,4,4) -> (N,6). xi = [v,w]."""
    R = T[..., :3, :3]                 # (N,3,3)
    t = T[..., :3,  3]                 # (N,3)

    # --- rotation log (vector) ---
    # so3_log returns (N,3)
    tr = (R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]).clamp(-1.0, 3.0)
    cos_th = (tr - 1.0) * 0.5
    th = torch.acos(cos_th.clamp(-1.0, 1.0)).unsqueeze(-1)         # (N,1)

    w_skew = 0.5 * (R - R.transpose(-1, -2))                       # (N,3,3)
    vee = torch.stack([w_skew[..., 2, 1], w_skew[..., 0, 2], w_skew[..., 1, 0]], dim=-1)  # (N,3)

    small = (th.squeeze(-1) < 1e-8).unsqueeze(-1)                  # (N,1)
    s = torch.sin(th).clamp(min=1e-12)                              # (N,1)
    w = (th / (2.0 * s)) * vee                                     # (N,3)
    w = torch.where(small, vee, w)                                  # (N,3)

    # --- V matrix for translation (matrices) ---
    I = torch.eye(3, dtype=T.dtype, device=T.device).expand(t.shape[0], 3, 3)  # (N,3,3)
    # unit axis k: condition must be (N,1), not (N,1,1)
    has_rot = (th.squeeze(-1) > 0).unsqueeze(-1)                    # (N,1)
    k = torch.where(has_rot, w / th.clamp(min=1e-12), torch.zeros_like(w))     # (N,3)
    K = _so3_hat(k)                                                 # (N,3,3)

    sin_th = torch.sin(th).unsqueeze(-1)                            # (N,1,1)
    cos_th = torch.cos(th).unsqueeze(-1)                            # (N,1,1)

    V = I + (1.0 - cos_th) * K + (th.unsqueeze(-1) - sin_th) * (K @ K)  # (N,3,3)
    V_small = I + 0.5 * K + (1.0/6.0) * (K @ K)                          # (N,3,3)
    small_m = (th.squeeze(-1) < 1e-8).unsqueeze(-1).unsqueeze(-1)       # (N,1,1)
    V = torch.where(small_m, V_small, V)                                 # (N,3,3)

    v = torch.linalg.solve(V, t.unsqueeze(-1)).squeeze(-1)          # (N,3)

    xi = torch.cat([v, w], dim=-1)                                   # (N,6)
    return xi

def se3_inv(T: torch.Tensor) -> torch.Tensor:
    """(…,4,4) -> (…,4,4)"""
    R = T[..., :3, :3]
    t = T[..., :3,  3]
    RT = R.transpose(-1, -2)
    Tin = torch.eye(4, dtype=T.dtype, device=T.device).expand(*T.shape[:-2], 4, 4).clone()
    Tin[..., :3, :3] = RT
    Tin[..., :3,  3] = -(RT @ t[..., None]).squeeze(-1)
    return Tin

# -------------------- Robust M-estimator weights --------------------

def m_weight(r: torch.Tensor,
             kind: Literal["cauchy", "huber"] = "cauchy",
             c: float = 2.0) -> torch.Tensor:
    """
    Return the scalar weight ψ(r)/r for IRLS (influence function over residual norm).
    r: (...,) non-negative residual norms
    """
    eps = 1e-12
    if kind == "cauchy":
        # rho = (c^2/2) * log(1 + (r/c)^2)  ->  w = 1 / (1 + (r/c)^2)
        return 1.0 / (1.0 + (r / (c + eps))**2 + eps)
    elif kind == "huber":
        # rho = 0.5 r^2       if r<=c
        #      = c(r - 0.5c)  if r>c
        w = torch.ones_like(r)
        mask = (r > c)
        w[mask] = (c / (r[mask] + eps))
        return w
    else:
        raise ValueError("Unknown robust kind")

# -------------------- Robust SE(3) averaging --------------------

# @torch.no_grad()
# def robust_se3_average(
#     T_list: torch.Tensor,            # (N,4,4) proposals
#     w_conf: torch.Tensor,            # (N,)   confidence weights >=0
#     Sigma: torch.Tensor | None = None,   # (6,6) covariance (for whitening). If None => I
#     *,
#     max_iters: int = 20,
#     kind: Literal["cauchy", "huber"] = "cauchy",
#     c: float = 2.0,
#     init: torch.Tensor | None = None,    # (4,4) initial pose. If None, pick best-weighted medoid-ish init
#     tol: float = 1e-8,
#     device: str | torch.device = "cuda:0",
#     dtype: torch.dtype = torch.float64,
# ) -> Tuple[torch.Tensor, dict]:
#     """
#     Returns:
#       T*: (4,4) robust mean,
#       info: dict with 'iters', 'final_residual', 'num_used'
#     """
#     T_list = T_list.to(device=device, dtype=dtype)
#     w_conf = w_conf.to(device=device, dtype=dtype).clamp(min=0)
#     N = T_list.shape[0]
#     assert T_list.shape[-2:] == (4,4)
#     assert w_conf.shape == (N,)

#     # Whitening matrix S = Sigma^{-1/2}
#     if Sigma is None:
#         S = torch.eye(6, dtype=dtype, device=device)
#     else:
#         Sigma = Sigma.to(device=device, dtype=dtype)
#         # Compute inverse sqrt via Cholesky or eig
#         L = torch.linalg.cholesky(Sigma)                # Sigma = L L^T
#         S = torch.cholesky_inverse(L)                   # Sigma^{-1}
#         # want Sigma^{-1/2}; for robustness we can just use Sigma^{-1/2} via eig:
#         w_eig, V = torch.linalg.eigh(Sigma)
#         Sinv_sqrt = (V @ torch.diag_embed((1.0 / torch.sqrt(w_eig.clamp(min=1e-12)))) @ V.transpose(-1, -2))
#         S = Sinv_sqrt

    # # Initialization
    # if init is None:
    #     # simple init: pick pose with max confidence
    #     T = T_list[w_conf.argmax()]
    # else:
    #     T = init.to(device=device, dtype=dtype)

    # info = {}
    # for it in range(max_iters):
    #     # residuals in Lie algebra: xi_i = log(T^{-1} T_i)
    #     xi = se3_log(se3_inv(T) @ T_list)     # (N,6)
    #     # print("xw shape:", xi.shape)

    #     # whitened norms r_i = || S xi_i ||
    #     xi_w = (S @ xi.unsqueeze(-1)).squeeze(-1)   # (N,6)
    #     r = torch.linalg.norm(xi_w, dim=-1)         # (N,)

    #     # robust IRLS weights
    #     w_rob = m_weight(r, kind=kind, c=c)        # (N,)
    #     w = (w_conf * w_rob)                        # (N,)

    #     # If all weights ~0, stop
    #     if torch.all(w <= 1e-12):
    #         break

    #     # Weighted average in se(3): Δ = sum_i w_i * xi_i / sum_i w_i
    #     # (You could add anisotropic scaling here; we used S only for robust weights.)
    #     w_sum = w.sum()
    #     # print(w.shape, xi.shape)
    #     Delta = (w[:, None] * xi).sum(dim=0) / (w_sum + 1e-12)  # (6,)

    #     # Update on the manifold
    #     T_new = se3_exp(Delta) @ T

    #     # Convergence check (parameter step)
    #     if torch.linalg.norm(Delta) < tol:
    #         T = T_new
    #         break
    #     T = T_new

    # # Final stats
    # xi = se3_log(se3_inv(T) @ T_list)
    # xi_w = (S @ xi.unsqueeze(-1)).squeeze(-1)
    # r = torch.linalg.norm(xi_w, dim=-1)
    # w_rob = m_weight(r, kind=kind, c=c)
    # w = w_conf * w_rob
    # final_loss = (w * r).sum().item()

    # info.update(dict(iters=it+1, final_residual=final_loss, num_used=int((w>0).sum().item())))
    # return T, info


def main():
    ap = argparse.ArgumentParser(
        description="Robust pose aggregation"
    )
    ap.add_argument("--raw_poses_root", type=str, required=True, help="Path json containing raw poses")
    ap.add_argument("--scene", type=str, required=True, help="Name of the scene")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- config ---
    dev   = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64

    with open(args.raw_poses_root) as file:
        dict_raw_poses = json.load(file)
    df = pd.DataFrame(dict_raw_poses)

    # compute device poses 
    df["Trs"] = df["Trs"].apply(lambda x: np.asarray(x, dtype=np.float64))
    df["Tws"] = df["pose_est"].apply(lambda d: np.asarray(d["pose"], dtype=np.float64))
    df["Twr"] = df.apply(lambda r: r["Tws"] @ invert_se3_np(r["Trs"]), axis=1)


    # build (device_id, timestamp) -> (list_of_Twr, weights) dict (tensors on the right device/dtype)
    def _weight_from_pose_est(p):
        # example: inverse of error; tweak as you like
        return 1.0 / (1e-7 + p["error"])

    grp_series = (
        df.groupby(["device_id", "timestamp"])
        .apply(lambda g: (
            # list of 4x4 tensors
            [torch.as_tensor(T, device=dev, dtype=dtype) for T in g["Twr"].values],
            # 1D weights tensor
            torch.as_tensor([_weight_from_pose_est(p) for p in g["pose_est"].values],
                            device=dev, dtype=dtype)
        ))
    )

    dev_id_to_pose_weight = grp_series.to_dict()  # keys: (device_id, timestamp)

    # run robust_se3_average per key
    results = []  # rows to write: [timestamp, device_id, qw, qx, qy, qz, tx, ty, tz]

    for (device_id, ts), (pose_list, w_conf) in sorted(dev_id_to_pose_weight.items(),
                                                    key=lambda kv: (kv[0][0], kv[0][1])):
        # skip empty
        if not pose_list:
            continue

        # Stack to (N,4,4)
        T_list = torch.stack(pose_list, dim=0)  # (N,4,4)

        # Guard: if weights length mismatches N, fix/truncate
        N = T_list.shape[0]
        if w_conf.numel() != N:
            w_conf = w_conf.flatten()
            N2 = min(N, w_conf.numel())
            T_list = T_list[:N2]
            w_conf = w_conf[:N2]

        T_star = T_list[torch.argmax(w_conf)]
        # # If only one proposal, just take it; else robust average
        # if T_list.shape[0] == 1:
        #     T_star = T_list[0]
        # else:
        #     # (optional) anisotropic covariance between trans/rot, else pass Sigma=None
        #     Sigma = None
        #     # init: best weight
        #     init_T = T_list[torch.argmax(w_conf)]
        #     T_star, _info = robust_se3_average(
        #         T_list, w_conf,
        #         Sigma=Sigma,
        #         max_iters=30,
        #         kind="cauchy", c=2.0,
        #         init=init_T,
        #         device=dev, dtype=dtype
        #     )

        # extract R,t and convert to quaternion
        R = T_star[:3, :3].detach().cpu().numpy()
        t = T_star[:3,  3].detach().cpu().numpy()
        qw, qx, qy, qz = _q_from_R(R)  # your function

        # append row (covariance left empty for now)
        results.append([str(ts), device_id, f"{qw:.17g}", f"{qx:.17g}", f"{qy:.17g}", f"{qz:.17g}",
                        f"{t[0]:.17g}", f"{t[1]:.17g}", f"{t[2]:.17g}"])

    # write to txt (CSV with commas). No header unless you want one.
    map_device = args.raw_poses_root.split("_")[2]
    query_device = args.raw_poses_root.split("_")[4].split(".")[0]
    out_txt = f"{args.out_dir}/{args.scene}_map_{map_device}_query_{query_device}.txt"
    with open(out_txt, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # Optionally write a header:
        w.writerow(["# timestamp","device_id","qw","qx","qy","qz","tx","ty","tz", "*covar"])
        w.writerows(results)

    print(f"Wrote {len(results)} rows to {out_txt}")


if __name__ == "__main__":
    main()