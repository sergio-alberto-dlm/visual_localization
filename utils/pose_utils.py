import torch 

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
    Ti = torch.eye(4, dtype=T.dtype)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti
