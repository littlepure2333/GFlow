
import time
import math
import torch
import msplat

import torch.nn.functional as F
import numpy as np

import math

# Adapted from NoPe-NeRF https://github.com/ActiveVisionLab/nope-nerf/blob/main/utils_poses/comp_ate.py
def rotation_error(pose_error):
    """Compute rotation error
    Args:
        pose_error (4x4 array): relative pose error
    Returns:
        rot_error (float): rotation error
    """
    a = pose_error[0, 0]
    b = pose_error[1, 1]
    c = pose_error[2, 2]
    d = 0.5*(a+b+c-1.0)
    rot_error = np.arccos(max(min(d, 1.0), -1.0))
    return rot_error

def translation_error(pose_error):
    """Compute translation error
    Args:
        pose_error (4x4 array): relative pose error
    Returns:
        trans_error (float): translation error
    """
    dx = pose_error[0, 3]
    dy = pose_error[1, 3]
    dz = pose_error[2, 3]
    trans_error = np.sqrt(dx**2+dy**2+dz**2)
    return trans_error

def compute_rpe(gt, pred):
    trans_errors = []
    rot_errors = []
    for i in range(len(gt)-1):
        gt1 = gt[i]
        gt2 = gt[i+1]
        gt_rel = np.linalg.inv(gt1) @ gt2

        pred1 = pred[i]
        pred2 = pred[i+1]
        pred_rel = np.linalg.inv(pred1) @ pred2
        rel_err = np.linalg.inv(gt_rel) @ pred_rel
        
        trans_errors.append(translation_error(rel_err))
        rot_errors.append(rotation_error(rel_err))
    rpe_trans = np.mean(np.asarray(trans_errors))
    rpe_rot = np.mean(np.asarray(rot_errors))
    return rpe_trans, rpe_rot

def compute_ate(gt, pred):
    """Compute RMSE of ATE (Absolute Trajectory Error)
    Args:
        gt: ground-truth translation
        pred: predicted translation
    """
    errors = []

    for pred_xyz, gt_xyz in zip(pred, gt):
        align_err = gt_xyz - pred_xyz
        errors.append(np.sqrt(np.sum(align_err ** 2)))
        
    ate = np.sqrt(np.mean(np.asarray(errors) ** 2)) 
    return ate

def xy_grid(W, H, device=None, origin=(0, 0), unsqueeze=None, cat_dim=-1, homogeneous=False, **arange_kw):
    """ Output a (H,W,2) array of int32 
        with output[j,i,0] = i + origin[0]
             output[j,i,1] = j + origin[1]
    """
    if device is None:
        # numpy
        arange, meshgrid, stack, ones = np.arange, np.meshgrid, np.stack, np.ones
    else:
        # torch
        arange = lambda *a, **kw: torch.arange(*a, device=device, **kw)
        meshgrid, stack = torch.meshgrid, torch.stack
        ones = lambda *a: torch.ones(*a, device=device)

    tw, th = [arange(o, o+s, **arange_kw) for s, o in zip((W, H), origin)]
    grid = meshgrid(tw, th, indexing='xy')
    if homogeneous:
        grid = grid + (ones((H, W)),)
    if unsqueeze is not None:
        grid = (grid[0].unsqueeze(unsqueeze), grid[1].unsqueeze(unsqueeze))
    if cat_dim is not None:
        grid = stack(grid, cat_dim)
    return grid

def inv(mat):
    """ Invert a torch or numpy matrix
    """
    if isinstance(mat, torch.Tensor):
        return torch.linalg.inv(mat)
    if isinstance(mat, np.ndarray):
        return np.linalg.inv(mat)
    raise ValueError(f'bad matrix type = {type(mat)}')

def pix2world(uv, depth, intr, extr):
    rel_pts3d = depth2pts3d(depth, uv, intr[0], intr[2:])
    # print("rel_pts3d.shape", rel_pts3d.shape)
    # extr (world2cam): (3, 4) -> (4, 4) 
    extr_homo = torch.cat((extr, torch.tensor([[0, 0, 0, 1]], device=extr.device).float()), dim=0)
    cam2world = inv(extr_homo)
    # print("cam2world.shape", cam2world.shape)
    world_pts3d = geotrf(cam2world, rel_pts3d)
    # print("world_pts3d.shape", world_pts3d.shape)
    return world_pts3d

def depth2pts3d(depth, xys, focal, pp):
    return torch.cat((depth * (xys - pp) / focal, depth), dim=-1)

def geotrf(Trf, pts, ncol=None, norm=False):
    """ Apply a geometric transformation to a list of 3-D points.

    H: 3x3 or 4x4 projection matrix (typically a Homography)
    p: numpy/torch/tuple of coordinates. Shape must be (...,2) or (...,3)

    ncol: int. number of columns of the result (2 or 3)
    norm: float. if != 0, the resut is projected on the z=norm plane.

    Returns an array of projected 2d points.
    """
    assert Trf.ndim >= 2
    if isinstance(Trf, np.ndarray):
        pts = np.asarray(pts)
    elif isinstance(Trf, torch.Tensor):
        pts = torch.as_tensor(pts, dtype=Trf.dtype)

    # adapt shape if necessary
    output_reshape = pts.shape[:-1]
    ncol = ncol or pts.shape[-1]

    # optimized code
    if (isinstance(Trf, torch.Tensor) and isinstance(pts, torch.Tensor) and
            Trf.ndim == 3 and pts.ndim == 4):
        d = pts.shape[3]
        if Trf.shape[-1] == d:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf, pts)
        elif Trf.shape[-1] == d + 1:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf[:, :d, :d], pts) + Trf[:, None, None, :d, d]
        else:
            raise ValueError(f'bad shape, not ending with 3 or 4, for {pts.shape=}')
    else:
        if Trf.ndim >= 3:
            n = Trf.ndim - 2
            assert Trf.shape[:n] == pts.shape[:n], 'batch size does not match'
            Trf = Trf.reshape(-1, Trf.shape[-2], Trf.shape[-1])

            if pts.ndim > Trf.ndim:
                # Trf == (B,d,d) & pts == (B,H,W,d) --> (B, H*W, d)
                pts = pts.reshape(Trf.shape[0], -1, pts.shape[-1])
            elif pts.ndim == 2:
                # Trf == (B,d,d) & pts == (B,d) --> (B, 1, d)
                pts = pts[:, None, :]

        if pts.shape[-1] + 1 == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf[..., :-1, :] + Trf[..., -1:, :]
        elif pts.shape[-1] == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf
        else:
            pts = Trf @ pts.T
            if pts.ndim >= 2:
                pts = pts.swapaxes(-1, -2)

    if norm:
        pts = pts / pts[..., -1:]  # DONT DO /= BECAUSE OF WEIRD PYTORCH BUG
        if norm != 1:
            pts *= norm

    res = pts[..., :ncol].reshape(*output_reshape, ncol)
    return res

def pix2world_old(uv, depth, intr, extr):
    """
    convert uv and depth to world coordinates
    uv: (N, 2)
    depth: (N, 1)
    intr: (4,)
    extr: (3, 4)

    return: (N, 3)
    """
    # Create the intrinsic matrix K
    K = torch.eye(3).cuda()
    K[0, 0] = intr[0]
    K[1, 1] = intr[1]
    K[0, 2] = intr[2]
    K[1, 2] = intr[3]
    
    # Invert the intrinsic matrix
    K_inv = torch.inverse(K)
    
    # Prepare the uv coordinates (adjust from image to normalized camera coordinates)
    uv_hom = torch.cat([uv + 0.5, torch.ones(uv.shape[0], 1).cuda()], dim=1)
    uv_norm = uv_hom * depth
    
    # Convert image coordinates to camera coordinates
    pt_cam = torch.matmul(K_inv, uv_norm.t())
    # pt_cam = torch.einsum("...ij,...i->...j", K_inv, uv_norm)
    
    # Invert the extrinsic matrix
    R = extr[:3, :3]
    t = extr[:3, -1].unsqueeze(dim=1)
    
    # Inverse rotation and translation
    R_inv = torch.inverse(R)
    
    # Compute world coordinates
    # xyz = torch.matmul(R_inv, pt_cam[:3]) + t_inv
    # import pdb; pdb.set_trace()
    # xyz = torch.einsum("...ij,...j->...i", R_inv, pt_cam[:,:3])
    xyz = torch.matmul(R_inv, pt_cam[:3] - t)
    return xyz.t()

def project_point_torch_impl(
    xyz, 
    intr, 
    extr,
    W,
    H,
    nearest=0.2,
    extent=1.3
):
    
    K = torch.eye(3).cuda()
    K[0, 0] = intr[0]
    K[1, 1] = intr[1]
    K[0, 2] = intr[2]
    K[1, 2] = intr[3]

    R = extr[:3, :3]
    t = extr[:3, -1].unsqueeze(dim=1)
    
    pt_cam = torch.matmul(R, xyz.t()) + t
    
    # Apply camera intrinsic matrix
    p_proj = torch.matmul(K, pt_cam)

    depth = p_proj[2]
    uv = p_proj[:2] / depth - 0.5

    uv = uv.t()
    
    depth = torch.nan_to_num(depth)
    near_mask = depth <= nearest
    extent_mask_x = torch.logical_or(uv[:, 0] < (1 - extent) * W * 0.5, 
                                     uv[:, 0] > (1 + extent) * W * 0.5)
    extent_mask_y = torch.logical_or(uv[:, 1] < (1 - extent) * H * 0.5, 
                                     uv[:, 1] > (1 + extent) * H * 0.5)
    extent_mask = torch.logical_or(extent_mask_x, extent_mask_y)
    mask = torch.logical_or(near_mask, extent_mask)

    uv_masked = uv.clone()
    depth_masked = depth.clone()
    uv_masked[:, 0][mask] = 0
    uv_masked[:, 1][mask] = 0
    depth_masked[mask] = 0
    
    return uv_masked, depth_masked.unsqueeze(-1)

if __name__ == "__main__":
    seed = 123
    torch.manual_seed(seed)
    torch.set_printoptions(precision=20)
    
    iters = 100
    N = 1000
    
    print("=============================== running test on project_points ===============================")
    
    W = 1600
    H = 1200
    
    intr = torch.Tensor([2892.33, 2883.18, 823.205, 619.071]).cuda().float()
    extr = torch.Tensor([
            [0.970263, 0.00747983, 0.241939, -191.02],
            [-0.0147429, 0.999493, 0.0282234, 3.2883],
            [-0.241605, -0.030951, 0.969881, 22.5401]
        ]).cuda().float()
    
    xyz = torch.rand((N, 3)).cuda()
    xyz[:, 0] = xyz[:, 0] * 500
    xyz[:, 1] = xyz[:, 1] * 500
    xyz[:, 2] = xyz[:, 2] * 400 + 400

    xyz1 = xyz.clone().requires_grad_()
    xyz2 = xyz.clone().requires_grad_()
    intr1 = intr.clone().requires_grad_()
    intr2 = intr.clone().requires_grad_()
    extr1 = extr.clone().requires_grad_()
    extr2 = extr.clone().requires_grad_()
    
    for i in range(iters):
        (
            out_pytorch_uv, 
            out_pytorch_depth
        ) = project_point_torch_impl(
            xyz1, 
            intr1, 
            extr1,
            W, H,
            nearest=0,
            extent = math.inf)
        p = pix2world(out_pytorch_uv, out_pytorch_depth, intr1, extr1)
        try:
            torch.testing.assert_close(p, xyz1, rtol=1e-5, atol=1e-5)
        except:
            print("Pix2World Function Pass Failed for Iteration {}".format(i))
    print("Pix2World Function Pass.")