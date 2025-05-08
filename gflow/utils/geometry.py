
import math
import torch

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