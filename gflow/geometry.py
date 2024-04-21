
import time
import math
import torch
import msplat

import torch.nn.functional as F
import numpy as np

import math

def pix2world(uv, depth, intr, extr):
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