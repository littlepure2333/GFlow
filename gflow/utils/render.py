import torch
import msplat
from .color import apply_float_colormap
import numpy as np

def render_multiple(input_group, 
                    return_type=["rgb", "uv", "depth", "depth_map", "depth_map_color", "center"],
                    center_scale=10.0,):
    xyz, scale, rotate, opacity, rgb, intr, extr, bg, W, H = input_group
    """
    render on demanding return_type:
    rgb: (3,H,W)
    uv: (N,2)
    depth: (N,1)
    depth_map: (1,H,W)
    depth_map_color: (3,H,W)
    center: (3,H,W)
    """
    return_dict = {}
    # project points
    (uv, depth) = msplat.project_point(
        xyz, 
        intr, extr, W, H
    )
    # clamp uv to [0, 100]
    # uv[:,0] = torch.clamp(uv[:,0], 0, 100) # horizental
    # uv[:,1] = torch.clamp(uv[:,1], 0, 500) # vertical

    visible = depth != 0
    if "uv" in return_type:
        return_dict["uv"] = uv

    if "depth" in return_type:
        return_dict["depth"] = depth

    # compute cov3d
    cov3d = msplat.compute_cov3d(
        scale, 
        rotate, 
        visible
    )

    # ewa project
    (conic, radius, tiles_touched) = msplat.ewa_project(
        xyz, 
        cov3d, 
        intr, extr, uv, 
        W, H, visible
    )

    # sort
    (gaussian_ids_sorted, tile_range) = msplat.sort_gaussian(
        uv, depth, W, H, radius, tiles_touched
    )

    if "rgb" in return_type:
        # alpha blending image
        rendered_rgb = msplat.alpha_blending(
            uv, conic, 
            opacity,
            rgb, 
            gaussian_ids_sorted, tile_range, 
            bg, W, H,
        )
        return_dict["rgb"] = rendered_rgb

    if "depth_map" in return_type:
        rendered_depth_map = msplat.alpha_blending(
            uv, conic, 
            opacity,
            depth, 
            gaussian_ids_sorted, tile_range, 
            bg, W, H,
        )
        return_dict["depth_map"] = rendered_depth_map

    if "depth_map_color" in return_type:
        # apply colormap
        depth_color = apply_float_colormap(
            depth, colormap="turbo", non_zero=True
        )

        # alpha blending
        rendered_depth_map_color = msplat.alpha_blending(
            uv, conic, 
            opacity,
            depth_color, 
            gaussian_ids_sorted, tile_range, 
            bg, W, H,
        )
        return_dict["depth_map_color"] = rendered_depth_map_color
    
    if "center" in return_type:
        radius = torch.ones_like(radius) * center_scale
        # conics (inverse of covariance) of 2D gaussians in upper triangular format
        conic = torch.ones_like(conic, device=conic.device) * torch.Tensor([1, 0, 1]).to(conic.device)
        opacity = torch.ones_like(opacity)

        rendered_center = msplat.alpha_blending(
            uv, conic, 
            opacity,
            rgb, 
            gaussian_ids_sorted, tile_range, 
            bg, W, H,
        )
        return_dict["center"] = rendered_center

    return return_dict

def render_traj(input_group, point_num, line_scale=1., point_scale=2.):
    xyz, scale, rotate, opacity, rgb, intr, extr, bg, W, H = input_group
    """
    render trajectory and output image: (3,H,W)
    """
    # project points
    (uv, depth) = msplat.project_point(
        xyz, 
        intr, extr, W, H
    )
    visible = depth != 0

    # compute cov3d
    cov3d = msplat.compute_cov3d(
        scale, 
        rotate, 
        visible
    )

    # ewa project
    (conic, radius, tiles_touched) = msplat.ewa_project(
        xyz, 
        cov3d, 
        intr, extr, uv, 
        W, H, visible
    )

    # sort
    (gaussian_ids_sorted, tile_range) = msplat.sort_gaussian(
        uv, depth, W, H, radius, tiles_touched
    )

    # conics (inverse of covariance) of 2D gaussians in upper triangular format
    # shape: (N, 3)
    conic = torch.ones_like(conic, device=conic.device) * torch.Tensor([1, 0, 1]).to(conic.device) * line_scale
    # make the last N points to be larger
    conic[:-point_num] = torch.ones_like(conic[:-point_num]) * torch.Tensor([1, 0, 1]).to(conic.device) * point_scale

    rendered_traj = msplat.alpha_blending(
        uv, conic, 
        opacity,
        rgb, 
        gaussian_ids_sorted, tile_range, 
        bg, W, H,
    )

    return rendered_traj

def render2img(rendered):
    """
    convert rendered (3,H,W) to img (H,W,3) in numpy format, and 255 scale
    """
    rendered = rendered.detach().permute(1, 2, 0)
    rendered = torch.clamp(rendered, 0.0, 1.0)
    rendered_np = (rendered.cpu().numpy() * 255).astype(np.uint8)

    return rendered_np