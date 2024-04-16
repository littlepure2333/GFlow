import torch
import msplat
import utils
import numpy as np


def render_return_rgb_uv_d(xyz, scale, rotate, opacity, rgb, intr, extr, bg, W, H):
    """
    return rendered_rgb(3,H,W), uv(N,2) and depth(N,1)
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

    # alpha blending image
    rendered_rgb = msplat.alpha_blending(
        uv, conic, 
        opacity,
        rgb, 
        gaussian_ids_sorted, tile_range, 
        bg, W, H,
    )

    return rendered_rgb, uv, depth

def render_depth(xyz, scale, rotate, opacity, rgb, intr, extr, bg, W, H):
    """
    renturn rendered_depth (1,H,W)
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

    # alpha blending
    rendered_depth = msplat.alpha_blending(
        uv, conic, 
        opacity,
        depth, 
        gaussian_ids_sorted, tile_range, 
        bg, W, H,
    )

    return rendered_depth

def render_depth_map(xyz, scale, rotate, opacity, rgb, intr, extr, bg, W, H):
    """
    renturn rendered_depth_map with color (3,H,W)
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

    # apply colormap
    depth_color = utils.apply_float_colormap(
        depth, colormap="turbo", non_zero=True
    )

    # alpha blending
    rendered_depth_map = msplat.alpha_blending(
        uv, conic, 
        opacity,
        depth_color, 
        gaussian_ids_sorted, tile_range, 
        bg, W, H,
    )

    return rendered_depth_map

def render_center(xyz, scale, rotate, opacity, rgb, intr, extr, bg, W, H):
    """
    renturn rendered_center with color (3,H,W)
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

    # depths_center = torch.ones_like(depths)
    radius = torch.ones_like(radius) * 10
    # conics (inverse of covariance) of 2D gaussians in upper triangular format
    conic = torch.ones_like(conic, device=conic.device) * torch.Tensor([1, 0, 1]).to(conic.device)
    opacity = torch.ones_like(opacity)

    # alpha blending
    rendered_center = msplat.alpha_blending(
        uv, conic, 
        opacity,
        rgb, 
        gaussian_ids_sorted, tile_range, 
        bg, W, H,
    )

    return rendered_center

def render2img(rendered):
    """
    convert rendered (3,H,W) to img (H,W,3) in numpy format, and 255 scale
    """
    rendered = rendered.detach().permute(1, 2, 0)
    rendered = torch.clamp(rendered, 0.0, 1.0)
    rendered_np = (rendered.cpu().numpy() * 255).astype(np.uint8)

    return rendered_np