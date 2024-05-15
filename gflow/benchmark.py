from typing import Optional
from pathlib import Path

from utils import image_path_to_tensor, apply_float_colormap
from trainer import SimpleGaussian

import glob
from helper import hungarian_match

import os
import tyro
import torch
import imageio
import numpy as np
from typing import Literal
import piqa
from tqdm import tqdm
import json
from geometry import compute_rpe, compute_ate


def calculate_iou(mask1, mask2):
    """Calculate Intersection over Union (IoU) of two masks.
    Input: mask1, mask2 - torch.Tenser (height, width)
    """
    # Calculate intersection
    intersection = torch.logical_and(mask1, mask2).sum().float()

    # Calculate union
    union = torch.logical_or(mask1, mask2).sum().float()

    # Calculate IoU
    iou = intersection / union

    return iou.item()

def read_log_camera(json_path: str) -> np.ndarray:
    """Read camera pose from log file.
    Input: json_path - str
    Output: camera pose (R|T) - np.ndarray (4,4)
    """
    with open(json_path, "r") as f:
        json_data = json.load(f)
    camera_position = np.array(json_data[-1]["position"]).reshape(-1, 1)
    camera_rotaion = np.array(json_data[-1]["rotation"])
    camera_pose = np.concatenate([camera_rotaion, camera_position], axis=1)
    # make it homogeneous
    camera_pose = np.vstack([camera_pose, np.array([0,0,0,1])])
    return camera_pose

def read_gt_camera(json_path: str) -> np.ndarray:
    """Read ground truth camera pose from log file.
    Input: json_path - str
    Output: camera pose (R|T) - np.ndarray (4,4)
    """
    with open(json_path, "r") as f:
        json_data = json.load(f)
    camera_pose = np.array(json_data["pose"])
    return camera_pose


def main(
    log_path: str="/home/wangshizun/projects/msplat/logs/2024_05_14-06_28_24",
    sequence_path: str="/home/wangshizun/projects/msplat/images/car-turn/car-turn",
    eval_recon: bool = True,
    eval_track: bool = True,
    eval_seg: bool = True,
    eval_camera: bool = True,
) -> None:
    csv_list = []
    """Evaluate reconstruction quality (PSNR, SSIM)"""
    if eval_recon:
        print("Evaluating reconstruction quality (PSNR, SSIM)...")
        # recursively search for all images in the log path
        log_image_path = os.path.join(log_path, "images")
        log_image_paths = sorted(glob.glob(os.path.join(log_image_path, "img_0*.png"))) + sorted(glob.glob(os.path.join(log_image_path, "img_0*.jpg")))
        
        psnr_list = []
        ssim_list = []
        lpips_list = []
        psnr  = piqa.PSNR()
        ssim  = piqa.SSIM()
        lpips = piqa.LPIPS()
        for image_path in tqdm(log_image_paths):
            image = image_path_to_tensor(image_path).permute(2,0,1).unsqueeze(0) # (B, C, H, W)
            base_name = os.path.basename(image_path)
            base_name = base_name.split(".")[0].split("_")[-1]
            image_gt_path = os.path.join(sequence_path, f"{base_name}.jpg")
            image_gt = image_path_to_tensor(image_gt_path).permute(2,0,1).unsqueeze(0) # (B, C, H, W)
            psnr_list.append(psnr(image, image_gt))
            ssim_list.append(ssim(image, image_gt))
            lpips_list.append(lpips(image, image_gt))
        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)
        avg_lpips = np.mean(lpips_list)
        print(f"Average PSNR: {avg_psnr}")
        csv_list.append(f"PSNR:\t{avg_psnr}\n")
        print(f"Average SSIM: {avg_ssim}")
        csv_list.append(f"SSIM:\t{avg_ssim}\n")
        print(f"Average LPIPS: {avg_lpips}")
        csv_list.append(f"LPIPS:\t{avg_lpips}\n")

    """Evaluate tracking quality (MOTA, MOTP)"""
    if eval_track:
        pass
    # TODO implement tracking evaluation from TAP-Vid



    """Evaluate segmentation quality (IoU)"""
    if eval_seg:
        print("Evaluating segmentation quality (IoU)...")
        # recursively search for all images in the log path
        log_seg_path = os.path.join(log_path, "images_seg")
        log_seg_paths = sorted(glob.glob(os.path.join(log_seg_path, "move_mask_*.png"))) + sorted(glob.glob(os.path.join(log_image_path, "move_mask_*.jpg")))
        iou_list = []
        seg_gt = None
        i = 0
        for seg_path in tqdm(log_seg_paths):
            seg = image_path_to_tensor(seg_path) # (H, W)
            seg = seg > 0.5
            base_name = os.path.basename(seg_path)
            base_name = base_name.split(".")[0].split("_")[-1]
            # seg_gt_path = os.path.join(sequence_path+"seg", f"{base_name}.png")
            # seg_gt = image_path_to_tensor(seg_gt_path) # (H, W)
            # seg_gt = seg_gt > 0.5
            if i == 0:
                seg_gt = seg
            i += 1

            # Calculate IoU
            iou = calculate_iou(seg, seg_gt)
            iou_list.append(iou)

        avg_iou = np.mean(iou_list)
        print(f"Average IoU: {avg_iou}")
        csv_list.append(f"IoU:\t{avg_iou}\n")

    """Evaluate trajectory quality (ATE, RPE)"""
    if eval_camera:
        print("Evaluating trajectory quality (ATE, RPE)...")
        # recursively search for all images in the log path
        log_camera_path = os.path.join(log_path, "json")
        log_camera_paths = sorted(glob.glob(os.path.join(log_camera_path, "*.json")))
        camera_pose_list = []
        camera_pose_gt_list = []
        for camera_path in tqdm(log_camera_paths):
            camera_pose = read_log_camera(camera_path)
            camera_pose_list.append(camera_pose)

            base_name = os.path.basename(camera_path)
            camera_gt_path = os.path.join(sequence_path+"_camera_dust3r", f"{base_name}")
            camera_pose_gt = read_gt_camera(camera_gt_path)
            camera_pose_gt_list.append(camera_pose_gt)

        ate = compute_ate(camera_pose_gt_list, camera_pose_list)
        rpe_t, rpe_r = compute_rpe(camera_pose_gt_list, camera_pose_list)
        rpe_t = rpe_t * 100
        rpe_r = rpe_r * 180 / np.pi

        print(f"ATE: {ate}")
        csv_list.append(f"ATE:\t{ate}\n")
        print(f"RPE_translation: {rpe_t}")
        csv_list.append(f"RPE_t:\t{rpe_t}\n")
        print(f"RPE_rotation: {rpe_r}")
        csv_list.append(f"RPE_r:\t{rpe_r}\n")


    # save metrics to csv
    with open(os.path.join(log_path, "metrics.csv"), "w") as f:
        for line in csv_list:
            f.write(line)


if __name__ == "__main__":
    tyro.cli(main)