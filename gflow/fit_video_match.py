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


def main(
    height: int = 256,
    width: int = 256,
    num_points: int = 1000,
    save_imgs: bool = False,
    save_ckpt: bool = True,
    # fps: int = 30,
    sequence_path: Optional[str] = "/home/wangshizun/projects/gsplat/images/beauty_0/beauty_0",
    iterations_first: int = 10,
    iterations_after: int = 100,
    lr: float = 0.01,
    lr_rgbs: float = 0.02,
    lr_camera: float = 0.01,
    lr_after: float = 1e-3,
    lr_rgbs_after: float = 1e-5,
    lr_camera_after: float = 1e-3,
    gpu: int = 0,
    ckpt_path: Optional[str] = None,
    lambda_depth: float = 0.,
    lambda_delta: float = 0.,
    lambda_mov: float = 0.,
    background: Literal["black", "white"] = "black",
    camera_first: bool = False,
    iterations_camera: int = 10,
    frame_start: int = 0,
    frame_range: Optional[int] = -1,
    densify_interval: int = 0,
    grad_threshold: float = 5e-3,
    lambda_var: float = 1e-7,
    skip_interval: int = 1,
    resize: int = None,
    blur: bool = False,
    loss_verbose: bool = False,
    add: bool = False,
    depth_scale: float = 16.0,
    load_path: str = "/home/wangshizun/projects/gsplat/logs/2024_03_01-16_51_25",
    traj_only: bool = False,
    fixed_camera: bool = False,
    traj_num: int = 200,
) -> None:
    
    frames_sequence = []
    frames_center_sequence = []
    frames_depth_sequence = []
    frames_sequence_traj = []
    frames_center_sequence_traj = []
    frames_depth_sequence_traj = []
    frames_sequence_traj_upon = []

    ckpt_paths = sorted(glob.glob(os.path.join(load_path, "ckpt", "*.tar")))
    # print(ckpt_paths)

    img_paths = []
    for ckpt_path in ckpt_paths:
        img_path = os.path.join(sequence_path, os.path.basename(ckpt_path).split('.')[0]+'.png')
        if not os.path.exists(img_path):
            img_path = os.path.join(sequence_path, os.path.basename(ckpt_path).split('.')[0]+'.jpg')
        img_paths.append(img_path)

    img_depth_paths = []
    for ckpt_path in ckpt_paths:
        img_depth_path = os.path.join(sequence_path+'_depth', os.path.basename(ckpt_path).split('.')[0]+'.png')
        if not os.path.exists(img_depth_path):
            img_depth_path = os.path.join(sequence_path+'_depth', os.path.basename(ckpt_path).split('.')[0]+'.jpg')
        img_depth_paths.append(img_depth_path)

    img_occ_paths = []
    for ckpt_path in ckpt_paths:
        img_occ_path = os.path.join(sequence_path+'_flow', os.path.basename(ckpt_path).split('.')[0]+'_occ_bwd.png')
        if not os.path.exists(img_occ_path):
            img_occ_path = os.path.join(sequence_path+'_flow', os.path.basename(ckpt_path).split('.')[0]+'_occ_bwd.jpg')
        img_occ_paths.append(img_occ_path)

    # load the first frame
    gt_image0 = image_path_to_tensor(img_paths[0], resize=resize, blur=blur)
    trainer = SimpleGaussian(gt_image=gt_image0, num_points=num_points, background=background, depth_scale=depth_scale)
    print("[check] trainer.W", trainer.W)
    print("[check] trainer.H", trainer.H)
    ckpt_paths = sorted(glob.glob(os.path.join(load_path, "ckpt_match", "*.tar")))
    ckpt_paths = sorted(glob.glob(os.path.join(load_path, "ckpt", "*.tar")))

    for i, ckpt_path in enumerate(ckpt_paths):
        print(f"[{i}/{len(ckpt_paths) - 1}] Gaussian splatting {ckpt_path}")
        trainer.load_checkpoint(ckpt_path)
        if i == 0:
            num_points = len(trainer.get_attribute("xyz"))
            # traj_num = 200 # 0.4% of the points
            interval = int(num_points / traj_num)
            traj_index = range(num_points)[::interval]
            # traj_index = range(num_points)[50:100]
            # help me to choose the index of the trainer.means that the trainer.means[:, 2] > 0.5 & trainer.means[:, 2] < 0.6, and sort them by trainer.means[:, 3]
            # traj_index = [i for i in range(num_points) if trainer.means[i, 1] > -0.3 and trainer.means[i, 1] < -0.1 and trainer.means[i, 0] > -0.4 and trainer.means[i, 0] < -0.1]
            # traj_index = sorted(traj_index, key=lambda x: trainer.means[x, 2])[:5] # sort order is from small to large
            # traj_index = traj_index[:10]
        (out_img, out_img_center, out_img_depth, 
         out_img_traj, out_img_traj_upon ) = trainer.eval(
            traj_index=traj_index,
            line_scale=0.2, 
            point_scale=2., 
            alpha=0.8
        )
        print("[check] scales.max", trainer.get_attribute("scale").max())
        print("[check] scales.min", trainer.get_attribute("scale").min())
        print("[check] extr\n", trainer.extr)

        frames_sequence.append(out_img)
        frames_center_sequence.append(out_img_center)
        frames_depth_sequence.append(out_img_depth)
        frames_sequence_traj.append(out_img_traj)
        frames_sequence_traj_upon.append(out_img_traj_upon)
        torch.cuda.empty_cache()

    
    # generate video
    for name, frames, fps in zip(
        ["sequence", "sequence_center", "sequence_depth", "sequence_traj", "sequence_traj_upon"],
        [frames_sequence, frames_center_sequence, frames_depth_sequence, frames_sequence_traj, frames_sequence_traj_upon],
        [5, 5, 5, 5, 5, 5, 5]):
        mp4_path = os.path.join(trainer.dir, f"{name}.mp4")
        save_video(mp4_path, frames, fps)
    

def save_video(mp4_path, frames, fps):
    # Convert frames to uint8 before saving with imageio
    frames_uint8 = [frame.astype(np.uint8) for frame in frames]

    # Write the frames to an MP4 file
    with imageio.get_writer(mp4_path, fps=fps) as writer:  # Adjust fps as needed
        for frame in frames_uint8:
            writer.append_data(frame)


if __name__ == "__main__":
    tyro.cli(main)