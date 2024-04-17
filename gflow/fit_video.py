from typing import Optional
from pathlib import Path
from utils import image_path_to_tensor
from trainer import SimpleGaussian
import os
import tyro
import torch
import imageio
import numpy as np
from typing import Literal
from helper import readFlow



def main(
    height: int = 256,
    width: int = 256,
    num_points: int = 1000,
    save_imgs: bool = False,
    save_ckpt: bool = True,
    # fps: int = 30,
    sequence_path: Optional[Path] = "/home/wangshizun/projects/gsplat/images/beauty_0/beauty_0",
    iterations_first: int = 10,
    iterations_after: int = 100,
    iterations_new: int = 100,
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
    lambda_rig: float = 0.,
    lambda_flow: float = 0.,
    background: Literal["black", "white"] = "black",
    camera_first: bool = False,
    iterations_camera: int = 10,
    frame_start: int = 0,
    frame_range: Optional[int] = -1,
    densify_times: int = 1,
    densify_interval: int = 0,
    grad_threshold: float = 5e-3,
    densify_times_after: int = 1,
    densify_interval_after: int = 0,
    grad_threshold_after: float = 5e-3,
    densify_times_new: int = 1,
    densify_interval_new: int = 0,
    grad_threshold_new: float = 5e-3,
    lambda_var: float = 1e-7,
    skip_interval: int = 1,
    resize: int = None,
    blur: bool = False,
    loss_verbose: bool = False,
    add: bool = False,
    depth_scale: float = 16.0,
    num_knn: int = 20,
    slow_color: bool = False,
    slow_means: bool = False,
) -> None:
    
    frames_sequence_optimize = []
    frames_center_sequence_optimize = []
    frames_depth_sequence_optimize = []
    frames_sequence = []
    frames_center_sequence = []
    frames_depth_sequence = []

    # read all images(png, jpg, ...) in the folder and sort them
    img_paths = sorted(Path(sequence_path).glob("*.png")) + sorted(Path(sequence_path).glob("*.jpg"))
    if frame_range == -1:
        frame_range = len(img_paths)
    img_depth_paths = sorted(Path(str(sequence_path)+'_depth').glob("*.png")) + sorted(Path(str(sequence_path)+'_depth').glob("*.jpg"))
    img_paths = img_paths[frame_start:frame_start+frame_range][::skip_interval]
    img_depth_paths = img_depth_paths[frame_start:frame_start+frame_range][::skip_interval]
    img_occ_paths = sorted(Path(str(sequence_path)+'_flow').glob("*occ_bwd.png")) + sorted(Path(str(sequence_path)+'_flow').glob("*occ_bwd.jpg"))
    img_occ_paths = img_occ_paths[frame_start:frame_start+frame_range-1][::skip_interval]
    flow_paths = sorted(Path(str(sequence_path)+'_flow').glob("*pred.flo"))
    flow_paths = flow_paths[frame_start:frame_start+frame_range-1][::skip_interval]

    # gt_images = [image_path_to_tensor(img_path, resize=resize, blur=blur) for img_path in img_paths]
    # gt_depths = [image_path_to_tensor(img_depth_path, resize=resize, blur=blur) / 255.0  for img_depth_path in img_depth_paths]

    # fit the first frame
    # frame0 = gt_images[0]
    save_name0 = os.path.basename(img_paths[0]).split('.')[0]
    gt_image0 = image_path_to_tensor(img_paths[0], resize=resize, blur=blur)
    gt_depth0 = image_path_to_tensor(img_depth_paths[0], resize=resize, blur=blur)

    trainer = SimpleGaussian(gt_image=gt_image0, gt_depth=gt_depth0, num_points=num_points, background=background, depth_scale=depth_scale)
    trainer.init_gaussians_from_image(gt_image=gt_image0, gt_depth=gt_depth0, num_points=num_points, t3=trainer.t3)
    frames, frames_center, frames_depth = trainer.train(
        iterations=iterations_first,
        lr=lr,
        # lr_rgbs=lr_rgbs,
        # lr_camera=lr_camera,
        save_imgs=True,
        save_videos=True,
        save_ckpt=True,
        ckpt_name=save_name0,
        lambda_depth=lambda_depth,
        densify_times=densify_times,
        densify_interval=densify_interval,
        grad_threshold=grad_threshold,
    )
    frames_sequence_optimize += frames
    frames_center_sequence_optimize += frames_center
    frames_depth_sequence_optimize += frames_depth
    frames_sequence.append(frames[-1])
    frames_center_sequence.append(frames_center[-1])
    frames_depth_sequence.append(frames_depth[-1])

    # fit the subsequent frames
    # for img_path in img_paths[1:]:
    for i, img_path in enumerate(img_paths[1:]):
        try:
        # if True:
            save_name = os.path.basename(img_path).split('.')[0]
            print(f"[{i+1}/{len(img_paths) - 1}] fitting {img_path}")
            trainer.load_checkpoint(trainer.checkpoint_path)
            gt_image = image_path_to_tensor(img_path, resize=resize, blur=blur)
            gt_depth = image_path_to_tensor(img_depth_paths[i+1], resize=resize, blur=blur)
            occ_mask = image_path_to_tensor(img_occ_paths[i], resize=resize, blur=blur)
            gt_flow = readFlow(flow_paths[i], resize=resize, blur=blur)
            # transform 1 is near to 1 is far
            gt_depth = 1 - gt_depth
            trainer.set_gt_image(gt_image)
            trainer.set_gt_depth(gt_depth)
            trainer.set_gt_flow(gt_flow)

            if iterations_after > 0:
                # print(f"[{i+1}/{len(img_paths) - 1}] Rehearshal stage for new points.................")
                print(f"[{i+1}/{len(img_paths) - 1}] Optimize all .................")
                if add:
                    trainer.add_gaussians_from_image(gt_image=gt_image, gt_depth=gt_depth, mask=occ_mask)
                    pass
                frames, frames_center, frames_depth = trainer.train(
                    iterations=iterations_after,
                    lr=lr_after,
                    save_imgs=True,
                    save_videos=False,
                    save_ckpt=True,
                    ckpt_name=save_name,
                    # fps=fps,
                    lambda_depth=lambda_depth,
                    lambda_flow=lambda_flow,
                    densify_times=densify_times_after,
                    densify_interval=densify_interval_after,
                    grad_threshold=grad_threshold_after,
                    # mask=None,
                    mask=occ_mask,
                    # mask=occ_mask.permute(1, 2, 0),
                    # slow_color=slow_color,
                    # slow_means=slow_means,
                )

            print("[check] scales' max, min", trainer.get_attribute("scale").max().item(), trainer.get_attribute("scale").min().item())
            print("[check] depth_scale", trainer.depth_scale)
            print("[check] depth far, near", trainer.last_depth[trainer.last_depth>0].max(), trainer.last_depth[trainer.last_depth>0].min())
            print("[check] camera extr\n", trainer.extr)
            # print("[check] scales.max", trainer.get_attribute("scale").max())
            # print("[check] scales.min", trainer.get_attribute("scale").min())
            # print("[check] far depth", trainer.prev_depth_abs[trainer.prev_depth_abs>0].max())
            # print("[check] near depth", trainer.prev_depth_abs[trainer.prev_depth_abs>0].min())
            # print("[check] viewmat\n", trainer.get_viewmat)

            frames_sequence_optimize += frames
            frames_center_sequence_optimize += frames_center
            frames_depth_sequence_optimize += frames_depth
            frames_sequence.append(frames[-1])
            frames_center_sequence.append(frames_center[-1])
            frames_depth_sequence.append(frames_depth[-1])
        except Exception as e:
            # raise e
            if i == 0:
                raise e
            else:
                print(f"*************Error fitting {img_path}: {e}")
                continue
    
    # generate video
    for name, frames, fps in zip(
        ["sequence", "sequence_center", "sequence_depth", "sequence_optimize", "sequence_center_optimize", "sequence_depth_optimize"],
        [frames_sequence, frames_center_sequence, frames_depth_sequence, frames_sequence_optimize, frames_center_sequence_optimize, frames_depth_sequence_optimize],
        [5, 5, 5, 30, 30, 30]):

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