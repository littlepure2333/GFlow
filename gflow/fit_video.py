from typing import Optional
from pathlib import Path
from utils import image_path_to_tensor, read_depth, read_camera, process_frames_to_video, process_traj_to_tracks
from trainer import SimpleGaussian
from traj_visualizer import TrajVisualizer
import os
import tyro
import torch
import imageio
import numpy as np
from typing import Literal
from helper import readFlow
import msplat

# TODO background augmentation


def print_color(msg, color="green"):
    if color == "red":
        print("\033[91m {}\033[00m".format(msg))
    elif color == "green":
        print("\033[92m {}\033[00m".format(msg))
    elif color == "yellow":
        print("\033[93m {}\033[00m".format(msg))
    elif color == "blue":
        print("\033[94m {}\033[00m".format(msg))
    elif color == "purple":
        print("\033[95m {}\033[00m".format(msg))
    elif color == "cyan":
        print("\033[96m {}\033[00m".format(msg))
    elif color == "white":
        print("\033[97m {}\033[00m".format(msg))
    else:
        print(msg)

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
    lambda_rgb: float = 1.,
    lambda_depth: float = 0.,
    lambda_still: float = 0.,
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
    densify_iter: int = 0,
    grad_threshold_after: float = 5e-3,
    densify_times_new: int = 1,
    densify_interval_new: int = 0,
    grad_threshold_new: float = 5e-3,
    lambda_var: float = 1e-7,
    skip_interval: int = 1,
    resize: int = None,
    blur: bool = False,
    loss_verbose: bool = False,
    depth_scale: float = 1.0,
    depth_offset: float = 0.,
    num_knn: int = 20,
    slow_color: bool = False,
    slow_means: bool = False,
    traj_num: int = 0,
    traj_offset: int = 0,
) -> None:
    
    frames_sequence_optimize = []
    frames_center_sequence_optimize = []
    frames_depth_sequence_optimize = []
    frames_sequence = []
    frames_center_sequence = []
    frames_depth_sequence = []
    frames_still_sequence = []
    frames_still_center_sequence = []
    frames_move_sequence = []
    frames_move_center_sequence = []
    frames_sequence_traj = []
    frames_sequence_traj_upon = []
    sequence_traj = []


    # read all images(png, jpg, ...) in the folder and sort them
    img_paths = sorted(Path(sequence_path).glob("*.png")) + sorted(Path(sequence_path).glob("*.jpg"))
    if frame_range == -1:
        frame_range = len(img_paths)
    img_depth_paths = sorted(Path(str(sequence_path)+'_depth_dust3r').glob("*.npy"))
    # img_depth_paths = sorted(Path(str(sequence_path)+'_depth_ZoeDepth').glob("*.npy"))
    # img_depth_paths = sorted(Path(str(sequence_path)+'_depth').glob("*.png")) + sorted(Path(str(sequence_path)+'_depth').glob("*.jpg"))
    img_paths = img_paths[frame_start:frame_start+frame_range][::skip_interval]
    img_depth_paths = img_depth_paths[frame_start:frame_start+frame_range][::skip_interval]
    img_occ_paths = sorted(Path(str(sequence_path)+'_flow').glob("*occ_bwd.png")) + sorted(Path(str(sequence_path)+'_flow').glob("*occ_bwd.jpg"))
    img_occ_paths = img_occ_paths[frame_start:frame_start+frame_range-1][::skip_interval]
    flow_paths = sorted(Path(str(sequence_path)+'_flow').glob("*pred.flo"))
    flow_paths = flow_paths[frame_start:frame_start+frame_range][::skip_interval]
    camera_paths = sorted(Path(str(sequence_path)+'_camera_dust3r').glob("*.json"))
    camera_paths = camera_paths[frame_start:frame_start+frame_range][::skip_interval]
    focal, pp, extr_list = read_camera(camera_paths)
    depth_scale = focal / 20
    

    # gt_images = [image_path_to_tensor(img_path, resize=resize, blur=blur) for img_path in img_paths]
    # gt_depths = [image_path_to_tensor(img_depth_path, resize=resize, blur=blur) / 255.0  for img_depth_path in img_depth_paths]

    # fit the first frame
    # frame0 = gt_images[0]
    save_name0 = os.path.basename(img_paths[0]).split('.')[0]
    gt_image0 = image_path_to_tensor(img_paths[0], resize=resize, blur=blur)
    # gt_depth0 = image_path_to_tensor(img_depth_paths[0], resize=resize, blur=blur)
    gt_depth0 = read_depth(img_depth_paths[0], resize=resize, depth_scale=depth_scale, depth_offset=depth_offset).unsqueeze(-1)
    gt_flow0 = readFlow(flow_paths[0], resize=resize, blur=blur)
    trainer = SimpleGaussian(gt_image=gt_image0, gt_depth=gt_depth0, gt_flow=gt_flow0,
                             num_points=num_points, background=background)
    trainer.load_camera(focal, pp)
    # trainer.load_camera(focal, pp, extr_list[0])
    trainer.init_gaussians_from_image(gt_image=gt_image0, gt_depth=gt_depth0, num_points=num_points)
    frames, frames_center, frames_depth, still_rgb_np, still_center_np, move_rgb_np, move_center_np = trainer.train(
        iterations=iterations_first,
        # lr=0.,
        lr=lr,
        # lr_rgbs=lr_rgbs,
        lr_camera=lr_camera,
        save_imgs=True,
        save_videos=True,
        save_ckpt=True,
        ckpt_name=save_name0,
        lambda_rgb=lambda_rgb,
        lambda_depth=lambda_depth,
        lambda_var=lambda_var,
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
    if still_rgb_np is not None:
        frames_still_sequence.append(still_rgb_np)
        frames_still_center_sequence.append(still_center_np)
        frames_move_sequence.append(move_rgb_np)
        frames_move_center_sequence.append(move_center_np)
    
    # render trajectory
    if traj_num:
        num_points = len(trainer.get_attribute("xyz"))
        interval = int(num_points / traj_num)
        traj_index = range(num_points)[traj_offset::interval]
        (out_img, out_img_center, out_img_depth, 
         out_img_traj, out_img_traj_upon ) = trainer.eval(
            traj_index=traj_index,
            line_scale=0.2, 
            point_scale=2., 
            alpha=0.8
        )
        frames_sequence_traj.append(out_img_traj)
        frames_sequence_traj_upon.append(out_img_traj_upon)
        traj_points = trainer.get_attribute('xyz')[traj_index]
        (traj_uv, depth) = trainer.project_points(traj_points)
        sequence_traj.append(traj_uv.detach().cpu().numpy())

    # fit the subsequent frames
    # for img_path in img_paths[1:]:
    for i, img_path in enumerate(img_paths[1:]):
        save_name = os.path.basename(img_path).split('.')[0]
        print_color(f"[{i+1}/{len(img_paths) - 1}] fitting {img_path}")
        trainer.load_checkpoint(trainer.checkpoint_path)
        gt_image = image_path_to_tensor(img_path, resize=resize, blur=blur)
        # gt_depth = image_path_to_tensor(img_depth_paths[i+1], resize=resize, blur=blur)
        gt_depth = read_depth(img_depth_paths[i+1], resize=resize, depth_scale=depth_scale, depth_offset=depth_offset).unsqueeze(-1)
        occ_mask = image_path_to_tensor(img_occ_paths[i], resize=resize, blur=blur)
        gt_flow = readFlow(flow_paths[i+1], resize=resize, blur=blur)
        # transform 1 is near to 0 is near
        # gt_depth = (1 - gt_depth) * depth_scale + depth_offset
        trainer.set_gt_image(gt_image)
        trainer.set_gt_depth(gt_depth)
        trainer.set_gt_flow(gt_flow)

        if camera_first:
            print_color(f"[{i+1}/{len(img_paths) - 1}] fitting camera-only first.................")
            frames, frames_center, frames_depth, still_rgb_np, still_center_np, move_rgb_np, move_center_np = trainer.train(
                iterations=iterations_camera,
                # lr=lr,
                lr=0.,
                # lr_rgbs=lr_rgbs,
                lr_camera=lr_camera_after,
                save_imgs=True,
                save_videos=False,
                save_ckpt=True,
                ckpt_name=save_name,
                lambda_rgb=0.,
                lambda_depth=lambda_depth,
                # lambda_depth=0.,
                lambda_var=lambda_var,
                lambda_still=0.,
                lambda_flow=lambda_flow,
                densify_times=densify_times,
                densify_interval=densify_interval,
                grad_threshold=grad_threshold,
                camera_only=True,
            )
            frames_sequence_optimize += frames
            frames_center_sequence_optimize += frames_center
            frames_depth_sequence_optimize += frames_depth
            # frames_sequence.append(frames[-1])
            # frames_center_sequence.append(frames_center[-1])
            # frames_depth_sequence.append(frames_depth[-1])

            print_color(f"[check] scales' max, min: {trainer.get_attribute('scale').max().item()} {trainer.get_attribute('scale').min().item()}")
            # print_color("[check] depth_scale", trainer.depth_scale)
            print_color(f"[check] depth far, near {trainer.last_depth[trainer.last_depth>0].max()} {trainer.last_depth[trainer.last_depth>0].min()}")
            print_color(f"[check] camera intr: \n {trainer.intr}")
            print_color(f"[check] camera extr: \n {trainer.extr}")

        if iterations_after > 0:
            # print_color(f"[{i+1}/{len(img_paths) - 1}] Rehearshal stage for new points.................")
            print_color(f"[{i+1}/{len(img_paths) - 1}] Optimize all .................")
            frames, frames_center, frames_depth, still_rgb_np, still_center_np, move_rgb_np, move_center_np = trainer.train(
                iterations=iterations_after,
                lr=lr_after,
                # lr_camera=lr_camera_after,
                lr_camera=0.,
                save_imgs=True,
                save_videos=False,
                save_ckpt=True,
                ckpt_name=save_name,
                # fps=fps,
                lambda_rgb=lambda_rgb,
                # lambda_depth=0.,
                lambda_depth=lambda_depth,
                lambda_var=lambda_var,
                lambda_still=lambda_still,
                # lambda_flow=0.,
                lambda_flow=lambda_flow,
                densify_times=densify_times_after,
                densify_interval=densify_interval_after,
                densify_iter=densify_iter,
                grad_threshold=grad_threshold_after,
                # mask=None,
                mask=occ_mask,
                # mask=occ_mask.permute(1, 2, 0),
                # slow_color=slow_color,
                # slow_means=slow_means,
            )

        print_color(f"[check] scales' max, min: {trainer.get_attribute('scale').max().item()} {trainer.get_attribute('scale').min().item()}")
        # print_color("[check] depth_scale", trainer.depth_scale)
        print_color(f"[check] depth far, near {trainer.last_depth[trainer.last_depth>0].max()} {trainer.last_depth[trainer.last_depth>0].min()}")
        print_color(f"[check] camera intr: \n {trainer.intr}")
        print_color(f"[check] camera extr: \n {trainer.extr}")
        # print_color("[check] scales.max", trainer.get_attribute("scale").max())
        # print_color("[check] scales.min", trainer.get_attribute("scale").min())
        # print_color("[check] far depth", trainer.prev_depth_abs[trainer.prev_depth_abs>0].max())
        # print_color("[check] near depth", trainer.prev_depth_abs[trainer.prev_depth_abs>0].min())
        # print_color("[check] viewmat\n", trainer.get_viewmat)

        frames_sequence_optimize += frames
        frames_center_sequence_optimize += frames_center
        frames_depth_sequence_optimize += frames_depth
        frames_sequence.append(frames[-1])
        frames_center_sequence.append(frames_center[-1])
        frames_depth_sequence.append(frames_depth[-1])
        if still_rgb_np is not None:
            frames_still_sequence.append(still_rgb_np) 
            frames_still_center_sequence.append(still_center_np)
            frames_move_sequence.append(move_rgb_np)
            frames_move_center_sequence.append(move_center_np)
        # render trajectory
        if traj_num:
            (out_img, out_img_center, out_img_depth, 
            out_img_traj, out_img_traj_upon ) = trainer.eval(
                traj_index=traj_index,
                line_scale=0.2, 
                point_scale=2., 
                alpha=0.8
            )
            frames_sequence_traj.append(out_img_traj)
            frames_sequence_traj_upon.append(out_img_traj_upon)
            traj_points = trainer.get_attribute('xyz')[traj_index]
            (traj_uv, depth) = trainer.project_points(traj_points)
            sequence_traj.append(traj_uv.detach().cpu().numpy())
    
    # generate video
    for name, frames, fps in zip(
        ["sequence", "sequence_center", "sequence_depth", "sequence_optimize", "sequence_center_optimize", "sequence_depth_optimize", "sequence_still", "sequence_still_center", "sequence_move", "sequence_move_center"],
        [frames_sequence, frames_center_sequence, frames_depth_sequence, frames_sequence_optimize, frames_center_sequence_optimize, frames_depth_sequence_optimize, frames_still_sequence, frames_still_center_sequence, frames_move_sequence, frames_move_center_sequence],
        [5, 5, 5, 30, 30, 30, 5, 5, 5, 5]):

        mp4_path = os.path.join(trainer.dir, f"{name}.mp4")
        save_video(mp4_path, frames, fps)
    
    # generate video for trajectories
    for name, frames, fps in zip(
        ["sequence_traj", "sequence_traj_upon"],
        [frames_sequence_traj, frames_sequence_traj_upon],
        [5, 5]):
        mp4_path = os.path.join(trainer.dir, f"{name}.mp4")
        save_video(mp4_path, frames, fps)
    
    # import pickle
    # pickle.dump(sequence_traj, open(os.path.join(trainer.dir, "sequence_traj.pkl"), "wb"))
    # pickle.dump(frames_sequence, open(os.path.join(trainer.dir, "frames_sequence.pkl"), "wb"))
    # save trajectory
    frames_video_torch = process_frames_to_video(frames_sequence)
    tracks_traj = process_traj_to_tracks(sequence_traj)

    traj_visualizer = TrajVisualizer(save_dir=trainer.dir, pad_value=0, linewidth=3, fps=5)
    traj_visualizer.visualize(video=frames_video_torch,tracks=tracks_traj, filename="sequence_traj_vis")

    
def save_video(mp4_path, frames, fps):
    # Convert frames to uint8 before saving with imageio
    frames_uint8 = [frame.astype(np.uint8) for frame in frames]

    # Write the frames to an MP4 file
    with imageio.get_writer(mp4_path, fps=fps) as writer:  # Adjust fps as needed
        for frame in frames_uint8:
            writer.append_data(frame)


if __name__ == "__main__":
    tyro.cli(main)