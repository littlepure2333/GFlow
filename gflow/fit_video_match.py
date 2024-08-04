from typing import Optional
from pathlib import Path
from utils import image_path_to_tensor, read_depth, read_camera, read_mask, process_frames_to_video, process_traj_to_tracks, process_segm_mask, process_occu, print_color, find_closest_point
from helper import readFlow
import cv2
from traj_visualizer import TrajVisualizer
# from utils import image_path_to_tensor, apply_float_colormap
from trainer import SimpleGaussian

import glob
from helper import hungarian_match

import os
import tyro
import torch
import imageio
import numpy as np
from typing import Literal
from torch.nn import MaxPool2d
import pickle

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
    depth_offset: float = 0.0,
    load_path: str = "/home/wangshizun/projects/gsplat/logs/2024_03_01-16_51_25",
    traj_only: bool = False,
    fixed_camera: bool = False,
    traj_num: int = 200,
    traj_offset: int = 0,
) -> None:
    
    frames_sequence = []
    frames_center_sequence = []
    frames_depth_sequence = []
    frames_center_sequence_traj = []
    frames_depth_sequence_traj = []
    frames_sequence_traj = []
    frames_sequence_traj_upon = []
    sequence_traj = []
    sequence_traj_occlusion = []
    frames_move_seg = []

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
        img_depth_path = os.path.join(sequence_path+'_depth_dust3r', os.path.basename(ckpt_path).split('.')[0]+'.png')
        if not os.path.exists(img_depth_path):
            img_depth_path = os.path.join(sequence_path+'_depth_dust3r', os.path.basename(ckpt_path).split('.')[0]+'.jpg')
        img_depth_paths.append(img_depth_path)

    flow_paths = []
    for ckpt_path in ckpt_paths:
        img_occ_path = os.path.join(sequence_path+'_flow', os.path.basename(ckpt_path).split('.')[0]+'_pred.flo')
        flow_paths.append(img_occ_path)

    img_occ_paths = []
    for ckpt_path in ckpt_paths:
        img_occ_path = os.path.join(sequence_path+'_flow', os.path.basename(ckpt_path).split('.')[0]+'_occ_bwd.png')
        if not os.path.exists(img_occ_path):
            img_occ_path = os.path.join(sequence_path+'_flow', os.path.basename(ckpt_path).split('.')[0]+'_occ_bwd.jpg')
        img_occ_paths.append(img_occ_path)

    mask_paths = []
    for ckpt_path in ckpt_paths:
        img_occ_path = os.path.join(sequence_path+'_seg', os.path.basename(ckpt_path).split('.')[0]+'.png')
        if not os.path.exists(img_occ_path):
            img_occ_path = os.path.join(sequence_path+'_seg', os.path.basename(ckpt_path).split('.')[0]+'.jpg')
        mask_paths.append(img_occ_path)

    camera_paths = []
    for ckpt_path in ckpt_paths:
        img_occ_path = os.path.join(sequence_path+'_camera_dust3r', os.path.basename(ckpt_path).split('.')[0]+'.json')
        camera_paths.append(img_occ_path)

    # load the first frame
    gt_image0 = image_path_to_tensor(img_paths[0], resize=resize, blur=blur)
    # gt_depth0 = read_depth(img_depth_paths[0], resize=resize, depth_scale=depth_scale, depth_offset=depth_offset).unsqueeze(-1)
    gt_flow0 = readFlow(flow_paths[0], resize=resize, blur=blur)
    trainer = SimpleGaussian(gt_image=gt_image0, num_points=num_points, background=background)
    print("[check] trainer.W", trainer.W)
    print("[check] trainer.H", trainer.H)
    ckpt_paths = sorted(glob.glob(os.path.join(load_path, "ckpt_match", "*.tar")))
    ckpt_paths = sorted(glob.glob(os.path.join(load_path, "ckpt", "*.tar")))
    trainer.load_checkpoint(ckpt_paths[0])

    num_points = len(trainer.get_attribute("xyz"))
    # traj_num = 200 # 0.4% of the points
    interval = int(num_points / traj_num)
    traj_index = range(num_points)[traj_offset::interval]


    tap_vid = False
    grad_traj = True
    split_interval = None

    # render trajectory
    if traj_num:
        num_points = len(trainer.get_attribute("xyz"))
        interval = int(num_points / traj_num)
        traj_index = range(num_points)[traj_offset::interval]

        if grad_traj:
            erosion_kernel = np.ones((20,20),np.uint8)
            move_seg = trainer.move_seg
            move_seg_erosion = cv2.erode(move_seg, erosion_kernel, iterations = 1)

            still_seg = 255 - move_seg
            still_seg_erosion = cv2.erode(still_seg, erosion_kernel, iterations = 1)

            # Set the query points to a grid-like pattern according to specific stride
            stride_still = 100
            stride_moving = 20
            sparse_query_points = []
            for i in range(stride_still, trainer.H, stride_still):
                for j in range(stride_still, trainer.W, stride_still):
                    if still_seg_erosion[i, j]: # within still region
                    # if True:
                        sparse_query_points.append(np.array([j, i]))
                        # print("add still")
            sparse_query_points = np.array(sparse_query_points)
            concentrated_query_points = []
            for i in range(stride_moving, trainer.H-stride_moving, stride_moving):
                for j in range(stride_moving, trainer.W-stride_moving, stride_moving):
                    if move_seg_erosion[i, j]: # within moving region
                    # if True:
                        concentrated_query_points.append(np.array([j, i]))
                        # print("add moving")
            
            concentrated_query_points = np.array(concentrated_query_points)
            uv = trainer.last_uv.detach().cpu().numpy()
            still_mask = trainer.still_mask.detach().cpu().numpy()
            # erode the still mask
            # still_mask = -MaxPool2d(kernel_size=5, stride=1, padding=2)(-still_mask.float())
            # dilate the still mask
            # still_mask = MaxPool2d(kernel_size=5, stride=1, padding=2)(still_mask)
            sparse_cloest_points = find_closest_point(uv, sparse_query_points)
            concentrated_cloest_points = find_closest_point(uv, concentrated_query_points)
            cloest_points_still = sparse_cloest_points[still_mask[sparse_cloest_points]]
            cloest_points_move = concentrated_cloest_points[~still_mask[concentrated_cloest_points]]
            split_interval = cloest_points_still.shape[0]
            cloest_points = np.concatenate([cloest_points_still, cloest_points_move])
            traj_index = cloest_points.tolist()

        if tap_vid:
            traj_info = pickle.load(os.path.join(sequence_path, "video.pkl"))
            query_points = traj_info["points"][:,0,:].copy()
            query_points[:,0] = query_points[:,0] * trainer.W
            query_points[:,1] = query_points[:,1] * trainer.H
            cloest_points = find_closest_point(trainer.last_uv, query_points)
            traj_index = cloest_points.tolist()




    for i, ckpt_path in enumerate(ckpt_paths):
        print(f"[{i}/{len(ckpt_paths) - 1}] Gaussian splatting {ckpt_path}")
        trainer.load_checkpoint(ckpt_path)
        
        (out_img, out_img_center, out_img_depth, 
         out_img_traj, out_img_traj_upon ) = trainer.eval(
            traj_index=traj_index,
            line_scale=0.5, 
            point_scale=2., 
            alpha=0.8,
            split_interval=split_interval,
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

        # render trajectory
        traj_points = trainer.get_attribute('xyz')[traj_index]
        (traj_uv, depth) = trainer.project_points(traj_points)
        sequence_traj.append(traj_uv.detach().cpu().numpy())
        sequence_traj_occlusion.append(trainer.move_seg)

    print("frames_sequence_traj_upon.length", len(frames_sequence_traj_upon))
    # generate video
    for name, frames, fps in zip(
        ["sequence", "sequence_center", "sequence_depth", "sequence_traj", "sequence_traj_upon"],
        [frames_sequence, frames_center_sequence, frames_depth_sequence, frames_sequence_traj, frames_sequence_traj_upon],
        [5, 5, 5, 5, 5, 5, 5]):
        mp4_path = os.path.join(trainer.dir, f"{name}.mp4")
        save_video(mp4_path, frames, fps)

    frames_video_torch = process_frames_to_video(frames_sequence)
    print("frames_video_torch.shape", frames_video_torch.shape)
    tracks_traj = process_traj_to_tracks(sequence_traj)
    occulasions = process_occu(sequence_traj_occlusion, tracks_traj)

    traj_visualizer = TrajVisualizer(save_dir=trainer.dir, pad_value=0, linewidth=2, fps=5, show_first_frame=2)
    # traj_visualizer.visualize(video=frames_video_torch,tracks=tracks_traj, segm_mask=segm_mask, occulasions=occulasions, filename="sequence_traj_vis", compensate_for_camera_motion=True)
    traj_visualizer.visualize(video=frames_video_torch,tracks=tracks_traj, occulasions=occulasions, filename="sequence_traj_vis", still_length=cloest_points_still.shape[0])
    if grad_traj:
        traj_visualizer = TrajVisualizer(save_dir=trainer.dir, pad_value=0, linewidth=2, fps=5, show_first_frame=2)
        # traj_visualizer.visualize(video=frames_video_torch,tracks=tracks_traj, segm_mask=segm_mask, occulasions=occulasions, filename="sequence_traj_vis", compensate_for_camera_motion=True)
        traj_visualizer.visualize(video=frames_video_torch,tracks=tracks_traj[:,:,:cloest_points_still.shape[0],:], occulasions=occulasions, filename="sequence_traj_vis_still")
        traj_visualizer = TrajVisualizer(save_dir=trainer.dir, pad_value=0, linewidth=2, fps=5, show_first_frame=2)
        # traj_visualizer.visualize(video=frames_video_torch,tracks=tracks_traj, segm_mask=segm_mask, occulasions=occulasions, filename="sequence_traj_vis", compensate_for_camera_motion=True)
        traj_visualizer.visualize(video=frames_video_torch,tracks=tracks_traj[:,:,-len(cloest_points_move):,:], occulasions=occulasions, filename="sequence_traj_vis_move")

    

def save_video(mp4_path, frames, fps):
    # Convert frames to uint8 before saving with imageio
    frames_uint8 = [frame.astype(np.uint8) for frame in frames]

    # Write the frames to an MP4 file
    with imageio.get_writer(mp4_path, fps=fps) as writer:  # Adjust fps as needed
        for frame in frames_uint8:
            writer.append_data(frame)


if __name__ == "__main__":
    tyro.cli(main)