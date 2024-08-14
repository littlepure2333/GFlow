from typing import Optional
from pathlib import Path
from utils import image_path_to_tensor, read_depth, read_camera, read_mask, process_frames_to_video, process_traj_to_tracks, process_segm_mask, process_occu, print_color, find_closest_point
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
import time
from torch.nn import MaxPool2d
import cv2

# [ ] TODO background augmentation


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
    lambda_scale: float = 0.,
    lambda_delta: float = 0.,
    lambda_mov: float = 0.,
    lambda_rig: float = 0.,
    lambda_flow: float = 0.,
    background: Literal["black", "white", "cyan"] = "black",
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
    eps: float = 10, 
    min_samples: float = 20,
    logs_suffix: str = "_logs",
    load_extr: bool = True,
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
    sequence_traj_occlusion = []
    frames_move_seg = []


    # read all images(png, jpg, ...) in the folder and sort them
    img_paths = sorted(Path(sequence_path).glob("*.png")) + sorted(Path(sequence_path).glob("*.jpg"))
    if frame_range == -1:
        frame_range = len(img_paths)-1
    # img_depth_paths = sorted(Path(str(sequence_path)+'_depth_mast3r').glob("*.npy"))
    img_depth_paths = sorted(Path(str(sequence_path)+'_depth_mast3r_s2').glob("*.npy"))
    # img_depth_paths = sorted(Path(str(sequence_path)+'_depth_ZoeDepth').glob("*.npy"))
    # img_depth_paths = sorted(Path(str(sequence_path)+'_depth').glob("*.png")) + sorted(Path(str(sequence_path)+'_depth').glob("*.jpg"))
    img_paths = img_paths[frame_start:frame_start+frame_range][::skip_interval]
    img_depth_paths = img_depth_paths[frame_start:frame_start+frame_range][::skip_interval]
    # img_occ_paths = sorted(Path(str(sequence_path)+'_flow').glob("*occ_bwd.png")) + sorted(Path(str(sequence_path)+'_flow').glob("*occ_bwd.jpg"))
    img_occ_paths = sorted(Path(str(sequence_path)+'_flow_refine').glob("*occ_bwd.png")) + sorted(Path(str(sequence_path)+'_flow_refine').glob("*occ_bwd.jpg"))
    img_occ_paths = img_occ_paths[frame_start:frame_start+frame_range-1][::skip_interval]
    # flow_paths = sorted(Path(str(sequence_path)+'_flow').glob("*pred.flo"))
    flow_paths = sorted(Path(str(sequence_path)+'_flow_refine').glob("*pred.flo"))
    flow_paths = flow_paths[frame_start:frame_start+frame_range][::skip_interval]
    mask_paths = sorted(Path(str(sequence_path)+'_mask').glob("*.png"))
    mask_paths = mask_paths[frame_start:frame_start+frame_range][::skip_interval]
    mask_exist = len(mask_paths) > 0

    # move_mask_paths = sorted(Path(str(sequence_path)+'_move_mask').glob("*.png"))
    # move_mask_paths = move_mask_paths[frame_start:frame_start+frame_range][::skip_interval]

    move_mask_paths = sorted(Path(str(sequence_path)+'_epipolar').glob("*_open.png"))
    move_mask_paths = move_mask_paths[frame_start:frame_start+frame_range][::skip_interval]
    # camera_paths = sorted(Path(str(sequence_path)+'_camera_mast3r').glob("*.json"))
    camera_paths = sorted(Path(str(sequence_path)+'_camera_mast3r_s2').glob("*.json"))
    camera_paths = camera_paths[frame_start:frame_start+frame_range][::skip_interval]
    
    # read camera
    focal, pp, extr_list = read_camera(camera_paths)

    # read move masks
    move_masks = [read_mask(move_mask_path, resize=resize) for move_mask_path in move_mask_paths]
    
    
    start_time = time.time()
    # gt_images = [image_path_to_tensor(img_path, resize=resize, blur=blur) for img_path in img_paths]
    # gt_depths = [image_path_to_tensor(img_depth_path, resize=resize, blur=blur) / 255.0  for img_depth_path in img_depth_paths]

    # fit the first frame
    # frame0 = gt_images[0]
    save_name0 = os.path.basename(img_paths[0]).split('.')[0]
    gt_image0 = image_path_to_tensor(img_paths[0], resize=resize, blur=blur)
    H, W = gt_image0.shape[-2:]
    # depth_scale = float(focal) / float(H)
    # [ ] TODO decide the scale by distance 
    # depth_scale = float(focal) / 20.
    depth_scale = 1.
    # depth_offset = 10.
    # depth_scale = float(focal)
    # gt_depth0 = image_path_to_tensor(img_depth_paths[0], resize=resize, blur=blur)
    gt_depth0 = read_depth(img_depth_paths[0], resize=resize, depth_scale=depth_scale, depth_offset=depth_offset).unsqueeze(-1)
    gt_flow0 = readFlow(flow_paths[0], resize=resize, blur=blur)
    # [ ] TODO do not need to load flow at the first frame
    trainer = SimpleGaussian(gt_image=gt_image0, gt_depth=gt_depth0, gt_flow=gt_flow0,
                             num_points=num_points, background=background, sequence_path=sequence_path, logs_suffix=logs_suffix)
    trainer.load_camera(focal=focal, pp=pp)
    if load_extr:
        trainer.load_camera(extr=extr_list[0])
    trainer.init_gaussians_from_image(gt_image=gt_image0, gt_depth=gt_depth0, num_points=num_points)
    frames, frames_center, frames_depth, still_rgb_np, still_center_np, move_rgb_np, move_center_np, move_seg = trainer.train(
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
        lambda_scale=lambda_scale,
        densify_times=densify_times,
        densify_interval=densify_interval,
        grad_threshold=grad_threshold,
        eps=eps,
        min_samples=min_samples,
        move_mask=move_masks[0].to(trainer.device),
    )
    frames_sequence_optimize += frames
    frames_center_sequence_optimize += frames_center
    frames_depth_sequence_optimize += frames_depth
    frames_sequence.append(frames[-1])
    frames_center_sequence.append(frames_center[-1])
    frames_depth_sequence.append(frames_depth[-1])
    frames_move_seg.append(move_seg)
    if still_rgb_np is not None:
        frames_still_sequence.append(still_rgb_np)
        frames_still_center_sequence.append(still_center_np) 
        frames_move_sequence.append(move_rgb_np)
        frames_move_center_sequence.append(move_center_np)
    if mask_exist:
        mask0 = read_mask(mask_paths[0]).to(trainer.device)
        trainer.init_mask_prompt_pts(mask0, ckpt_name=save_name0)
    
    tap_vid = False
    grad_traj = True

    # render trajectory
    if traj_num:
        num_points = len(trainer.get_attribute("xyz"))
        interval = int(num_points / traj_num)
        traj_index = range(num_points)[traj_offset::interval]

        if grad_traj:
            erosion_kernel = np.ones((10,10),np.uint8)
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
            print("cloest_points.shape", cloest_points.shape)
            traj_index = cloest_points.tolist()

        if tap_vid:
            traj_info = pickle.load(os.path.join(sequence_path, "video.pkl"))
            query_points = traj_info["points"][:,0,:].copy()
            query_points[:,0] = query_points[:,0] * trainer.W
            query_points[:,1] = query_points[:,1] * trainer.H
            cloest_points = find_closest_point(trainer.last_uv, query_points)
            traj_index = cloest_points.tolist()

        (out_img, out_img_center, out_img_depth, 
         out_img_traj, out_img_traj_upon ) = trainer.eval(
            traj_index=traj_index,
            line_scale=0.5, 
            point_scale=2., 
            alpha=0.8,
            split_interval=split_interval,
        )
        frames_sequence_traj.append(out_img_traj)
        frames_sequence_traj_upon.append(out_img_traj_upon)
        traj_points = trainer.get_attribute('xyz')[traj_index]
        (traj_uv, depth) = trainer.project_points(traj_points)
        sequence_traj.append(traj_uv.detach().cpu().numpy())
        # traj_W_indices = np.round(traj_uv[:, 1].detach().cpu().numpy()).clip(0, frames_sequence[-1].shpe[0]-1).astype(int)
        # traj_H_indices = np.round(traj_uv[:, 0].detach().cpu().numpy()).clip(0, frames_sequence[-1].shape[1]-1).astype(int)
        # original_colors = trainer.get_attribute("rgb")[traj_index].detach().cpu().numpy()
        # projected_colors = frames_sequence[-1][traj_W_indices, traj_H_indices]/255.
        # color_difference = np.linalg.norm(projected_colors-original_colors, axis=1)a
        # sequence_traj_occlusion.append(color_difference)
        sequence_traj_occlusion.append(trainer.move_seg)

    # fit the subsequent frames
    # for img_path in img_paths[1:]:
    for i, img_path in enumerate(img_paths[1:]):
        save_name = os.path.basename(img_path).split('.')[0]
        print_color(f"[{i+1}/{len(img_paths) - 1}] fitting {img_path}")
        # trainer.load_checkpoint(trainer.checkpoint_path)
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

        if load_extr:
            trainer.load_camera(extr=extr_list[i+1])

        if camera_first:
            print_color(f"[{i+1}/{len(img_paths) - 1}] fitting camera-only first.................")
            frames, frames_center, frames_depth, still_rgb_np, still_center_np, move_rgb_np, move_center_np, move_seg = trainer.train(
                iterations=iterations_camera,
                # lr=lr,
                # lr=0.,
                # lr_rgbs=lr_rgbs,
                lr_camera=lr_camera_after,
                save_imgs=True,
                save_videos=False,
                save_ckpt=True,
                ckpt_name=save_name,
                # lambda_rgb=0.,
                lambda_rgb=lambda_rgb,
                lambda_depth=lambda_depth,
                # lambda_depth=0.,
                # lambda_var=lambda_var,
                lambda_var=0.,
                lambda_still=0.,
                lambda_flow=lambda_flow,
                densify_times=densify_times,
                densify_interval=densify_interval,
                grad_threshold=grad_threshold,
                camera_only=True,
                move_mask=move_masks[i+1].to(trainer.device),
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
            print_color(f"[check] camera extr: \n {trainer.get_extr()}")

        if iterations_after > 0:
            # print_color(f"[{i+1}/{len(img_paths) - 1}] Rehearshal stage for new points.................")
            print_color(f"[{i+1}/{len(img_paths) - 1}] Optimize all .................")
            frames, frames_center, frames_depth, still_rgb_np, still_center_np, move_rgb_np, move_center_np, move_seg = trainer.train(
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
                lambda_scale=lambda_scale,
                # lambda_flow=0.,
                lambda_flow=lambda_flow,
                densify_times=densify_times_after,
                densify_interval=densify_interval_after,
                densify_iter=densify_iter,
                grad_threshold=grad_threshold_after,
                # mask=None,
                mask=occ_mask,
                eps=eps,
                min_samples=min_samples,
                move_mask=move_masks[i+1].to(trainer.device),
                # mask=occ_mask.permute(1, 2, 0),
                # slow_color=slow_color,
                # slow_means=slow_means,
            )

        print_color(f"[check] scales' max, min: {trainer.get_attribute('scale').max().item()} {trainer.get_attribute('scale').min().item()}")
        # print_color("[check] depth_scale", trainer.depth_scale)
        print_color(f"[check] depth far, near {trainer.last_depth[trainer.last_depth>0].max()} {trainer.last_depth[trainer.last_depth>0].min()}")
        print_color(f"[check] camera intr: \n {trainer.intr}")
        print_color(f"[check] camera extr: \n {trainer.get_extr()}")
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
        frames_move_seg.append(move_seg)
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
                line_scale=0.5, 
                point_scale=2., 
                alpha=0.8,
                split_interval=split_interval,
            )
            frames_sequence_traj.append(out_img_traj)
            frames_sequence_traj_upon.append(out_img_traj_upon)
            traj_points = trainer.get_attribute('xyz')[traj_index]
            (traj_uv, depth) = trainer.project_points(traj_points)
            sequence_traj.append(traj_uv.detach().cpu().numpy())
            # traj_W_indices = np.round(traj_uv[:, 1].detach().cpu().numpy()).clip(0, frames_sequence[-1].shape[0]-1).astype(int)
            # traj_H_indices = np.round(traj_uv[:, 0].detach().cpu().numpy()).clip(0, frames_sequence[-1].shape[1]-1).astype(int)
            # # import pdb; pdb.set_trace()
            # original_colors = trainer.get_attribute("rgb")[traj_index].detach().cpu().numpy()*trainer.get_attribute("opacity")[traj_index].detach().cpu().numpy()
            # projected_colors = frames_sequence[-1][traj_W_indices, traj_H_indices]/255.
            # color_difference = np.linalg.norm(projected_colors-original_colors, axis=1)
            # sequence_traj_occlusion.append(color_difference)
            sequence_traj_occlusion.append(trainer.move_seg)


    end_time = time.time()
    total_time = end_time - start_time
    # convert to minutes
    total_time = total_time / 60

    # generate video
    for name, frames, fps in zip(
        ["sequence", "sequence_center", "sequence_depth", "sequence_optimize", "sequence_center_optimize", "sequence_depth_optimize", "sequence_still", "sequence_still_center", "sequence_move", "sequence_move_center", "sequence_move_seg"],
        [frames_sequence, frames_center_sequence, frames_depth_sequence, frames_sequence_optimize, frames_center_sequence_optimize, frames_depth_sequence_optimize, frames_still_sequence, frames_still_center_sequence, frames_move_sequence, frames_move_center_sequence, frames_move_seg],
        [5, 5, 5, 30, 30, 30, 5, 5, 5, 5, 5]):

        mp4_path = os.path.join(trainer.dir, f"{name}.mp4")
        save_video(mp4_path, frames, fps)
    
    # generate video for trajectories
    for name, frames, fps in zip(
        ["sequence_traj", "sequence_traj_upon"],
        [frames_sequence_traj, frames_sequence_traj_upon],
        [5, 5]):
        mp4_path = os.path.join(trainer.dir, f"{name}.mp4")
        save_video(mp4_path, frames, fps)
    
    import pickle
    pickle.dump(sequence_traj, open(os.path.join(trainer.dir, "sequence_traj.pkl"), "wb"))
    pickle.dump(frames_sequence, open(os.path.join(trainer.dir, "frames_sequence.pkl"), "wb"))
    # pickle.dump(trainer.first_move_seg, open(os.path.join(trainer.dir, "first_move_seg.pkl"), "wb"))
    pickle.dump(sequence_traj_occlusion, open(os.path.join(trainer.dir, "sequence_traj_occlusion.pkl"), "wb"))
    # save trajectory

    frames_video_torch = process_frames_to_video(frames_sequence)
    tracks_traj = process_traj_to_tracks(sequence_traj)
    # import pdb; pdb.set_trace()
    # segm_mask = process_segm_mask(trainer.first_move_seg)
    # occulasions = process_color_occu(sequence_traj_occlusion)
    # occulasions = occulasions>0.4
    # import pdb; pdb.set_trace()
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

    print_color(f"Total time: {total_time} mins", color="green")
    print_color(f"Total time: {total_time} mins", color="green")
    print_color(f"Total time: {total_time} mins", color="green")

    
def save_video(mp4_path, frames, fps):
    # Convert frames to uint8 before saving with imageio
    frames_uint8 = [frame.astype(np.uint8) for frame in frames]

    # Write the frames to an MP4 file
    with imageio.get_writer(mp4_path, fps=fps) as writer:  # Adjust fps as needed
        for frame in frames_uint8:
            writer.append_data(frame)


if __name__ == "__main__":
    tyro.cli(main)