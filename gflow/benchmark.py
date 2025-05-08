from utils import image_path_to_tensor

import glob
import os
import tyro
import torch
import numpy as np
import piqa
from tqdm import tqdm
import json
import imageio
import pickle
from typing import Optional
from typing import Literal
from utils import image_path_to_tensor, process_frames_to_video, read_depth, extract_first_visible_points, find_closest_point
from utils.traj_visualizer import TrajVisualizer
from utils.measures import f_boundary, jaccard
import utils.render as render
from scipy.spatial.transform import Rotation
from utils.tapvid import compute_tapvid_metrics
from evo.core.trajectory import PoseTrajectory3D
from evo.core import sync
import evo.main_ape as main_ape
import evo.main_rpe as main_rpe
from evo.core.metrics import PoseRelation
from evo.core.metrics import Unit
from trainer import SimpleGaussian
import csv

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

def eval_tracking(
    sequence_path: str,
    load_path: str,
    tracking_results_path: Optional[str] = './tracking_results',
    background: Literal["black", "white"] = "black",
    resize: int = None,
    blur: bool = False,
) -> None:
    

    ckpt_paths = sorted(glob.glob(os.path.join(load_path, "ckpt", "*.tar")))

    img_paths = []
    for ckpt_path in ckpt_paths:
        img_path = os.path.join(sequence_path, os.path.basename(ckpt_path).split('.')[0]+'.png')
        if not os.path.exists(img_path):
            img_path = os.path.join(sequence_path, os.path.basename(ckpt_path).split('.')[0]+'.jpg')
        img_paths.append(img_path)

    # load the first frame
    gt_image0 = image_path_to_tensor(img_paths[0], resize=resize, blur=blur)
    trainer = SimpleGaussian(gt_image=gt_image0, num_points=1000, background=background)
    ckpt_paths = sorted(glob.glob(os.path.join(load_path, "ckpt", "*.tar")))
    trainer.load_checkpoint(ckpt_paths[0], show=False)

    pickle_path = os.path.join(sequence_path, "tracking.pkl")
    if not os.path.exists(pickle_path):
        return None, None, None
    with open(pickle_path, "rb") as f:
        traj_info = pickle.load(f)
    gt_tracks = traj_info["points"].copy()
    gt_occluded = traj_info["occluded"].copy()
    query_points = extract_first_visible_points(gt_tracks, gt_occluded)
    query_points[:,1] = query_points[:,1] * trainer.H
    query_points[:,2] = query_points[:,2] * trainer.W
    traj_pred = np.zeros((query_points.shape[0], len(ckpt_paths), 2), dtype=np.float32)
    queried_indexs = []
    cloest_indexs = []
    occulasions = np.empty((query_points.shape[0], len(ckpt_paths)))
    sequence_frames = []
    record_first_shift=[]

    for i, ckpt_path in enumerate(tqdm(ckpt_paths)):
        trainer.load_checkpoint(ckpt_path, show=False)
        output_img = imageio.imread(os.path.join(load_path, "images", 'img_'+os.path.basename(ckpt_path).split('.')[0]+'.png'))
        sequence_frames.append(output_img)
        depth_gt_path = os.path.join(sequence_path + "_depth_mast3r_s2", os.path.basename(ckpt_path).split('.')[0]+".npy")
        query_point_index = np.where(query_points[:,0] == i)[0].tolist()
        queried_indexs.extend(query_point_index)
        (uv, depth_p) = trainer.project_points(trainer.get_attribute("xyz"))
        input_group = [
            trainer.get_attribute("xyz"),
            trainer.get_attribute("scale"),
            trainer.get_attribute("rotate"),
            trainer.get_attribute("opacity"),
            trainer.get_attribute("rgb"),
            trainer.intr,
            trainer.get_extr(),
            trainer.bg,
            trainer.W,
            trainer.H,
        ]
        return_dict = render.render_multiple(input_group, ["uv", "depth", "depth_map", "depth_map_color"])
        uv = return_dict["uv"]
        depth = return_dict["depth"]
        depth_map = return_dict["depth_map"]

        query_points_add = query_points[query_point_index][:,-1:0:-1]
        cloest_points = find_closest_point(uv.cpu().detach().numpy(), query_points_add)
        cloest_indexs.extend(cloest_points)
        record_first_shift.extend(query_points_add-uv[cloest_points].cpu().detach().numpy())

        torch.cuda.empty_cache()

        # render trajectory
        traj_points = trainer.get_attribute('xyz')[cloest_indexs]
        (traj_uv, depth) = trainer.project_points(traj_points)
        # import pdb; pdb.set_trace()
        traj_pred[queried_indexs, i] = traj_uv.detach().cpu().numpy() + np.array(record_first_shift)


        # Find the corresponding rendered depth information for each tracking point
        depth_map_np = depth_map.detach().cpu().numpy()
        rendered_depth_info = depth_map_np[0, np.round(traj_uv[:, 1].detach().cpu().numpy()).astype(int), np.round(traj_uv[:, 0].detach().cpu().numpy()).astype(int)]

        occulasions[queried_indexs, i] = np.abs(rendered_depth_info - depth.detach().cpu().numpy().T) > 0.05

    pickle_path = os.path.join(tracking_results_path, os.path.basename(sequence_path), "eval")
    os.makedirs(pickle_path, exist_ok=True)
    traj_pred_path = os.path.join(pickle_path, "traj_pred.pkl")
    with open(traj_pred_path, "wb") as f:
        pickle.dump(traj_pred, f)
    occulasions_path = os.path.join(pickle_path, "occulasions.pkl")
    with open(occulasions_path, "wb") as f:
        pickle.dump(occulasions, f)

    query_points = query_points[None, :]
    gt_tracks = traj_info["points"].copy()
    gt_tracks = gt_tracks[None, :, :-1, :]
    gt_tracks[:, :, :, 0] = gt_tracks[:, :, :, 0]*trainer.W
    gt_tracks[:, :, :, 1] = gt_tracks[:, :, :, 1]*trainer.H
    gt_occluded = traj_info["occluded"].copy()
    gt_occluded = gt_occluded[None, :, :-1]
    pred_tracks = traj_pred.copy()
    pred_tracks = pred_tracks[None, :]
    pred_occluded = occulasions.copy()
    pred_occluded = pred_occluded[None, :]

    frames_video_torch = process_frames_to_video(sequence_frames)
    traj_visualizer = TrajVisualizer(save_dir=pickle_path, pad_value=0, linewidth=2, fps=5, show_first_frame=2)
    traj_visualizer.visualize(video=frames_video_torch, tracks=torch.from_numpy(pred_tracks).permute(0, 2, 1, 3), occulasions=torch.from_numpy(gt_occluded).permute(0,2,1), filename="sequence_traj_vis")
    traj_visualizer.visualize(video=frames_video_torch, tracks=torch.from_numpy(gt_tracks).permute(0, 2, 1, 3), occulasions=torch.from_numpy(gt_occluded).permute(0,2,1), filename="sequence_traj_vis_gt")

    
    gt_tracks[:, :, :, 0] = gt_tracks[:, :, :, 0]/trainer.W*255
    gt_tracks[:, :, :, 1] = gt_tracks[:, :, :, 1]/trainer.H*255
    pred_tracks[:, :, :, 0] = pred_tracks[:, :, :, 0]/trainer.W*255
    pred_tracks[:, :, :, 1] = pred_tracks[:, :, :, 1]/trainer.H*255
    tapvid_result = compute_tapvid_metrics(query_points, gt_occluded, gt_tracks, pred_occluded, pred_tracks, 'strided')
    tapvid_result["video_name"] =os.path.basename(sequence_path)
    print(tapvid_result)
    occ_acc = tapvid_result["occlusion_accuracy"][0]
    AJ = tapvid_result["average_jaccard"][0]
    APTS = tapvid_result["average_pts_within_thresh"][0]
    return occ_acc, AJ, APTS

def main(
    log_path: str,
    sequence_path: str,
    csv_name: str="metrics",
    eval_recon: bool = True,
    eval_track: bool = True,
    eval_seg: bool = True,
    eval_camera: bool = True,
) -> None:
    csv_dir = {}
    """Evaluate reconstruction quality (PSNR, SSIM, LPIPS)"""
    if eval_recon:
        print("Evaluating reconstruction quality (PSNR, SSIM, LPIPS)...")
        # recursively search for all images in the log path
        log_image_path = os.path.join(log_path, "images")
        log_image_paths = sorted(glob.glob(os.path.join(log_image_path, "img_0*.png"))) + sorted(glob.glob(os.path.join(log_image_path, "img_0*.jpg")))
        """4dgs"""
        # log_image_paths = sorted(glob.glob(os.path.join(log_path, "*.png")))
        
        psnr_list = []
        ssim_list = []
        lpips_list = []
        psnr  = piqa.PSNR()
        ssim  = piqa.SSIM()
        lpips = piqa.LPIPS()
        for image_path in tqdm(log_image_paths):
            image = image_path_to_tensor(image_path).permute(2,0,1).unsqueeze(0) # (B, C, H, W)
            image = torch.clamp(image, 0, 1)

            base_name = os.path.basename(image_path)
            base_name = base_name.split(".")[0].split("_")[-1]
            image_gt_path = os.path.join(sequence_path, f"{base_name}.jpg")
            """4dgs"""
            # image_gt = image_path_to_tensor(image_gt_path).permute(2,0,1).unsqueeze(0) # (B, C, H, W)
            """gflow"""
            image_gt = image_path_to_tensor(image_gt_path,resize=480).permute(2,0,1).unsqueeze(0) # (B, C, H, W)
            # clamp to [0, 1]
            image_gt = torch.clamp(image_gt, 0, 1)

            psnr_list.append(psnr(image, image_gt))
            ssim_list.append(ssim(image, image_gt))
            lpips_list.append(lpips(image, image_gt))
        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)
        avg_lpips = np.mean(lpips_list)
        print(f"Average PSNR: {avg_psnr}")
        csv_dir["PSNR"] = avg_psnr
        print(f"Average SSIM: {avg_ssim}")
        csv_dir["SSIM"] = avg_ssim
        print(f"Average LPIPS: {avg_lpips}")
        csv_dir["LPIPS"] = avg_lpips

    """Evaluate tracking quality (OA, AJ, APTS)"""
    if eval_track:
        print("Evaluating tracking quality (OA, AJ, APTS)...")
        occ_acc, AJ, APTS = eval_tracking(sequence_path=sequence_path, load_path=log_path, tracking_results_path=os.path.join(log_path, "tracking_results"))
        print(f"Occlusion Accuracy: {occ_acc}")
        csv_dir["Occlusion_Accuracy"] = occ_acc
        print(f"Average Jaccard: {AJ}")
        csv_dir["Average_Jaccard"] = AJ
        print(f"Average PTS within threshold: {APTS}")
        csv_dir["Average_PTS_within_threshold"] = APTS

    """Evaluate segmentation quality (J, F, J&F)"""
    if eval_seg:
        print("Evaluating segmentation quality (J, F, J&F)...")
        # recursively search for all images in the log path
        log_seg_path = os.path.join(log_path, "images_seg")
        log_seg_paths = sorted(glob.glob(os.path.join(log_seg_path, "move_mask_*.png"))) + sorted(glob.glob(os.path.join(log_seg_path, "move_mask_*.jpg")))
        j0_list = []
        f0_list = []
        jf0_list = []
        for seg_path in tqdm(log_seg_paths):
            seg0 = image_path_to_tensor(seg_path) # (H, W)
            seg0 = seg0 > 0.5

            base_name = os.path.basename(seg_path)
            base_name = base_name.split(".")[0].split("_")[-1]
            seg_gt_path = os.path.join(sequence_path+"_epipolar", f"{base_name}_open.png")
            seg_gt = image_path_to_tensor(seg_gt_path) # (H, W)
            seg_gt = seg_gt.mean(dim=-1, keepdim=True)
            seg_gt = seg_gt > 0.5

            seg0_np = seg0.squeeze().cpu().numpy()
            # seg1_np = seg1.squeeze().cpu().numpy()
            seg_gt_np = seg_gt.squeeze().cpu().numpy()

            # Calculate J
            j0 = jaccard.db_eval_iou(seg_gt_np, seg0_np)
            j0_list.append(j0)
            # Calculate F
            f0 = f_boundary.db_eval_boundary(seg0_np, seg_gt_np)
            f0_list.append(f0)
            # Calculate J&F
            jf0 = (j0 + f0) / 2
            jf0_list.append(jf0)

        avg_j0 = np.mean(j0_list)
        print(f"Average J 0: {avg_j0}")
        csv_dir["J_zero"] = avg_j0
        avg_f0 = np.mean(f0_list)
        print(f"Average F 0: {avg_f0}")
        csv_dir["F_zero"] = avg_f0
        avg_jf0 = np.mean(jf0_list)
        print(f"Average J&F 0: {avg_jf0}")
        csv_dir["J&F_zero"] = avg_jf0



    """Evaluate trajectory quality (ATE, RPE)"""
    if eval_camera:
        print("Evaluating trajectory quality (ATE, RPE)...")
        # Loading the camera poses
        camera_gt_base_path = os.path.join(sequence_path+"_camera_mast3r_s2")
        camera_gt_paths = sorted(glob.glob(os.path.join(camera_gt_base_path, "*.json")))
        tstamps = [float(os.path.basename(p).split(".")[0]) for p in camera_gt_paths]
        tum_gt_poses = []
        xyzs, wxyzs = [], []
        for camera_gt_path in tqdm(camera_gt_paths):
            camera_pose_gt = read_gt_camera(camera_gt_path)
            camera_pose_gt_inv = np.linalg.inv(camera_pose_gt)
            xyz = camera_pose_gt_inv[:3,-1]
            xyzs.append(xyz)
            R = Rotation.from_matrix(camera_pose_gt_inv[:3,:3])
            xyzw = R.as_quat()
            wxyz = np.array([xyzw[-1], xyzw[0], xyzw[1], xyzw[2]])
            wxyzs.append(wxyz)
            tum_gt_pose = np.concatenate([xyz, xyzw], 0)
            tum_gt_poses.append(tum_gt_pose)
        
        tum_gt_poses = np.stack(tum_gt_poses, 0)
        tum_gt_poses[:,:3] = tum_gt_poses[:,:3] - np.mean(tum_gt_poses[:,:3], 0, keepdims=True)
        tt = np.expand_dims(np.stack(tstamps, 0), -1)
        tum_gt_poses = np.concatenate([tt, tum_gt_poses], -1)
        xyzs = xyzs[:-1]
        wxyzs = wxyzs[:-1]
        tstamps = tstamps[:-1]
        traj_ref = PoseTrajectory3D(
            positions_xyz=np.stack(xyzs, 0),
            orientations_quat_wxyz=np.stack(wxyzs, 0),
            timestamps=np.array(tstamps)
        )
        poses_set = []
        ckpt_paths = sorted(glob.glob(os.path.join(log_path, "ckpt", "*.tar")))
        for ckpt_path in ckpt_paths:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            extr = checkpoint["extr"].detach().numpy()
            extr = np.concatenate([extr, np.array([[0,0,0,1]])], 0)
            c2w = np.linalg.inv(extr)
            poses_set.append(c2w[:3, :])
        poses_set = np.stack(poses_set)
        print("poses_set shape: ", poses_set.shape)

        eye = np.stack([np.eye(4)]*poses_set.shape[0], 0)
        eye[:, :3, :] = poses_set
        poses_set = np.copy(eye)

        poses = []
        tum_poses = []
        for pose_instance in poses_set:
            R, t = pose_instance[:3,:3], pose_instance[:3,3]
            rot = Rotation.from_matrix(R)
            quad = rot.as_quat() # xyzw
            quad_wxyz = np.array([quad[3], quad[0], quad[1], quad[2]])
            pose_t = np.concatenate([t, quad_wxyz], 0) # [tx, ty, tz, qw, qx, qy, qz]
            # tum pose format
            tum_pose_t = np.concatenate([t, quad], 0)
            poses.append(pose_t)
            tum_poses.append(tum_pose_t)
        poses = np.stack(poses, 0)
        tum_poses = np.stack(tum_poses, 0)
        traj_est = PoseTrajectory3D(
            positions_xyz=poses[:,0:3],
            orientations_quat_wxyz=poses[:,3:],
            timestamps=np.array(tstamps))
        
        # if less than 80% images got valid poses, we treat this sequence as failed.
        if len(poses) < 0.8 * len(tstamps):
            return None, None, None
        traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
        try:
            # ATE
            result = main_ape.ape(traj_ref, traj_est, est_name='traj',
                pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)
            ate = result.stats['rmse']

            # RPE rotation and translation
            delta_list = [1]
            rpe_rots, rpe_transs = [], []
            for delta in delta_list:
                result = main_rpe.rpe(traj_ref, traj_est, est_name='traj',
                    pose_relation=PoseRelation.rotation_angle_deg, align=True, correct_scale=True,
                    delta=delta, delta_unit=Unit.frames, rel_delta_tol=0.01, all_pairs=True)
                rot = result.stats['rmse']
                rpe_rots.append(rot)

            for delta in delta_list:
                result = main_rpe.rpe(traj_ref, traj_est, est_name='traj',
                    pose_relation=PoseRelation.translation_part, align=True, correct_scale=True,
                    delta=delta, delta_unit=Unit.frames, rel_delta_tol=0.01, all_pairs=True)
                trans = result.stats['rmse']
                rpe_transs.append(trans)
            rpe_trans, rpe_rot = np.mean(rpe_transs), np.mean(rpe_rots)

        except:
            print("Seq " + log_path + " not valid")
            ate = None
            rpe_trans, rpe_rot = None, None
        
        print(f"ATE: {ate}")
        csv_dir["ATE"] = ate
        print(f"RPE_t: {rpe_trans}")
        csv_dir["RPE_t"] = rpe_trans
        print(f"RPE_r: {rpe_rot}")
        csv_dir["RPE_r"] = rpe_rot


    # save metrics to csv
    """4dgs"""
    # with open(os.path.join(log_path, "../../..", f"{csv_name}.csv"), "w") as f:
    """gflow"""
    with open(os.path.join(log_path, f"{csv_name}.csv"), "w") as f:
        for key in csv_dir.keys():
            f.write(f"{key},{csv_dir[key]}\n")
        f.close()
    
    return csv_dir


if __name__ == "__main__":
    tyro.cli(main)