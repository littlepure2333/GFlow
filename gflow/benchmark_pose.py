import os
import numpy as np
import argparse
import glob
from scipy.spatial.transform import Rotation
import evo
from evo.core.trajectory import PoseTrajectory3D
from evo.core import sync
import evo.main_ape as main_ape
import evo.main_rpe as main_rpe
from evo.core.metrics import PoseRelation
from evo.core.metrics import Unit
import sintel_io
from scipy.spatial.transform import Rotation
import torch

def eval_one_seq(method, gt_dir, input_dir):
    print("Evaluating sequence: ", input_dir)
    # pre-process the sintel-format poses to tum format poses
    gt_pose_lists = sorted(glob.glob(os.path.join(gt_dir, '*.cam')))
    tstamps = [float(x.split('/')[-1][:-4].split("_")[-1]) for x in gt_pose_lists]
    gt_poses = [sintel_io.cam_read(f)[1] for f in gt_pose_lists]
    xyzs, wxyzs = [], []
    tum_gt_poses = []
    print("GT poses length: ", len(gt_poses))
    for gt_pose in gt_poses:
        gt_pose = np.concatenate([gt_pose, np.array([[0,0,0,1]])], 0)
        gt_pose_inv = np.linalg.inv(gt_pose) # world2cam -> cam2world
        xyz = gt_pose_inv[:3,-1]
        xyzs.append(xyz)
        R = Rotation.from_matrix(gt_pose_inv[:3,:3])
        xyzw = R.as_quat() # scalar-last for scipy
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

    if method == 'ParticleSfM':
        # ParticleSfM
        all_pose_files = sorted(glob.glob(input_dir+'/particleSfM/colmap_outputs_converted/poses/*.txt'))
        all_poses = []
        for pose_file in all_pose_files:
            pose = np.loadtxt(pose_file)
            if pose.shape[0] == 3:
                pose = np.concatenate([pose, np.array([[0,0,0,1]])], 0)
            cam2world = np.linalg.inv(pose)
            all_poses.append(cam2world[:3, :])
        poses_est = np.stack(all_poses)

    elif method == 'BARF':
        # BARF
        poses_est = np.load(input_dir+'/BARF_poses.npy')
        poses_est = np.concatenate([poses_est[..., 1:2], -poses_est[..., :1], poses_est[..., 2:4]], -1)

    elif method == 'nerfmm':
        # NeRF--
        poses_est = np.load(input_dir+'/nerfmm_poses.npy')[:, :3, :]
        poses_est = np.concatenate([poses_est[..., 1:2], -poses_est[..., :1], poses_est[..., 2:4]], -1)

    elif method == 'Ours':
        # Ours
        poses_est = np.load(input_dir+'/poses_bounds_ours.npy')[:, :15].reshape(-1, 3, 5)[:, :, :4] # shape: (N, 3, 4)
        poses_est = np.concatenate([poses_est[..., 1:2], -poses_est[..., :1], poses_est[..., 2:4]], -1)
    elif method == 'gflow':
        # gflow
        # create a empty list to store the poses
        poses_est = []
        # list all files in the input_dir
        ckpt_paths = sorted(glob.glob(os.path.join(input_dir, "ckpt", "*.tar")))
        for ckpt_path in ckpt_paths:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            extr = checkpoint["extr"].detach().numpy() # world2cam
            print(extr)
            extr = np.concatenate([extr, np.array([[0,0,0,1]])], 0)
            # print(extr.shape)
            c2w = np.linalg.inv(extr)
            # c2w[:3, 1:3]*= -1
            # c2w = extr
            poses_est.append(c2w[:3, :])
        
        print("Est poses length: ", len(poses_est))
        # concate the poses
        poses_est = np.stack(poses_est)



    eye = np.stack([np.eye(4)]*poses_est.shape[0], 0)
    eye[:, :3, :] = poses_est
    poses_est = np.copy(eye)

    # pre-process the colmap-format poses to tum format poses
    # pose_names = sorted(glob.glob(input_dir + "/*.txt"))
    poses = []
    tum_poses = []
    for pose_instance in poses_est:
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
    # np.savetxt("./Ours.txt", tum_poses)
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
        print("Seq " + input_dir + " not valid")
        ate = None
        rpe_trans, rpe_rot = None, None
    print(ate)
    print(rpe_trans)
    print(rpe_rot)
    return ate, rpe_trans, rpe_rot

scenes = [
    'alley_2', 'ambush_4', 'ambush_5', 'ambush_6', 'cave_2', 'cave_4', 
    'market_2', 'market_5', 'market_6', 'shaman_3', 
    # 'sleeping_1', 'sleeping_2', 
    'temple_2', 'temple_3']
method = 'gflow'  # ParticleSfM, BARF, nerfmm, Ours
ate_list = []
rpe_trans_list = []
rpe_rot_list = []
all_path = "/home/wangshizun/projects/msplat/data_share/sintel/images"
log_suffix = "logs_cam_init_only"
log_suffix = "logs_cam_optim_only"
csv_content = ""
for sq in scenes:
    gt_dir = os.path.join(all_path, sq, f"{sq}_cam")
    log_dir_latest = os.path.join(all_path, sq, f"{sq}_{log_suffix}/0_latest")
    if os.path.exists(log_dir_latest):
        # the subfolder in the log_path
        subfolders = glob.glob(os.path.join(log_dir_latest, "*"))
        if len(subfolders) > 0:
            input_dir = subfolders[0]
        else:
            print(f"No valid log path found in: {log_dir_latest}")
            continue
    
    ate, rpe_trans, rpe_rot = eval_one_seq(method, gt_dir, input_dir)
    csv_content += "{}\t{}\t{}\t{}\n".format(sq, ate, rpe_trans, rpe_rot)
    ate_list.append(ate)
    rpe_trans_list.append(rpe_trans)
    rpe_rot_list.append(rpe_rot)

avg_ate = np.mean(np.array(ate_list))
avg_rpe_trans = np.mean(np.array(rpe_trans_list))
avg_rpe_rot = np.mean(np.array(rpe_rot_list))
csv_content += "Average\t{}\t{}\t{}\n".format(avg_ate, avg_rpe_trans, avg_rpe_rot)
print("average")
print(avg_ate)
print(avg_rpe_trans)
print(avg_rpe_rot)
# save to csv
csv_path = f"{all_path}/pose_{log_suffix}.csv"
with open(f"{log_suffix}_eval.csv", "w") as f:
    f.write(csv_content)


