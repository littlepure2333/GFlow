import pickle
import torch
import numpy as np
import msplat
from utils import *
from traj_visualizer import TrajVisualizer
from tapvid import compute_tapvid_metrics

frames_sequence = pickle.load(open("/home/jiangzhenxiang/Developing/gflow/logs/2024_05_18-22_19_02/frames_sequence.pkl", "rb"))
sequence_traj = pickle.load(open("/home/jiangzhenxiang/Developing/gflow/logs/2024_05_18-22_19_02/sequence_traj.pkl", "rb"))
sequence_traj_occulusion = pickle.load(open("/home/jiangzhenxiang/Developing/gflow/logs/2024_05_18-22_19_02/sequence_traj_occlusion.pkl", "rb"))

frames_video_torch = process_frames_to_video(frames_sequence)
tracks_traj = process_traj_to_tracks(sequence_traj)
occulasions = process_occu(sequence_traj_occulusion, tracks_traj)

def check_coord_within(coord, W, H):
    return coord[0] >= 0 and coord[0] < W and coord[1] >= 0 and coord[1] < H

def get_traj_all_within(tracks_traj, occulasions, W, H):
    new_tracks_traj = []
    new_occulasions = []
    for b in range(tracks_traj.shape[0]):
        temp_tracks_list = []
        temp_occulasions_list = []
        for i in range(tracks_traj.shape[2]):
            all_within = True
            for t in range(tracks_traj.shape[1]):
                coord = (tracks_traj[b, t, i, 0].item(), tracks_traj[b, t, i, 1].item())
                if not check_coord_within(coord, W, H):
                    # print(coord)
                    all_within = False
                    break
            if all_within:
                temp_tracks_list.append(tracks_traj[b, :, i, :])
                temp_occulasions_list.append(occulasions[b, :, i])
        new_tracks_traj.append(np.stack(temp_tracks_list))
        new_occulasions.append(np.stack(temp_occulasions_list))
    new_tracks_traj = torch.from_numpy(np.stack(new_tracks_traj))
    new_tracks_traj = new_tracks_traj.permute(0, 2, 1, 3)
    new_occulasions = torch.from_numpy(np.stack(new_occulasions))
    new_occulasions = new_occulasions.permute(0, 2, 1)
    return new_tracks_traj, new_occulasions

# print(tracks_traj.shape)
# print(occulasions.shape)

tracks_traj, occulasions = get_traj_all_within(tracks_traj, occulasions, frames_video_torch.shape[4], frames_video_torch.shape[3])

# print(tracks_traj.shape)
# print(occulasions.shape)

def get_query_points(tracks_traj):
    query_points = []
    for b in range(tracks_traj.shape[0]):
        temp_queru_points = []
        for i in range(tracks_traj.shape[2]):
            temp_queru_points.append(np.array([0, tracks_traj[0,0,i,1], tracks_traj[0,0,i,0]]))
        temp_queru_points = np.stack(temp_queru_points)
        query_points.append(temp_queru_points)
    query_points = np.stack(query_points)
    # print(query_points.shape)
    # query_points = np.einsum('ijk->jik', query_points)
    # query_points[:, :, 0], query_points[:, :, 1], query_points[:, :, 2] = 0, query_points[:, :, 1], query_points[:, :, 0]
    # query_points = query_points.permute(1, 0, 2) 
    return query_points

query_points = get_query_points(tracks_traj)
# print(query_points.shape)

def get_pred_occluded(occulasions):
    pred_occlued = []
    for i in range(occulasions.shape[2]):
        pred_occlued.append(occulasions[:, :, i])
    pred_occlued = np.stack(pred_occlued)
    pred_occlued = np.einsum('ijk->jik', pred_occlued)
    # pred_occlued = pred_occlued.permute(1, 0, 2)
    return pred_occlued

pred_occlued = get_pred_occluded(occulasions)
# print(pred_occlued.shape)

def get_pred_tracks(tracks_traj):
    pred_tracks = []
    for i in range(tracks_traj.shape[2]):
        pred_tracks.append(tracks_traj[:, :, i])
    pred_tracks = np.stack(pred_tracks)
    pred_tracks = np.einsum('ijkl->jikl', pred_tracks)
    # pred_tracks = pred_tracks.permute(1, 0, 2, 3)
    return pred_tracks

pred_tracks = get_pred_tracks(tracks_traj)
# print(pred_tracks.shape)

# %TODO Load GT Data
# Add noise to the GT data(copied from pred data), Removed When use the real Ground Truth
gt_occluded = pred_occlued
gt_tracks = pred_tracks
gt_occluded = np.random.rand(*gt_occluded.shape) > 0.5
gt_tracks = gt_tracks + np.random.randn(*gt_tracks.shape) * 0.5

print(compute_tapvid_metrics(query_points, gt_occluded, gt_tracks, pred_occlued, pred_tracks, 'strided'))