import torch
import numpy as np
import math

def process_traj_to_tracks(sequence_traj):
    tracks_traj = torch.from_numpy(np.stack(sequence_traj))
    tracks_traj = tracks_traj[None, :].float()
    return tracks_traj

def process_occu(sequence_traj_occlusion, tracks):
    tracks = tracks.clone()
    tracks[:, :, :, 0] = tracks[:, :, :, 0].clip(0, sequence_traj_occlusion[0].shape[1]-1)
    tracks[:, :, :, 1] = tracks[:, :, :, 1].clip(0, sequence_traj_occlusion[0].shape[0]-1)
    occulasions = torch.full((1, len(sequence_traj_occlusion), tracks.shape[2]), False)
    moving_part = []
    for j in range(tracks.shape[2]):
        moving_part.append(sequence_traj_occlusion[0][round(tracks[0, 0, j, 1].item()), round(tracks[0, 0, j, 0].item())])
    for i in range(len(sequence_traj_occlusion)):
        move_seg = sequence_traj_occlusion[i]
        for j in range(tracks.shape[2]):
            occulasions[0, i, j] = not moving_part[j] and move_seg[round(tracks[0, i, j, 1].item()), round(tracks[0, i, j, 0].item())]
    return occulasions

def find_closest_point(uv, coords):
    dists = np.sum((uv[:, None] - coords[None]) ** 2, axis=-1)
    return np.argmin(dists, axis=0)

def find_point_with_minimum_depth(uv, depth, coords, pixel_range=4):

    selected_indices_with_min_depth = []

    for coord in coords:
        # Calculate the distance between the current coord and all points in uv
        distances = np.linalg.norm(uv - coord, axis=1)
        
        # Select indices of points in uv where the distance is less than or equal to pixel_range
        in_range_indices = np.where(distances <= pixel_range)[0]
        
        if len(in_range_indices) > 0:
            # Find the index of the point with the minimum depth in the selected points
            min_depth_index_within_range = np.argmin(depth[in_range_indices])
            
            # Get the corresponding index in the original uv array
            selected_index = in_range_indices[min_depth_index_within_range]
        else:
            # If no points are within the range, find the closest point
            closest_index = np.argmin(distances)
            selected_index = closest_index
        
        # Store the index of the selected point
        selected_indices_with_min_depth.append(selected_index)
    
    return selected_indices_with_min_depth


def extract_first_visible_points(points, occluded):
    first_visible_indices = np.argmax(~occluded, axis=1)
    first_visible_points = points[np.arange(points.shape[0]), first_visible_indices, -1::-1]
    result = np.column_stack((first_visible_indices, first_visible_points))
    return result

def check_coord_within(coord, W, H, eps=0.5):
    return coord[0] >= 0 + eps and coord[0] < H - 1 - eps and coord[1] >= 0 + eps and coord[1] < W - 1 - eps

def compare_color(image, uv, color, opacity, threshold=0.05):
    # Find the cloest point in image by using the uv
    uv = uv.astype(int)
    real_color = color.detach().cpu().numpy() / (1+ math.exp(-float(opacity.detach().cpu().numpy()[0])))
    if not check_coord_within(uv, image.shape[0], image.shape[1]):
        return True
    # Get the color of the cloest point
    cloest_color = image[uv[1], uv[0]]/255.
    # Compare the color
    diff = np.sum((cloest_color - real_color) ** 2)
    return diff > threshold