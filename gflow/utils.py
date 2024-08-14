import os
import json
import time
import torch
import torchsnooper
from PIL import Image
import torchvision.transforms as transforms
from typing import Literal
import matplotlib
import numpy as np
import cv2
from scipy.spatial import ConvexHull
# from alphashape import alphashape
from skimage import draw
from matplotlib import pyplot as plt
import concavity
from concavity.utils import *
from concave_hull import concave_hull
import imageio
from matplotlib import cm
from tqdm import tqdm
import pointops

def signed_expm1(x):
    sign = torch.sign(x)
    return sign * torch.expm1(torch.abs(x))

def signed_log1p(x):
    sign = torch.sign(x)
    return sign * torch.log1p(torch.abs(x))

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

def image_path_to_tensor(image_path, resize: int = None, blur: bool = False, blur_sigma: float = 5.0, blur_kernel_size: int = 7):
    img = Image.open(image_path) # the range is [0,1] by Image.open
    trans_list = [
        transforms.ToTensor(),
    ]
    if resize is not None:
        trans_list.append(transforms.Resize(resize, antialias=True))
    if blur:
        trans_list.append(transforms.GaussianBlur(blur_kernel_size, blur_sigma))
    transform = transforms.Compose(trans_list)
    img_tensor = transform(img) # (C, H, W) and C is 3
    img_tensor = img_tensor.permute(1, 2, 0)[..., :3] # (H, W, C) and C is 3
    # img_tensor = resize_by_divide(img_tensor, 16)

    return img_tensor

def read_mask(mask_path, resize=None, device="cpu"):
    # print(resize)
    # print(mask_path)
    mask = imageio.imread(mask_path)
    if mask.ndim == 3:
        mask_tensor = torch.tensor(mask, dtype=torch.float32).permute(2, 0, 1)
    elif mask.ndim == 2:
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
    else:
        raise ValueError("The mask should be 2D or 3D")
    # print(mask_tensor.shape)
    # print(mask_tensor.min(), mask_tensor.max())
    # resize the mask
    if resize is not None:
        transform = transforms.Resize(resize, antialias=True)
        mask_tensor = transform(mask_tensor)
    # if mask_tensor has rgb channels, turn it to binary
    if mask_tensor.shape[-1] > 1:
        mask_tensor = mask_tensor.sum(dim=0)
    mask_tensor = mask_tensor.squeeze()
    # convert to bool type
    mask_tensor = mask_tensor > 0
    # save the mask as a binary image
    # mask_np = mask_tensor.numpy() * 255
    # mask_np = mask_np.astype(np.uint8)
    # imageio.imwrite("/home/wangshizun/projects/msplat/mask.png", mask_np)

    return mask_tensor


def read_depth(depth_path, resize=None, depth_scale=1.0, depth_offset=0.):
    depth = np.load(depth_path)
    depth_tensor = torch.tensor(depth, dtype=torch.float32)
    # print("depth_tensor.shape", depth_tensor.shape)
    depth_tensor = depth_tensor.unsqueeze(0)
    # resize the depth
    if resize is not None:
        transform = transforms.Resize(resize, antialias=True)
        depth_tensor = transform(depth_tensor)
    # print("depth_tensor.shape after", depth_tensor.shape)
    # print("depth_tensor.min()", depth_tensor.min())
    # print("depth_tensor.max()", depth_tensor.max())
    depth_tensor = depth_tensor.squeeze(0)
    return depth_tensor * depth_scale + depth_offset

def read_camera(camera_paths):
    """
    read focal, principle points, and camera pose
    """
    focal_list = []
    pose_list = []
    pp = None
    for camera_path in camera_paths:
        with open(camera_path, "r") as f:
            camera_dict = json.load(f)
        focal_list.append(camera_dict["focal"])
        pose_list.append(camera_dict["pose"][:3])
        pp = [round(camera_dict["pp"][0]), round(camera_dict["pp"][1])]
    
    focal = np.array(focal_list).mean()
    # focal = np.array(focal_list[0]).mean()
    # print(type(focal))
    # print(focal.shape)
    # print(focal)
    pose_list = np.array(pose_list)
    # print(pose_list.shape)
    # print(pose_list[0])
    # print(focal)
    # print(pp)

    return focal.item(), pp, pose_list

@torchsnooper.snoop()
def camera2world(uvd, intr, extr):
    '''
    uvd: [N, 3]
    intr: [3, 3]
    extr: [3, 4]
    convert camera coordinate to world coordinate according to a pinhole camera model

    return: world coordinate [N, 3]
    '''

    # cx, cy = intr[0, 2], intr[1, 2]
    # t3 = extr[2, 3]
    # uvd[:,0] = uvd[:,0] + cx * t3
    # uvd[:,1] = uvd[:,1] + cy * t3

    # uvd[:,0] = uvd[:,0] * uvd[:,2]
    # uvd[:,1] = uvd[:,1] * uvd[:,2]
    # import pdb
    # pdb.set_trace()
    # (3, N)
    camera_coord = torch.t(uvd)
    _camera_coord = torch.cat((camera_coord, torch.ones(1, camera_coord.shape[1]).to(camera_coord.device)), dim=0)


    _intr = torch.zeros(4, 4).to(intr.device)
    _intr[:3, :3] = intr
    _intr[-1, -1] = 1

    _extr = torch.zeros(4, 4).to(extr.device)
    _extr[:3, :] = extr
    _extr[-1, -1] = 1

    combined = torch.inverse(_intr @ _extr)


    world_coord = combined @ _camera_coord
    world_coord = world_coord[:3, :].t()
    # pdb.set_trace()
    return world_coord


# def camera2world(uvd, intr, extr):
#     '''
#     uvd: [N, 3]
#     intr: [3, 3]
#     extr: [3, 4]
#     convert camera coordinate to world coordinate according to a pinhole camera model

#     return: world coordinate [N, 3]
#     '''

#     cx, cy = intr[0, 2], intr[1, 2]
#     t3 = extr[2, 3]
#     uvd[:,0] = uvd[:,0] + cx * t3
#     uvd[:,1] = uvd[:,1] + cy * t3

#     uvd[:,0] = uvd[:,0] * uvd[:,2]
#     uvd[:,1] = uvd[:,1] * uvd[:,2]
#     # import pdb
#     # pdb.set_trace()
#     # (3, N)
#     camera_coord = torch.t(uvd)

#     _intr = torch.zeros(4, 4).to(intr.device)
#     _intr[:3, :3] = intr
#     _intr[-1, -1] = 1

#     _extr = torch.zeros(4, 4).to(extr.device)
#     _extr[:3, :] = extr
#     _extr[-1, -1] = 1

#     combined = torch.inverse(_intr @ _extr)

#     _camera_coord = torch.cat((camera_coord, torch.ones(1, camera_coord.shape[1]).to(camera_coord.device)), dim=0)

#     world_coord = combined @ _camera_coord
#     world_coord = world_coord[:3, :].t()
#     # pdb.set_trace()
#     return world_coord


def apply_float_colormap(image, colormap: Literal["turbo", "grey", "gist_rainbow"] = "turbo", non_zero: bool = False):
    # [ ] TODO replace to a simper one: https://github.com/NVlabs/CF-3DGS/blob/79f938596c1a32d54929c718a06450142323816d/trainer/trainer.py#L173
    # colormap = "turbo"
    # image = image[..., None]
    if non_zero:
        image = image - torch.min(image[image != 0])
    else:
        image = image - torch.min(image)
    image = image / (torch.max(image) + 1e-5)
    image = torch.clip(image, 0, 1)
    image = torch.nan_to_num(image, 0)
    # print(image.shape)
    if colormap == "grey":
        # return image.repeat(1, 1, 3)
        image = image.expand(*image.shape[:-1], 3).contiguous()
        # print(image.shape)
        # exit()
        return image
    image_long = (image * 255).long()
    image_long_min = torch.min(image_long)
    image_long_max = torch.max(image_long)
    assert image_long_min >= 0, f"the min value is {image_long_min}"
    assert image_long_max <= 255, f"the max value is {image_long_max}"

    result_np = cm.get_cmap(colormap)(image_long[..., 0].cpu().numpy())[:, :3]
    result_tensor = torch.tensor(result_np, device=image.device).float()

    return result_tensor
    # return torch.tensor(matplotlib.colormaps[colormap].colors, device=image.device)[
    #     image_long[..., 0]
    # ]

def gen_line_set(xyz1, xyz2, rgb, device):
    # xyz1 = torch.randn((N, 3))
    # xyz2 = torch.randn((N, 3))
    N = xyz1.shape[0]
    # print("xyz1.shape[0]", xyz1.shape[0])
    # print("xyz2.shape[0]", xyz2.shape[0])

    # 计算两个张量之间的差值
    diff = xyz2 - xyz1
    diff_norm = torch.norm(diff, dim=1, keepdim=True)

    line_set_xyz = []
    line_set_rgb = []
    line_set_scale = []
    point_set_xyz = []
    point_set_rgb = []
    point_set_scale = []

    for i in range(N):
        L = max(2, int(diff_norm[i] * 100))
        # print(L)
        for j in range(L):
            t = j / (L-1)  
            increment = xyz1[i] + t * diff[i]
            if j < L-1:
                line_set_xyz.append(increment)
                line_set_rgb.append(rgb[i])
                line_set_scale.append(torch.ones(3, device=device))
            if j == (L-1):
                point_set_xyz.append(increment)
                point_set_rgb.append(rgb[i])
                point_set_scale.append(torch.ones(3, device=device))

    # last N are points
    line_set_xyz = torch.stack(line_set_xyz+point_set_xyz).to(device)
    line_set_rgb = torch.stack(line_set_rgb+point_set_rgb).to(device)
    line_set_scale = torch.stack(line_set_scale+point_set_scale).to(device)
    point_set_xyz = torch.stack(point_set_xyz).to(device)
    point_set_rgb = torch.stack(point_set_rgb).to(device)
    point_set_scale = torch.stack(point_set_scale).to(device)

    return line_set_xyz, line_set_rgb

def save_splat_file(splat_data, output_path):
    with open(output_path, "wb") as f:
        f.write(splat_data)
    print("Data written to {}".format(output_path))

def extract_camera_parameters(intrinsic_matrix, extrinsic_matrix, W, H, img_name="00001"):
    # Extract focal lengths and principal point from the intrinsic matrix
    [fx, fy, cx, cy] = intrinsic_matrix.detach().cpu().numpy().tolist()
    '''
    # Extract rotation matrix and translation vector from the extrinsic matrix
    extr = extrinsic_matrix.T.detach().cpu().numpy()
    extrinsic_matrix_inv = np.linalg.pinv(extr)
    # print(extr)
    # print(extrinsic_matrix_inv)
    R = extrinsic_matrix_inv[:, :3]
    R = np.linalg.inv(R)

    # print(R)
    t = extrinsic_matrix_inv[:, 3]

    # Calculate camera position in world coordinates
    camera_position = R.dot(t)
    '''

    R = extrinsic_matrix[:3, :3]  # 旋转矩阵部分
    t = extrinsic_matrix[:3, 3]   # 平移向量部分

    # 计算相机在世界坐标系中的位置
    # 通过求逆矩阵的方法，位置为 -R.T @ t
    camera_position = -R.T @ t

    # 计算相机在世界坐标系中的旋转
    # 旋转矩阵在世界坐标系中为 R.T
    camera_rotation = R.T

    # Return all extracted parameters
    return [{
        "id": 0,
        "img_name": img_name,
        "width": W,
        "height": H,
        "position": camera_position.tolist(),
        "rotation": camera_rotation.tolist(),
        "fx": fx,
        "fy": fy
    }]

def construct_list_of_attributes():
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(3):
        l.append('f_dc_{}'.format(i))
    l.append('opacity')
    for i in range(3):
        l.append('scale_{}'.format(i))
    for i in range(4):
        l.append('rot_{}'.format(i))
    return l

class ConcaveHull2D:
    def __init__(self, points, knn=10, sigma=0., num_points_factor=2):
        """
        Initialize the AlphaShape class

        Args:
        points: numpy array, representing the point set, each row is the coordinate of a point, shape is (N, 2)
        alpha: alpha parameter, controls the shape of the concave hull, the smaller the value, the smoother the boundary
        """
        self.points = points
        self.knn = knn

        # judge if the points are in numpy format, if not, convert them to numpy
        if isinstance(points, torch.Tensor):
            self.points = points.detach().cpu().numpy()
        
        # calculate concave hull
        self.hull = concavity.concave_hull(self.points, self.knn)

        if sigma > 0:
            self.hull = concavity.gaussian_smooth_geom(self.hull, sigma=sigma, num_points_factor=num_points_factor)
    
    def area(self):
        """
        return the area of the concave hull
        """
        return self.hull.area
    
    def mask(self, width, height):
        """
        Return the mask of the concave hull
        
        Args:
        width: int, the width of the mask
        height: int, the height of the mask
        
        Returns:
        mask: numpy array, representing the segmentation mask of the concave hull, shape is (height, width), where True indicates inside the concave hull and False indicates outside
        """
        mask = polygon_to_mask(self.hull, width, height)

        return mask

def filter_sparse_mask(points_tensor, eps=1.5, min_samples=10):
    points_np = points_tensor.detach().cpu().numpy()
    # using sklearn DBSCAN to filter the sparse points
    from sklearn.cluster import DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points_np)
    sparse_mask_np = clustering.labels_ == -1
    # convert to tensor, bool type
    sparse_mask_tensor = torch.tensor(sparse_mask_np, device=points_tensor.device)

    return sparse_mask_tensor


class FastConcaveHull2D:
    def __init__(self, points, sigma=2, num_points_factor=5):
        self.points = points

        # judge if the points are in numpy format, if not, convert them to numpy
        if isinstance(points, torch.Tensor):
            self.points = points.detach().cpu().numpy()
        
        # calculate the convex hull
        points = concave_hull(self.points)
        self.hull = Polygon(points)
        if sigma > 0:
            self.hull = concavity.gaussian_smooth_geom(self.hull, sigma=sigma, num_points_factor=num_points_factor)
    
    def area(self):
        return self.hull.area

    def mask(self, width, height):
        mask = polygon_to_mask(self.hull, width, height)

        return mask

class ConvexHull2D:
    def __init__(self, points):
        self.points = points

       # judge if the points are in numpy format, if not, convert them to numpy
        if isinstance(points, torch.Tensor):
            self.points = points.detach().cpu().numpy()
        
        # calculate the convex hull
        self.hull = ConvexHull(self.points)
    
    def area(self):
        # return the area of the convex hull
        return self.hull.volume
    
    def mask(self, width, height):
        """ 
        input: width: int, the width of the mask
        return: mask: numpy array, representing the segmentation mask of the convex hull, shape is (H, W), where True indicates inside the convex hull and False indicates outside
        """
        # create a mask with the same shape as the image
        mask = np.zeros((width, height), dtype=np.uint8)

        # convert the points to integer
        points = np.round(self.hull.points[self.hull.vertices]).astype(int)

        # fill the convex hull with white color
        cv2.fillPoly(mask, [points], color=(1,))

        return mask

def and_with_most_common(set0, set1, set2):
    """
    set0, set1, set2 are torch tensor of [N, 2]
    compare set1 and set2, which got the most common components with set0
    then return the "and" result between set0 and the most common set
    """

    # convert to numpy
    set0 = set0.detach().cpu().numpy()
    set1 = set1.detach().cpu().numpy()
    set2 = set2.detach().cpu().numpy()

    # convert to set
    set0 = set([tuple(x) for x in set0])
    set1 = set([tuple(x) for x in set1])
    set2 = set([tuple(x) for x in set2])

    # find the most common set
    common_set = set1 if len(set0 & set1) > len(set0 & set2) else set2

    # convert to torch tensor
    common_set = torch.tensor(list(common_set), device=set0.device)

    # find the "and" result
    and_set = set0 & common_set

    return torch.tensor(list(and_set), device=set0.device)

def process_frames_to_video(frames_sequence):
    frames_video_torch = torch.from_numpy(np.stack(frames_sequence))
    frames_video_torch = frames_video_torch.permute(0,3,1,2)
    frames_video_torch = frames_video_torch[None, :].float()
    return frames_video_torch

def process_traj_to_tracks(sequence_traj):
    tracks_traj = torch.from_numpy(np.stack(sequence_traj))
    tracks_traj = tracks_traj[None, :].float()
    return tracks_traj

def process_segm_mask(move_seg):
    sege_mask = torch.from_numpy(move_seg).float()
    sege_mask = sege_mask[None, None, :].float()
    return sege_mask

def process_color_occu(sequence_traj_occlusion):
    occulasions = torch.from_numpy(np.stack(sequence_traj_occlusion))
    occulasions = occulasions[None, :].float()
    return occulasions

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

# def farthest_point_sampling(xyz, fps_number, device):
#     # farthest point sampling
#     N = xyz.shape[0]
#     fps_idx = [0]
#     for i in tqdm(range(1, fps_number)):
#         dists = torch.sum((xyz - xyz[fps_idx[-1]]) ** 2, dim=1)
#         idx = torch.argmax(dists)
#         fps_idx.append(idx)
#     return fps_idx


def farthest_point_sampling(xyz, fps_number, device):
    num_pts = xyz.shape[0]
    print(num_pts)
    start = time.time()
    batch_size = 1
    offset = torch.cumsum(torch.Tensor([num_pts] * batch_size).to(device), dim=0).to(torch.long)
    new_offset = torch.cumsum(torch.Tensor([fps_number] * batch_size).to(device), dim=0).to(torch.long)

    result = pointops.farthest_point_sampling(xyz, offset, new_offset)
    print(result.shape)
    end = time.time()
    print("FPS time:", end-start)

    return result