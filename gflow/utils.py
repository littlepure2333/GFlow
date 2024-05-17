import os
import json
import torch
import torchsnooper
from PIL import Image
import torchvision.transforms as transforms
from typing import Literal
import matplotlib
import numpy as np
import cv2
from scipy.spatial import ConvexHull
from alphashape import alphashape
from skimage import draw
from descartes import PolygonPatch
from matplotlib import pyplot as plt
import concavity
from concavity.utils import *
from concave_hull import concave_hull

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

def read_depth(depth_path, resize=None, depth_scale=1.0, depth_offset=0.):
    depth = np.load(depth_path)
    depth_tensor = torch.tensor(depth, dtype=torch.float32)
    # resize the depth
    if resize is not None:
        transform = transforms.Resize(resize, antialias=True)
        depth_tensor = transform(depth_tensor)
    # print("depth_tensor.shape", depth_tensor.shape)
    # print("depth_tensor.min()", depth_tensor.min())
    # print("depth_tensor.max()", depth_tensor.max())
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


def apply_float_colormap(image, colormap: Literal["turbo", "grey"] = "turbo", non_zero: bool = False):
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

    return torch.tensor(matplotlib.colormaps[colormap].colors, device=image.device)[
        image_long[..., 0]
    ]

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

    # Extract rotation matrix and translation vector from the extrinsic matrix
    R = extrinsic_matrix[:, :3]
    R = np.linalg.inv(R.detach().cpu().numpy())

    # print(R)
    t = extrinsic_matrix[:, 3]

    # Calculate camera position in world coordinates
    camera_position = R.dot(t.detach().cpu().numpy())

    # Return all extracted parameters
    return [{
        "id": 0,
        "img_name": img_name,
        "width": W,
        "height": H,
        "position": camera_position.tolist(),
        "rotation": R.tolist(),
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


class FastConcaveHull2D:
    def __init__(self, points, sigma=0., num_points_factor=2):
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