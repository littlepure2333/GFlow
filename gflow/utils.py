import torch
import torchsnooper
from PIL import Image
import torchvision.transforms as transforms
from typing import Literal
import matplotlib
import numpy as np


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

def extract_camera_parameters(intrinsic_matrix, extrinsic_matrix):
    # Extract focal lengths and principal point from the intrinsic matrix
    [fx, fy, cx, cy] = intrinsic_matrix.detach().cpu().numpy().tolist()

    # Extract rotation matrix and translation vector from the extrinsic matrix
    R = extrinsic_matrix[:, :3]
    print(R)
    t = extrinsic_matrix[:, 3]

    # Calculate camera position in world coordinates
    camera_position = -np.linalg.inv(R.detach().cpu().numpy()).dot(t.detach().cpu().numpy())

    # Return all extracted parameters
    return [{
        "id": 0,
        "img_name": "00001",
        "width": 1959,
        "height": 1090,
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