import torch
import torchsnooper
from PIL import Image
import torchvision.transforms as transforms
from typing import Literal
import matplotlib


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

def gen_line_set(xyz1, xyz2, rgbs, traj_point_scale, H, W, device):
    # xyz1 = torch.randn((N, 3))
    # xyz2 = torch.randn((N, 3))
    N = xyz1.shape[0]
    # print("xyz1.shape[0]", xyz1.shape[0])
    # print("xyz2.shape[0]", xyz2.shape[0])

    # 计算两个张量之间的差值
    diff = xyz2 - xyz1
    diff_norm = torch.norm(diff, dim=1, keepdim=True)

    line_set_means = []
    line_set_rgbs = []
    line_set_scales = []

    for i in range(N):
        L = max(2, int(diff_norm[i] * 1000))
        # print(L)
        for j in range(L):
            t = j / (L-1)  
            increment = xyz1[i] + t * diff[i]  
            line_set_means.append(increment)
            line_set_rgbs.append(rgbs[i])
            line_set_scales.append(torch.ones(3, device=device) * (traj_point_scale/(H+W)))
        line_set_scales[-1] = line_set_scales[-1] * 7.

    line_set_means = torch.stack(line_set_means).to(device)
    line_set_rgbs = torch.stack(line_set_rgbs).to(device)
    line_set_scales = torch.stack(line_set_scales).to(device)

    return line_set_means, line_set_rgbs, line_set_scales
