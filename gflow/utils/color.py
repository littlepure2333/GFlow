from matplotlib import cm
import torch
from typing import Literal


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

def apply_float_colormap(image, colormap: Literal["turbo", "grey", "gist_rainbow"] = "turbo", non_zero: bool = False):
    if non_zero:
        image = image - torch.min(image[image != 0])
    else:
        image = image - torch.min(image)
    image = image / (torch.max(image) + 1e-5)
    image = torch.clip(image, 0, 1)
    image = torch.nan_to_num(image, 0)
    if colormap == "grey":
        image = image.expand(*image.shape[:-1], 3).contiguous()
        return image
    image_long = (image * 255).long()
    image_long_min = torch.min(image_long)
    image_long_max = torch.max(image_long)
    assert image_long_min >= 0, f"the min value is {image_long_min}"
    assert image_long_max <= 255, f"the max value is {image_long_max}"

    result_np = cm.get_cmap(colormap)(image_long[..., 0].cpu().numpy())[:, :3]
    result_tensor = torch.tensor(result_np, device=image.device).float()

    return result_tensor