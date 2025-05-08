import torch
from PIL import Image
from torchvision import transforms
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

    return img_tensor

def process_frames_to_video(frames_sequence):
    frames_video_torch = torch.from_numpy(np.stack(frames_sequence))
    frames_video_torch = frames_video_torch.permute(0,3,1,2)
    frames_video_torch = frames_video_torch[None, :].float()
    return frames_video_torch