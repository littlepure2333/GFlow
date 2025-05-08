# Copyright (c) Meta Platforms, Inc. and affiliates.
'''
Adapted from https://github.com/facebookresearch/robust-dynrf/blob/main/scripts/generate_mask.py
'''

import argparse
import glob
import os

import cv2
import numpy as np
import skimage.morphology
import torch
from PIL import Image
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)[..., :3]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def get_uv_grid(H, W, homo=False, align_corners=False, device=None):
    """
    Get uv grid renormalized from -1 to 1
    :returns (H, W, 2) tensor
    """
    if device is None:
        device = torch.device("cpu")
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing="ij",
    )
    if align_corners:
        xx = 2 * xx / (W - 1) - 1
        yy = 2 * yy / (H - 1) - 1
    else:
        xx = 2 * (xx + 0.5) / W - 1
        yy = 2 * (yy + 0.5) / H - 1
    if homo:
        return torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)
    return torch.stack([xx, yy], dim=-1)


def compute_sampson_error(x1, x2, F):
    """
    :param x1 (*, N, 2)
    :param x2 (*, N, 2)
    :param F (*, 3, 3)
    """
    h1 = torch.cat([x1, torch.ones_like(x1[..., :1])], dim=-1)
    h2 = torch.cat([x2, torch.ones_like(x2[..., :1])], dim=-1)
    d1 = torch.matmul(h1, F.transpose(-1, -2))  # (B, N, 3)
    d2 = torch.matmul(h2, F)  # (B, N, 3)
    z = (h2 * d1).sum(dim=-1)  # (B, N)
    err = z**2 / (
        d1[..., 0] ** 2 + d1[..., 1] ** 2 + d2[..., 0] ** 2 + d2[..., 1] ** 2
    )
    return err

def run_maskrcnn(model, img_dir):
    threshold = 0.5
    o_image = Image.open(img_dir).convert("RGB")
    width, height = o_image.size
    if width > height:
        intHeight = 576
        intWidth = 1024
    else:
        intHeight = 1024
        intWidth = 576

    image = o_image.resize((intWidth, intHeight), Image.ANTIALIAS)

    image_tensor = torchvision.transforms.functional.to_tensor(image).cuda()

    tenHumans = torch.FloatTensor(intHeight, intWidth).fill_(1.0).cuda()

    objPredictions = model([image_tensor])[0]

    for intMask in range(objPredictions["masks"].size(0)):
        if objPredictions["scores"][intMask].item() > threshold:
            # person, vehicle, accessory, animal, sports
            if objPredictions["labels"][intMask].item() == 1:  # person
                tenHumans[objPredictions["masks"][intMask, 0, :, :] > threshold] = 0.0
            if (
                objPredictions["labels"][intMask].item() >= 2
                and objPredictions["labels"][intMask].item() <= 9
            ):  # vehicle
                tenHumans[objPredictions["masks"][intMask, 0, :, :] > threshold] = 0.0
            if (
                objPredictions["labels"][intMask].item() >= 26
                and objPredictions["labels"][intMask].item() <= 33
            ):  # accessory
                tenHumans[objPredictions["masks"][intMask, 0, :, :] > threshold] = 0.0
            if (
                objPredictions["labels"][intMask].item() >= 16
                and objPredictions["labels"][intMask].item() <= 25
            ):  # animal
                tenHumans[objPredictions["masks"][intMask, 0, :, :] > threshold] = 0.0
            if (
                objPredictions["labels"][intMask].item() >= 34
                and objPredictions["labels"][intMask].item() <= 43
            ):  # sports
                tenHumans[objPredictions["masks"][intMask, 0, :, :] > threshold] = 0.0
            if objPredictions["labels"][intMask].item() == 88:  # teddy bear
                tenHumans[objPredictions["masks"][intMask, 0, :, :] > threshold] = 0.0

    npyMask = skimage.morphology.erosion(
        tenHumans.cpu().numpy(), skimage.morphology.disk(1)
    )
    npyMask = ((npyMask < 1e-3) * 255.0).clip(0.0, 255.0).astype(np.uint8)
    return npyMask

def readFlow(fn, resize: int = None, blur: bool = False, blur_sigma: float = 5.0, blur_kernel_size: int = 7, device=None):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
            # Reshape testdata into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            flow = np.resize(data, (int(h), int(w), 2))

            trans_list = []
            flow = torch.from_numpy(flow).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
            if resize is not None:
                trans_list.append(transforms.Resize(resize, antialias=True))
            if blur:
                trans_list.append(transforms.GaussianBlur(blur_kernel_size, blur_sigma))
            transform = transforms.Compose(trans_list)
            flow_tensor = transform(flow).permute(1, 2, 0) # (C, H, W) -> (H, W, C)
            if device is None:
                device = torch.device("cpu")
            flow_tensor = flow_tensor.to(device)
            return flow_tensor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, help="images folder path")
    parser.add_argument("--threshold", type=float, default=0.01, help="epipolar error threshold for motion mask")
    args = parser.parse_args()
    img_dir = args.img_dir
    # flow_dir = img_dir + "_flow_refine"
    flow_dir = img_dir + "_flow_unimatch"
    epipolar_dir = img_dir + "_epipolar"
    move_mask_dir = img_dir + "_move_mask"

    image_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg"))) + sorted(glob.glob(os.path.join(img_dir, "*.png")))
    fwd_flow_paths = sorted(glob.glob(os.path.join(flow_dir, "*_pred.flo")))
    bwd_flow_paths = sorted(glob.glob(os.path.join(flow_dir, "*_pred_bwd.flo")))

    assert len(image_paths) == len(fwd_flow_paths) + 1, f"len(image_paths): {len(image_paths)}, len(fwd_flow_paths): {len(fwd_flow_paths)}"
    assert len(fwd_flow_paths) == len(bwd_flow_paths), f"len(fwd_flow_paths): {len(fwd_flow_paths)}, len(bwd_flow_paths): {len(bwd_flow_paths)}"

    img = load_image(image_paths[0])
    H = img.shape[2]
    W = img.shape[3]


    # RUN EPIPOLAR ERROR
    uv = get_uv_grid(H, W, align_corners=False)
    x1 = uv.reshape(-1, 2)
    frames_epipolar_error = []
    frames_mask = []    
    for idx, (fwd_flow_path, bwd_flow_path) in tqdm(enumerate(zip(fwd_flow_paths, bwd_flow_paths))):
        # if idx > 10:
        #     break
        err_list = []
        # for flow_path in [bwd_flow_path]: # Only calculate on backward flow.
        flow = readFlow(fwd_flow_path)
        flow = torch.stack(
            [
                2.0 * flow[..., 0] / (W - 1),
                2.0 * flow[..., 1] / (H - 1),
            ],
            dim=-1
        )

        x2 = x1 + flow.view(-1, 2)  # (H*W, 2)
        F, mask = cv2.findFundamentalMat(x1.numpy(), x2.numpy(), cv2.FM_LMEDS)
        F = torch.from_numpy(F.astype(np.float32))  # (3, 3)
        err = compute_sampson_error(x1, x2, F).reshape(H, W)
        fac = (H + W) / 2
        err = err * fac**2
        err_list.append(err)
        
        err = torch.amax(torch.stack(err_list, 0), 0) # (H, W) per pixel max error from both directions
        err = ((err / err.max())).numpy() # normalize to (0, 255)

        # move mask
        mask = err > args.threshold

        # open mask
        mask_open = skimage.morphology.binary_opening(
            mask, skimage.morphology.disk(2)
        )

        # erode mask
        mask_erode = skimage.morphology.binary_erosion(
            mask, skimage.morphology.disk(5)
        )

        # dilate mask
        mask_dilate = skimage.morphology.binary_dilation(
            mask, skimage.morphology.disk(3)
        )


        # file management
        if not os.path.exists(os.path.join(epipolar_dir)):
            os.makedirs(os.path.join(epipolar_dir))
        file_name = os.path.splitext(os.path.basename(image_paths[idx]))[0]

        # if idx == len(fwd_flow_paths) - 1 and i == 1:
        #     file_name = os.path.splitext(os.path.basename(image_paths[idx+1]))[0]
        
        # save epipolar error
        Image.fromarray((err * 255.0).astype(np.uint8)).save(
            os.path.join(epipolar_dir, file_name + "_epipolar_error.png")
        )

        # save opening mask
        Image.fromarray((mask_open * 255.0).astype(np.uint8)).save(
            os.path.join(epipolar_dir, file_name + "_open.png")
        )

        # save erode mask
        Image.fromarray((mask_erode * 255.0).astype(np.uint8)).save(
            os.path.join(epipolar_dir, file_name + "_erode.png")
        )

        # save dilate mask
        Image.fromarray((mask_dilate * 255.0).astype(np.uint8)).save(
            os.path.join(epipolar_dir, file_name + "_dilate.png")
        )

        frames_epipolar_error.append(err)
        frames_mask.append(mask)