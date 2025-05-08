import imageio
import torch
import numpy as np
import torchvision.transforms as transforms
import json

def read_flow(fn, resize: int = None, blur: bool = False, blur_sigma: float = 5.0, blur_kernel_size: int = 7):
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

            return flow_tensor


def read_mask(mask_path, resize=None, device="cpu"):
    mask = imageio.imread(mask_path)
    if mask.ndim == 3:
        mask_tensor = torch.tensor(mask, dtype=torch.float32).permute(2, 0, 1)
    elif mask.ndim == 2:
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
    else:
        raise ValueError("The mask should be 2D or 3D")
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

    return mask_tensor


def read_depth(depth_path, resize=None, depth_scale=1.0, depth_offset=0.):
    depth = np.load(depth_path)
    depth_tensor = torch.tensor(depth, dtype=torch.float32)
    depth_tensor = depth_tensor.unsqueeze(0)
    # resize the depth
    if resize is not None:
        transform = transforms.Resize(resize, antialias=True)
        depth_tensor = transform(depth_tensor)
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
    pose_list = np.array(pose_list)
    return focal.item(), pp, pose_list

## Sintel IO
TAG_FLOAT = 202021.25
def cam_read(filename):
    """ Read camera data, return (M,N) tuple.
    
    M is the intrinsic matrix, N is the extrinsic matrix, so that

    x = M*N*X,
    with x being a point in homogeneous image pixel coordinates, X being a
    point in homogeneous world coordinates.
    """
    f = open(filename,'rb')
    # print(filename)
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, ' cam_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    M = np.fromfile(f,dtype='float64',count=9).reshape((3,3))
    N = np.fromfile(f,dtype='float64',count=12).reshape((3,4))
    return M,N