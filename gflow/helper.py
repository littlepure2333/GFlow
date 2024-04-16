import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
import numpy as np
from os.path import *
import torchvision.transforms as transforms


TAG_CHAR = np.array([202021.25], np.float32)



def calc_dist(tensor1, tensor2, chunk_size=5000):
    distances = []
    for i in range(0, tensor1.size(0), chunk_size):
        dist_i = []
        for j in range(0, tensor2.size(0), chunk_size):
            tensor1_chunk = tensor1[i:i+chunk_size]
            tensor2_chunk = tensor2[j:j+chunk_size]
            dist_i.append(torch.cdist(tensor1_chunk, tensor2_chunk).cpu())
        dist_i = torch.cat(dist_i, dim=1)
        distances.append(dist_i)
    # the shape of return tensor is (tensor1.size(0), tensor2.size(0))
    return torch.cat(distances)

def hungarian_match(output_dict, target_dict):
    # create an empty tensor to store the distances
    shape1 = output_dict['means'].shape[0]
    shape2 = target_dict['means'].shape[0]
    distances = torch.zeros((shape1, shape2))
    print(distances.shape)
    for output_key, target_key in zip(output_dict.keys(), target_dict.keys()):
        # print(output_key)
        # print(output_dict[output_key].shape)
        # print(target_dict[target_key].shape)
        if output_key != target_key:
            raise ValueError("output_key and target_key must be the same")
        if output_key == 'means':
            distances += calc_dist(output_dict[output_key], target_dict[target_key])
            # check numerical error
            if torch.isnan(distances).any():
                raise ValueError("means distances contains nan")
        if output_key == 'rgbs':
            distances += 10 * calc_dist(torch.sigmoid(output_dict[output_key]), torch.sigmoid(target_dict[target_key]))
            # check numerical error
            if torch.isnan(distances).any():
                raise ValueError("rgbs distances contains nan")
        if output_key == 'scales':
            distances += 0.05 * calc_dist(torch.exp(output_dict[output_key]), torch.exp(target_dict[target_key]))
            # check numerical error
            if torch.isnan(distances).any():
                raise ValueError("scales distances contains nan")
        if output_key == 'opacities':
            distances += 0.001 * calc_dist(output_dict[output_key], target_dict[target_key])
            # check numerical error
            if torch.isnan(distances).any():
                raise ValueError("opacities distances contains nan")

    cost_matrix = distances.data.cpu().numpy()
    # check numerical error
    if np.isnan(cost_matrix).any():
        raise ValueError("cost_matrix contains nan")
    
    print("linear sum assigning")
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    print("finish linear sum assignment")

    # the matched points
    # matched_points_frame1 = frame1_points["means"][row_ind]
    # matched_points_frame2 = frame2_points["means"][col_ind]

    # print("the index of matched points in the first frame:", row_ind)
    # print("the index of matched points in the second frame:", col_ind)
    # print("the information of matched points in the first frame:", matched_points_frame1)
    # print("the information of matched points in the second frame:", matched_points_frame2)

    return row_ind, col_ind


def readFlow(fn, resize: int = None, blur: bool = False, blur_sigma: float = 5.0, blur_kernel_size: int = 7):
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


def writeFlow(filename, uv, v=None):
    """ Write optical flow to file.

    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert (uv.ndim == 3)
        assert (uv.shape[2] == 2)
        u = uv[:, :, 0]
        v = uv[:, :, 1]
    else:
        u = uv

    assert (u.shape == v.shape)
    height, width = u.shape
    f = open(filename, 'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width * nBands))
    tmp[:, np.arange(width) * 2] = u
    tmp[:, np.arange(width) * 2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()


if __name__ == "__main__":
    # 生成示例数据
    torch.manual_seed(42)
    np.random.seed(42)

    # 第一帧的点云数据
    means = 10*torch.randn((5, 3))
    frame1_points = {"means": means}

    # 在第二帧中随机挪动一点点，点的数量可能不同
    means_new = means.clone()[[4,2,1,3,0]][:4]  # 为了模拟点的数量不一致，只使用前4个点
    means_new[:, :3] += 0.1 * torch.randn((4, 3))
    frame2_points = {"means": means_new}

    # 匈牙利匹配
    hungarian_match(frame1_points, frame2_points)