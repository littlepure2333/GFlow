import os
import glob
from tqdm import tqdm
import cv2
import torch
import imageio
import numpy as np
from PIL import Image
# from transformers import pipeline
# add the path to the depth-anything package
import sys
sys.path.append('/home/wangshizun/projects/gsplat/depth_anything')
# from depth_anything.dpt import DepthAnything
# from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose
import torch.nn.functional as F
import tyro
from geometry import xy_grid

def resize_by_divide(image, divider=16):
    # image is tensor of shape (H, W, C)
    # resize the image to the size can be divided by divider
    H, W = image.shape[:2]
    # round up to the nearest multiple of divider
    H_new = H + (divider - H % divider) % divider
    W_new = W + (divider - W % divider) % divider
    # resize by pytorch and return tensor
    image_new = F.interpolate(image.permute(2, 0, 1).unsqueeze(0), (H_new, W_new), mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0)

    return image_new


def depth_estimate(image, encoder='vitl', device='cpu'):
    # img: [H, W, 3], range: [0, 1]
    # encoder = 'vits' # can also be 'vitb' or 'vitl'
    depth_anything = DepthAnything.from_pretrained(
        'LiheYoung/depth_anything_{:}14'.format(encoder), 
        cache_dir="/local_home/wangshizun/weights/").to(device).eval()

    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    # image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255.0
    image = image.cpu().numpy()
    h, w = image.shape[:2]
    image = transform({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0).to(device)

    # depth shape: 1xHxW
    with torch.no_grad():
        depth = depth_anything(image)
    depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    # depth = depth * 255.0
    
    # depth = depth.cpu().numpy().astype(np.uint8)

    return depth


def transform_probability_distribution(probabilities, alpha=0.1):
    # transformed_probabilities = probabilities + alpha * (1 - probabilities)
    # transformed_probabilities = probabilities + alpha * probabilities.max()
    # transformed_probabilities = np.sqrt(probabilities)
    transformed_probabilities = (probabilities + np.ones_like(probabilities) / probabilities.flatten().size) / 2
    normalized_probabilities = transformed_probabilities / np.sum(transformed_probabilities)
    return normalized_probabilities

def k_nearest_sklearn(x: torch.Tensor, k: int):
    """
        Find k-nearest neighbors using sklearn's NearestNeighbors.
    x: The data tensor of shape [num_samples, num_features]
    k: The number of neighbors to retrieve
    """
    # Convert tensor to numpy array
    # chech if x is a tensor or numpy array
    if isinstance(x, torch.Tensor):
        x_np = x.cpu().numpy()
    else:
        x_np = x

    # Build the nearest neighbors model
    from sklearn.neighbors import NearestNeighbors

    nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean").fit(x_np)

    # Find the k-nearest neighbors
    distances, indices = nn_model.kneighbors(x_np)

    # Exclude the point itself from the result and return
    return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)
    # return shape are [num_samples, k]

def image_sampling(gt_image, gt_depth, num_points=5000, device='cpu', mask=None):
    # gt_image: (H, W, 3) range: [0, 1]
    # gt_depth: (H, W), range: [0, 1]
    # read the image
    image = gt_image.cpu().numpy()*255
    H, W = image.shape[:2]

    # convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # compute the gradient of the image
    gradient_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # add uniform magnitude to the gradient magnitude, to avoid zero probability
    gradient_magnitude = gradient_magnitude + np.min(gradient_magnitude[gradient_magnitude>0])

    if mask is not None:
        gradient_magnitude = gradient_magnitude * mask.squeeze().numpy()


    # mediate the gradient magnitude
    probability_distribution = gradient_magnitude / np.sum(gradient_magnitude)

    # sample points from the probability distribution
    sampled_points = np.random.choice(np.arange(gray_image.size), size=num_points, p=probability_distribution.flatten())

    # convert the sampled points to coordinates on the image
    sampled_coordinates = np.unravel_index(sampled_points, gray_image.shape)

    # generate xy_grid
    xys = xy_grid(W, H)
    print(xys.shape)
    print(xys[...,0].min(), xys[...,0].max())
    print(xys[...,1].min(), xys[...,1].max())
    print("--------------")

    # convert the image to PIL format
    if gt_depth is None:
        gt_depth = depth_estimate(gt_image, device=device)
        gt_depth = gt_depth.unsqueeze(-1)

    XY = image.shape[:2][::-1]
    # xys = np.array(sampled_coordinates).T
    xys = np.array(sampled_coordinates).T[:,::-1].copy()
    xys_norm = (xys / np.array(XY) - 0.5) * 2

    depths_norm = gt_depth[sampled_coordinates]

    scales = 1 / probability_distribution[sampled_coordinates]
    scales_norm = scales / np.max(scales)

    rgbs = image[sampled_coordinates]
    rgbs_norm = rgbs / 255.

    return xys, depths_norm, scales_norm, rgbs_norm, gt_depth

def complex_texture_sampling(gt_image, gt_depth, num_points=5000, device='cpu', mask=None):
    # [ ] TODO clean up this method
    # gt_image: (H, W, 3) range: [0, 1]
    # gt_depth: (H, W), range: [0, 1]
    # read the image
    image = gt_image.cpu().numpy()*255
    H, W = image.shape[:2]

    # convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # compute the gradient of the image
    gradient_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # add uniform magnitude to the gradient magnitude, to avoid zero probability
    gradient_magnitude = gradient_magnitude + np.min(gradient_magnitude[gradient_magnitude>0])

    if mask is not None:
        gradient_magnitude = gradient_magnitude * mask.squeeze().numpy()


    # mediate the gradient magnitude
    probability_distribution = gradient_magnitude / np.sum(gradient_magnitude)

    # sample points from the probability distribution
    sampled_points = np.random.choice(np.arange(gray_image.size), size=num_points, p=probability_distribution.flatten())

    # convert the sampled points to coordinates on the image
    sampled_coordinates = np.unravel_index(sampled_points, gray_image.shape)

    # create a white image and mark the sampled points as orignial color
    # sampled_image = np.zeros_like(image)
    # sampled_image = np.ones_like(image) * 255
    # sampled_image[sampled_coordinates] = image[sampled_coordinates]

    # convert the image to PIL format
    if gt_depth is None:
        gt_depth = depth_estimate(gt_image, device=device)
        gt_depth = gt_depth.unsqueeze(-1)
        # gt_depth = resize_by_divide(gt_depth, 16)
    # else:
    #     img_depth_gt = imageio.imread(str(depth_gt_path)) / 255.0 
    #     img_depth_gt = torch.tensor(img_depth_gt, device=device).float()
    # print(img_depth_gt.shape)
    # print(img_depth_gt.max(), img_depth_gt.min())
    # sampled_image = np.repeat(img_depth_gt[..., np.newaxis], 3, axis=-1)

    XY = image.shape[:2][::-1]
    # xys = np.array(sampled_coordinates).T
    xys = np.array(sampled_coordinates).T[:,::-1].copy() # (num_points, 2) 2 is x, y, corresponding to W, H
    # print(xys.shape)
    # print(xys[:,0].min(), xys[:,0].max())
    # print(xys[:,1].min(), xys[:,1].max())
    # print("--------------")

    xys_norm = (xys / np.array(XY) - 0.5) * 2
    '''
    for depthAnything, 1 is near, 0 is far!!!
    '''
    # transform 1 is near to 0 is near
    # gt_depth =  1 - gt_depth
    # print(gt_depth.shape)
    # print(gt_depth.max(), gt_depth.min())
    depths_norm = gt_depth[sampled_coordinates]
    # print(depths_norm.shape)
    # depths_norm = img_depth_gt[sampled_coordinates][:,None]
    # depths_norm = (img_depth_gt[sampled_coordinates][:,None] - 0.5) * 2
    # print("depth_norm.shape", depth_norm.shape)
    # xyzs_norm = np.concatenate([xys_norm, depth_norm], axis=1)
    # print(xyzs_norm.shape)
    # if mask is not None:
    #     scales = 1 / probability_distribution_masked[sampled_coordinates]
    # else:
    #     scales = 1 / probability_distribution[sampled_coordinates]
    scales = 1 / probability_distribution[sampled_coordinates]
    # scales = np.ones_like(scales)
    # scales_norm = (1 / H) * (scales / np.max(scales))
    # scales_norm = 0.5 * scales / np.max(scales)
    scales_norm = scales * 100. / np.sum(scales)

    # scales_norm = 1 / num_points
    # scales_norm = np.clip(scales_norm, 0.001, np.inf)
    # scales_norm = scales_norm * (0.002 * H * W / num_points)
    # bgrs = image[sampled_coordinates]
    # bgr -> rgb
    # rgbs = bgrs[:,[2,1,0]]
    rgbs = image[sampled_coordinates]
    rgbs_norm = rgbs / 255.
    # rgbs_norm = rgbs 

    # img_depth_gt = img_depth_gt * 255.0
    # img_depth_gt = img_depth_gt.cpu().numpy().astype(np.uint8)
    # cv2.imwrite('/home/wangshizun/projects/gsplat/test/sampled_image1.jpg', img_depth_gt)

    return xys, depths_norm, scales_norm, rgbs_norm, gt_depth
    # return xys_norm, depths_norm, scales_norm, rgbs_norm, gt_depth
    # return xyzs_norm, scales_norm, rgbs_norm
    # return sampled_coordinates, depth
    # return sampled_coordinates, sampled_image

def main(
        image_folder: str = '/home/wangshizun/projects/gsplat/images/e2/e2',
) -> None:
    # image_path = '/home/wangshizun/projects/gsplat/images/beauty_0/beauty_0/00000.png'  # 替换成你的图片路径

    # sampled_coordinates, sampled_image = complex_texture_sampling(image_path, num_points=10000)
    
    # # 保存采样点图片
    # cv2.imwrite('/home/wangshizun/projects/gsplat/test/sampled_image1.jpg', sampled_image)

    # print("Sampled Points:", sampled_coordinates)
    # print("Sampled Points shape", np.array(sampled_coordinates).shape)

    # read a folder containing images, estimate the depth and then save the depth images

    image_paths = sorted(glob.glob(os.path.join(image_folder, '*.jpg'))) + sorted(glob.glob(os.path.join(image_folder, '*.png')))
    for image_path in tqdm(image_paths):
        img_depth_gt = depth_estimate(image_path, device='cuda')
        img_depth_gt = img_depth_gt * 255.0
        img_depth_gt = img_depth_gt.cpu().numpy().astype(np.uint8)
        img_depth_gt = Image.fromarray(img_depth_gt)
        img_depth_path = image_folder+'_depth'
        if not os.path.exists(img_depth_path):
            os.makedirs(img_depth_path)
        img_depth_gt.save(os.path.join(img_depth_path, os.path.basename(image_path)))

    # read depth images from a folder and normalize the depth images
    # depth_folder = '/home/wangshizun/projects/gsplat/images/beauty_0/beauty_0_depth'
    # depth_paths = glob.glob(os.path.join(depth_folder, '*.png'))
    # for depth_path in tqdm(depth_paths):
    #     img_depth_gt = Image.open(depth_path)
    #     img_depth_gt = np.array(img_depth_gt) / 255.0
    #     img_depth_gt = Image.fromarray(img_depth_gt)
    #     img_depth_gt.save(depth_path)


if __name__ == '__main__':
    tyro.cli(main)