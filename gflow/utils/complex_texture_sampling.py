import cv2
import numpy as np

def complex_texture_sampling(gt_image, gt_depth, num_points=5000, device='cpu', mask=None, drop_to=None):
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

    # mediate the gradient magnitude
    probability_distribution = gradient_magnitude / np.sum(gradient_magnitude)

    # sample points from the probability distribution
    sampled_points = np.random.choice(np.arange(gray_image.size), size=num_points, p=probability_distribution.flatten())
    if mask is not None:
        # cancel the sampled points that are in the mask
        mask_flat = mask.squeeze().numpy().flatten() # shape: (H,W) -> (H*W)
        sampled_points = sampled_points[~mask_flat[sampled_points]]
    
    if drop_to is not None:
        # random drop the points to drop_to
        if len(sampled_points) > drop_to:
            sampled_points = np.random.choice(sampled_points, size=drop_to, replace=False)

    # convert the sampled points to coordinates on the image
    sampled_coordinates = np.unravel_index(sampled_points, gray_image.shape)

    XY = image.shape[:2][::-1]
    xys = np.array(sampled_coordinates).T[:,::-1].copy() # (num_points, 2) 2 is x, y, corresponding to W, H

    depths_norm = gt_depth[sampled_coordinates]
    scales = 1 / probability_distribution[sampled_coordinates]
    scales_norm = scales * 100. / np.sum(scales)
    rgbs = image[sampled_coordinates]
    rgbs_norm = rgbs / 255.

    return xys, depths_norm, scales_norm, rgbs_norm, gt_depth