import path_to_dust3r

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images, rgb
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
import trimesh
import imageio
import matplotlib.pyplot as pl
import matplotlib
from PIL import Image
import torch
import os
import numpy as np
from scipy.spatial.transform import Rotation
import tyro
import json

def inv(mat):
    """ Invert a torch or numpy matrix
    """
    if isinstance(mat, torch.Tensor):
        return torch.linalg.inv(mat)
    if isinstance(mat, np.ndarray):
        return np.linalg.inv(mat)
    raise ValueError(f'bad matrix type = {type(mat)}')

def colorize(value, vmin=None, vmax=None, cmap='jet_r', invalid_val=-99, invalid_mask=None, background_color=(128, 128, 128, 255), gamma_corrected=False, value_transform=None):
    """Converts a depth map to a color image.

    Args:
        value (torch.Tensor, numpy.ndarry): Input depth map. Shape: (H, W) or (1, H, W) or (1, 1, H, W). All singular dimensions are squeezed
        vmin (float, optional): vmin-valued entries are mapped to start color of cmap. If None, value.min() is used. Defaults to None.
        vmax (float, optional):  vmax-valued entries are mapped to end color of cmap. If None, value.max() is used. Defaults to None.
        cmap (str, optional): matplotlib colormap to use. Defaults to 'magma_r'.
        invalid_val (int, optional): Specifies value of invalid pixels that should be colored as 'background_color'. Defaults to -99.
        invalid_mask (numpy.ndarray, optional): Boolean mask for invalid regions. Defaults to None.
        background_color (tuple[int], optional): 4-tuple RGB color to give to invalid pixels. Defaults to (128, 128, 128, 255).
        gamma_corrected (bool, optional): Apply gamma correction to colored image. Defaults to False.
        value_transform (Callable, optional): Apply transform function to valid pixels before coloring. Defaults to None.

    Returns:
        numpy.ndarray, dtype - uint8: Colored depth map. Shape: (H, W, 4)
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    value = value.squeeze()
    if invalid_mask is None:
        invalid_mask = value == invalid_val
    mask = np.logical_not(invalid_mask)

    # normalize
    vmin = np.percentile(value[mask],2) if vmin is None else vmin
    vmax = np.percentile(value[mask],85) if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.

    # squeeze last dim if it exists
    # grey out the invalid values

    value[invalid_mask] = np.nan
    cmapper = matplotlib.cm.get_cmap(cmap)
    if value_transform:
        value = value_transform(value)
        # value = value / value.max()
    value = cmapper(value, bytes=True)  # (nxmx4)

    # img = value[:, :, :]
    img = value[...]
    img[invalid_mask] = background_color

    #     return img.transpose((2, 0, 1))
    if gamma_corrected:
        # gamma correction
        img = img / 255
        img = np.power(img, 2.2)
        img = img * 255
        img = img.astype(np.uint8)
    return img

def main(input_dir: str,
         seg_size: int = 10,
         scene_graph: str = 'complete'):
    device = 'cuda'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    # load all images in the input_dir
    images_list = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.png') or f.endswith('.jpg')]
    # sort the images
    images_list.sort()

    model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    # you can put the path to a local checkpoint in model_name if needed
    # model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    model = AsymmetricCroCo3DStereo.from_pretrained("/home/wangshizun/projects/gflow/third_party/dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth").to(device)

    # iterate over the images in the input_dir every seg_size images
    for i in range(0, len(images_list), seg_size):
        print(f'>> processing images from {i} to {i+seg_size}, total {len(images_list)} images')
        inference_dust3r(model, images_list[i:i+seg_size], input_dir, batch_size, device, schedule, lr, niter, scene_graph)

def inference_dust3r(model, images_list, input_dir, batch_size, device, schedule, lr, niter, scene_graph="complete"):
    # load_images can take a list of images or a directory
    images = load_images(images_list, size=512)
    image0 = imageio.imread(images_list[0])
    orig_size = image0.shape[:2][::-1]
    # print(orig_size)
    # pairs = make_pairs(images, scene_graph='swin', prefilter=None, symmetrize=True)
    pairs = make_pairs(images, scene_graph=scene_graph, prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size)

    # at this stage, you have the raw dust3r predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']
    # here, view1, pred1, view2, pred2 are dicts of lists of len(2)
    #  -> because we symmetrize we have (im1, im2) and (im2, im1) pairs
    # in each view you have:
    # an integer image identifier: view1['idx'] and view2['idx']
    # the img: view1['img'] and view2['img']
    # the image shape: view1['true_shape'] and view2['true_shape']
    # an instance string output by the dataloader: view1['instance'] and view2['instance']
    # pred1 and pred2 contains the confidence values: pred1['conf'] and pred2['conf']
    # pred1 contains 3D points for view1['img'] in view1['img'] space: pred1['pts3d']
    # pred2 contains 3D points for view2['img'] in view1['img'] space: pred2['pts3d_in_other_view']

    # next we'll use the global_aligner to align the predictions
    # depending on your task, you may be fine with the raw output and not need it
    # with only two input images, you could use GlobalAlignerMode.PairViewer: it would just convert the output
    # if using GlobalAlignerMode.PairViewer, no need to run compute_global_alignment
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

    # retrieve useful values from scene:
    imgs = scene.imgs
    focals = scene.get_focals()
    pps = scene.get_principal_points()
    poses = scene.get_im_poses()
    pts3d = scene.get_pts3d(raw=True)

    depthmaps = scene.get_depthmaps() # raw=True
    confidence_masks = scene.get_masks()

    new_size = imgs[0].shape[:2]
    new_to_orig_scale = (orig_size[0] + orig_size[1]) / (new_size[0] + new_size[1])

    # save depth path
    depth_save_path = input_dir + '_depth_dust3r'
    os.makedirs(depth_save_path, exist_ok=True)
    pts3d_save_path = input_dir + '_pts3d_dust3r'
    os.makedirs(pts3d_save_path, exist_ok=True)
    for i, depthmaps_i in enumerate(depthmaps):
        # extrat every file name in images_list
        base_name_wo_extention = os.path.basename(images_list[i]).split('.')[0]
        depth = to_numpy(depthmaps_i)
        pts3d_i  = to_numpy(pts3d[i])

        # interpolate the depth using lanczos
        depth_save_for_np = Image.fromarray(depth)
        depth_save_for_np = depth_save_for_np.resize(orig_size, Image.LANCZOS)
        depth_save_for_np = np.array(depth_save_for_np)
        # print(depth_save_for_np.shape)
        np.save(os.path.join(depth_save_path, f'{base_name_wo_extention}.npy'), depth_save_for_np)
        # print(depth.shape)
        np.save(os.path.join(pts3d_save_path, f'{base_name_wo_extention}.npy'), pts3d_i)

        # save rgb image
        depth_image = Image.fromarray(colorize(depth))
        depth_image = depth_image.resize(orig_size, Image.LANCZOS)
        # convert to rgb
        if depth_image.mode != "RGB":
            depth_image = depth_image.convert("RGB")

        depth_image.save(os.path.join(depth_save_path, f'{base_name_wo_extention}.png'))

    # save camera path
    camera_save_path = input_dir + '_camera_dust3r'
    os.makedirs(camera_save_path, exist_ok=True)
    for i, focal_pose_pp in enumerate(zip(focals, poses, pps)):
        focal, pose, pp = focal_pose_pp
        focal = to_numpy(focal) * new_to_orig_scale
        pp = to_numpy(pp) * new_to_orig_scale
        
        pose = to_numpy(pose)
        # TODO cancel pose convention translation
        # print("pose before\n", pose)
        rot = np.eye(4)
        rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
        # pose = inv(pose @ OPENGL @ rot)
        # pose = pose @ OPENGL @ rot
        pose = inv(pose)
        # print("pose after\n", pose)

        base_name_wo_extention = os.path.basename(images_list[i]).split('.')[0]
        # save pose in json
        camera_dict = {
            'focal': focal.tolist(),
            'pose': pose.tolist(),
            'pp': pp.tolist(),
        }

        with open(os.path.join(camera_save_path, f'{base_name_wo_extention}.json'), 'w') as f:
            json.dump(camera_dict, f, indent=4)

if __name__ == '__main__':
    tyro.cli(main)
    # TODO shared intrinsics
    # TODO coherent depth and camera intr
    # TODO depth: Bilateral Filter, Median Filter, Guided Filter