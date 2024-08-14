import path_to_mast3r

import os
import copy
import imageio
import torch
import tyro
import json
import numpy as np
from PIL import Image
import matplotlib
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment, make_dense_pts3d

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy


def main(parent_dir: str = None,
         seg_size: int = 200,
         scene_graph: str = 'logwin',
         mask_path: str = None,):
    device = 'cuda'
    lr1 = 0.07
    niter1 = 500
    lr2 = 0.014
    niter2 = 200
    optim_level = "refine"
    winsize = 3
    # cache_dir = "/local_home/wangshizun/cache/"
    cache_dir = "/home/wangshizun/cache/"
    win_cyclic = False
    refid = 0
    matching_conf_thr = 5.
    shared_intrinsics = True
    # [ ] TODO fix principal points

    # model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    model_name = "/home/wangshizun/projects/mast3r/weights/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)

    # input_dir = "/home/wangshizun/projects/msplat/data_share/DAVIS/images/car-turn-test/car-turn"

    # list all folders in the parent_dir
    folder_names = [f for f in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, f))]
    folder_names.sort()
    # flag = False

    for folder_name in folder_names:
        # if "india" in folder_name: flag = True
        # if not flag: continue
        if "car-turn-test" in folder_name: continue
        if "car-turn-1080" in folder_name: continue
        input_dir = os.path.join(parent_dir, folder_name, folder_name)
        print(f'>> processing {input_dir}')

        # load all images in the input_dir
        images_list = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.png') or f.endswith('.jpg')]
        # sort the images
        images_list.sort()

        # iterate over the images in the input_dir every seg_size images
        for i in range(0, len(images_list), seg_size):
            print(f'>> processing images from {i} to {i+seg_size}, total {len(images_list)} images')
            inference_mast3r(model, device, images_list[i:i+seg_size], input_dir, cache_dir,
                            optim_level, lr1, niter1, lr2, niter2, matching_conf_thr,
                            scene_graph, winsize, win_cyclic, refid, shared_intrinsics)

def inference_mast3r(model, device, images_list, input_dir, cache_dir,
                     optim_level, lr1, niter1, lr2, niter2, matching_conf_thr,
                     scenegraph_type, winsize, win_cyclic, refid, shared_intrinsics, **kw):
    """
    from a list of images, run mast3r inference, sparse global aligner.
    then run get_3D_model_from_scene
    """
    imgs = load_images(images_list, size=512, verbose=True)
    image0 = imageio.imread(images_list[0])
    orig_size = image0.shape[:2][::-1]

    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
        images_list = [images_list[0], images_list[0] + '_2']

    scene_graph_params = [scenegraph_type]
    if scenegraph_type in ["swin", "logwin"]:
        scene_graph_params.append(str(winsize))
    elif scenegraph_type == "oneref":
        scene_graph_params.append(str(refid))
    if scenegraph_type in ["swin", "logwin"] and not win_cyclic:
        scene_graph_params.append('noncyclic')
    scene_graph = '-'.join(scene_graph_params)
    pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True)
    print(f'>> {len(pairs)} pairs of images to process')
    if optim_level == 'coarse':
        niter2 = 0

    # Sparse GA (forward mast3r -> matching -> 3D optim -> 2D refinement -> triangulation)
    scene = sparse_global_alignment(images_list, pairs, cache_dir,
                                    model, lr1=lr1, niter1=niter1, lr2=lr2, niter2=niter2, device=device,
                                    opt_depth='depth' in optim_level, shared_intrinsics=shared_intrinsics,
                                    matching_conf_thr=matching_conf_thr, **kw)


    # retrieve useful values from scene:
    imgs = scene.imgs
    focals = scene.get_focals()
    pps = scene.get_principal_points()
    poses = scene.get_im_poses() # cam2world

    pts3d, depthmaps, confs = scene.get_dense_pts3d(clean_depth=True)

    new_size = imgs[0].shape[:2]
    new_to_orig_scale = (orig_size[0] + orig_size[1]) / (new_size[0] + new_size[1])
    # print("new_size", new_size)
    h, w = new_size

    # pts3d (N, 3) -> (H, W, 3)
    pts3d = [dm[:h*w].view(h, w, 3) for dm in pts3d]
    
    # depthmaps (N) -> (H, W)
    depthmaps = [dm[:h*w].view(h, w) for dm in depthmaps]

    # save depth path
    depth_save_path = input_dir + '_depth_mast3r_s2'
    os.makedirs(depth_save_path, exist_ok=True)
    pts3d_save_path = input_dir + '_pts3d_mast3r_s2'
    os.makedirs(pts3d_save_path, exist_ok=True)
    print("saving depth map")
    for i, depthmaps_i in enumerate(tqdm(depthmaps)):
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
        # depth_image = depth_image.resize(orig_size, Image.LANCZOS)
        # convert to rgb
        if depth_image.mode != "RGB":
            depth_image = depth_image.convert("RGB")

        depth_image.save(os.path.join(depth_save_path, f'{base_name_wo_extention}.png'))

    # save camera path
    camera_save_path = input_dir + '_camera_mast3r_s2'
    os.makedirs(camera_save_path, exist_ok=True)
    print("saving camera")
    for i, focal_pose_pp in enumerate(tqdm(zip(focals, poses, pps))):
        focal, pose, pp = focal_pose_pp
        focal = to_numpy(focal) * new_to_orig_scale
        pp = to_numpy(pp) * new_to_orig_scale
        
        pose = to_numpy(pose)
        pose = inv(pose) # world2cam
        pose[:3, 3] = pose[:3, 3] * new_to_orig_scale
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


if __name__ == '__main__':
    tyro.cli(main)