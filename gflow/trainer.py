import math
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import imageio
import msplat
import utils
from datetime import datetime
import os
import utils.render as render
import cv2
import utils.geometry as geometry
from sklearn.cluster import KMeans
import roma

class SimpleGaussian:
    def __init__(self, gt_image, gt_depth=None, gt_flow=None, num_points=100000, background="black", sequence_path=None, logs_suffix="_logs", common_logs=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gt_image = gt_image.to(self.device)
        self.gt_depth = gt_depth.to(self.device) if gt_depth is not None else None
        self.gt_flow = gt_flow.to(self.device) if gt_flow is not None else None
        self.num_points = num_points
        self.clustering = KMeans(n_clusters=2, random_state=0)

        H, W, C = gt_image.shape 
        self.H = H
        self.W = W
        if background == "black":
            self.bg = 0.
        elif background == "white":
            self.bg = 1.
        elif background == "cyan":
            self.bg = 0.33
        else: 
            self.bg =0.
        fov = math.pi / 2.0
        fx = 0.5 * float(W) / math.tan(0.5 * fov)
        fy = 0.5 * float(H) / math.tan(0.5 * fov)
        self.intr = torch.Tensor([fx, fy, float(W) / 2, float(H) / 2]).cuda().float() 
        self.pose = torch.tensor([0., 0., 0., 1., 0., 0., 0.]).cuda().float()
        self.extr = self.get_extr() # world2camera
        # [ ] TODO create a camera class, opt for learning residual of camera pose

        # [ ] TODO support optimize multi-view consistent depth within a window
        
        N = int(num_points)

        def _2dgs(x):
            x = (torch.abs(x) + 1e-8).repeat(1, 2)
            # add the third channel to be zero
            x = torch.cat((x, torch.zeros_like(x[:, :1])), dim=1)
            return x
        
        def _isotropic(x):
            return (torch.abs(x) + 1e-8).repeat(1, 3)

        def sensitive_sigmoid(x, scale=10.):
            return torch.sigmoid(x * scale)
        
        def sensitive_logit(x, scale=10.):
            return torch.logit(x) / scale
        
        self._activations = {
            "scale": lambda x: torch.abs(x),
            "rotate": torch.nn.functional.normalize,
            "opacity": sensitive_sigmoid,
            "rgb": torch.sigmoid
        }
        
        # the inverse of the activation functions
        self._activations_inv = {
            "scale": lambda x: torch.abs(x),
            "rotate": torch.nn.functional.normalize,
            "opacity": sensitive_logit,
            "rgb": torch.logit
        }

        self._attributes = {
            "xyz":      torch.rand((N, 3), dtype=torch.float32).cuda() * 2 - 1,
            # "scale":    torch.rand((N, 1), dtype=torch.float32).cuda(),
            "scale":    torch.rand((N, 3), dtype=torch.float32).cuda(),
            "rotate":   self._activations_inv["rotate"](torch.rand((N, 4), dtype=torch.float32).cuda()), # TODO should be a quaternion
            "opacity":  self._activations_inv["opacity"](0.99*torch.ones((N, 1), dtype=torch.float32).cuda()),
            "rgb":      torch.rand((N, 3), dtype=torch.float32).cuda()
        }

        
        # Get current date and time as string
        now = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

        # Create a directory with the current date and time as its name
        logs_path = "logs"
        if common_logs:
            if logs_suffix is None:
                logs_path = "logs"
            else:
                logs_path = logs_suffix
        else:
            if logs_suffix is None:
                logs_path = str(sequence_path) + "_logs"
            else:
                logs_path = str(sequence_path) + f"_{logs_suffix}"
        log_now_path = os.path.join(logs_path, f"{now}")
        os.makedirs(log_now_path, exist_ok=True)
        log_latest_path = os.path.join(logs_path, "0_latest")
        os.makedirs(log_latest_path, exist_ok=True)
        # make a new directory, always soft link to the current run
        log_now_path_abs = os.path.abspath(log_now_path)
        os.system(f"rm {log_latest_path}/*")
        os.system(f"ln -s {log_now_path_abs} {log_latest_path}/{now}")
        self.dir = log_now_path

    # [ ] TODO @property
    def get_extr(self):
        # normalize rotation
        Q = self.pose[:4]
        T = utils.signed_expm1(self.pose[4:7])
        RT = roma.RigidUnitQuat(Q, T).normalize().to_homogeneous() # (4, 4)
        extr = RT[:3, :] # (3, 4)
        return extr

    def add_optimizer(self, lr=1e-2, lr_camera=0., exclude_key=None, camera_only=False, depth_invariant=True):
        self.lr = lr
        self.lr_camera = lr_camera
        for attribute_name in self._attributes.keys():
            if attribute_name != exclude_key:
                self._attributes[attribute_name] = nn.Parameter(self._attributes[attribute_name]).requires_grad_(True)
        # self.extr = nn.Parameter(self.extr).requires_grad_(True)
        self.pose = nn.Parameter(self.pose).requires_grad_(True)

        # assign different lr to attributes and extr
        optim_params=[
            {"params": self._attributes.values(), "lr": lr, "name": "attributes"},
            # {"params": self.extr, "lr": lr_camera, "name": "extr"},
            {"params": self.pose, "lr": lr_camera, "name": "extr"},
        ]

        if hasattr(self, 'pose_list'):
            for idx in range(len(self.pose_list)):
                # self.pose_list[idx] = nn.Parameter(self.pose_list[idx]).requires_grad_(True)
                optim_params.append({"params": self.pose_list[idx], "lr": lr_camera, "name": f"extr_{idx}"})

        if depth_invariant:
            self.depth_a = nn.Parameter(torch.ones(1).to(self.device)).requires_grad_(True)
            self.depth_b = nn.Parameter(torch.zeros(1).to(self.device)).requires_grad_(True)
            optim_params.append({"params": self.depth_a, "lr": lr, "name": "depth_a"})
            optim_params.append({"params": self.depth_b, "lr": lr, "name": "depth_b"})
        else:
            self.depth_a = nn.Parameter(torch.ones(1).to(self.device)).requires_grad_(False)
            self.depth_b = nn.Parameter(torch.zeros(1).to(self.device)).requires_grad_(False)
        
        self.optimizer = torch.optim.Adam(optim_params)

    def set_gt_image(self, gt_image):
        self.gt_image = gt_image.to(self.device)

    def set_gt_depth(self, gt_depth):
        self.gt_depth = gt_depth.to(self.device)

    def set_gt_flow(self, gt_flow):
        self.gt_flow = gt_flow.to(self.device)

    def load_camera(self, focal=None, pp=None, extr=None, scale=None, show=True):
        if focal is not None:
            self.intr[:2] = torch.tensor([focal, focal], dtype=torch.float32).to(self.device)
        if pp is not None:
            self.intr[2:] = torch.tensor(pp, dtype=torch.float32).to(self.device)
        if extr is not None:
            R = torch.tensor(extr[:3, :3]).to(self.device)
            T = torch.tensor(extr[:3, 3]).to(self.device)
            if scale is not None:
                # print("[camera] T before: ", T)
                T = T * scale
                # print("[camera] T after: ", T)
            pose = self.pose.detach().clone()
            pose[0:4] = roma.rotmat_to_unitquat(R)
            pose[4:7] = utils.signed_log1p(T)
            self.pose = nn.Parameter(pose).requires_grad_(True)
        if show:
            print("[camera] intr: ", self.intr)
            # print("[camera] extr: \n", self.extr)
            print("[camera] extr: \n", self.get_extr())
    
    def choose_idx(self, idx=None):
        if idx is None:
            idx = np.random.randint(0, 3)
        return idx

    def choose_extr(self, idx=None):
        if idx is None:
            # random choose one from self.pose_list
            idx = np.random.randint(0, len(self.pose_list))
        Q = self.pose_list[idx][:4]
        T = utils.signed_expm1(self.pose_list[idx][4:7])
        RT = roma.RigidUnitQuat(Q, T).normalize().to_homogeneous() # (4, 4)
        extr = RT[:3, :] # (3, 4)

        return extr
    
    def choose_move_mask(self, idx=None):
        if idx is None:
            idx = np.random.randint(0, len(self.move_masks))
        return self.move_masks[idx]
    
    def init_gaussians_from_image(self, gt_image, gt_depth=None, num_points=None, mask=None, drop_to=None):
        # gt_image (H, W, C) in [0, 1]
        if num_points is None:
            num_points = self.num_points
        xys, depths, scales, rgbs, gt_depth = utils.complex_texture_sampling(gt_image, gt_depth, num_points=num_points, device=self.device, mask=mask, drop_to=drop_to)
        new_num_points = xys.shape[0]
        """depths: 0 is near, 1 is far"""
        xys = torch.from_numpy(xys).to(self.device).float()
        depths = depths.to(self.device).float()
        self.gt_depth = gt_depth.to(self.device).float()

        self._attributes["xyz"] = geometry.pix2world(xys, depths, self.intr, self.get_extr())

        print("[init] x range: ", self._attributes["xyz"][:,0].min().item(), self._attributes["xyz"][:,0].max().item())
        print("[init] y range: ", self._attributes["xyz"][:,1].min().item(), self._attributes["xyz"][:,1].max().item())
        print("[init] z range: ", self._attributes["xyz"][:,2].min().item(), self._attributes["xyz"][:,2].max().item())

        scales = scales * (depths/depths.min()).squeeze().cpu().numpy()
        scales = torch.from_numpy(scales).float().unsqueeze(1).repeat(1, 3).to(self.device)
        scales = torch.clamp(scales, max=1e-3)
        self._attributes["scale"] = self._activations_inv["scale"](scales)

        rgbs = torch.from_numpy(rgbs).float().contiguous().to(self.device)
        eps = 1e-15  # avoid logit function input 0 or 1
        rgbs = torch.clamp(rgbs, min=eps, max=1-eps)
        # calculate the inverse of sigmoid function, i.e., logit function
        self._attributes["rgb"] = self._activations_inv["rgb"](rgbs)

        opacity = 0.99 * torch.ones((new_num_points, 1), device=self.device).float()
        self._attributes["opacity"] = self._activations_inv["opacity"](opacity)

        rotate = torch.rand((new_num_points, 4), dtype=torch.float32).cuda()
        self._attributes["rotate"] = self._activations_inv["rotate"](rotate)
        
    def current_pts_num(self):
        return self._attributes["xyz"].shape[0]
    
    def get_attribute(self, name):
        try:
            if name in self._activations.keys() and self._activations[name] is not None:
                return self._activations[name](self._attributes[name])
            else:
                return self._attributes[name]
        except:
            raise ValueError(f"Attribute or activation for {name} is not VALID!")

    def save_checkpoint(self, ckpt_name=None, camera_only=False):
        # save checkpoint
        checkpoint = {
            "attributes": self._attributes,
            "intr": self.intr,
            # "extr": self.extr,
            "extr": self.get_extr(),
            "still_mask": self.still_mask if hasattr(self, 'still_mask') else None,
            "move_seg": self.move_seg if hasattr(self, 'move_seg') else None,
            "last_uv": self.last_uv if hasattr(self, 'last_uv') else None,
            "width": self.W,
            "height": self.H,
        }
        if hasattr(self, 'pose_list'):
            checkpoint["pose_list"] = self.pose_list
        if ckpt_name is None:
            ckpt_name = 'ckpt'
        # make directory
        os.makedirs(os.path.join(self.dir, "ckpt"), exist_ok=True)
        self.checkpoint_path = os.path.join(self.dir, "ckpt", f"{ckpt_name}.tar")
        torch.save(checkpoint, self.checkpoint_path)

    def load_checkpoint(self, checkpoint_path, show=True):
        checkpoint = torch.load(checkpoint_path)
        self._attributes = checkpoint["attributes"]
        self.intr = checkpoint["intr"]
        self.extr = checkpoint["extr"]
        # self.pose = self.load_camera(extr=self.extr)
        self.load_camera(extr=self.extr, show=show)
        if "still_mask" in checkpoint.keys():
            self.still_mask = checkpoint["still_mask"]
        if "move_seg" in checkpoint.keys():
            self.move_seg = checkpoint["move_seg"]
        if "last_uv" in checkpoint.keys():
            self.last_uv = checkpoint["last_uv"]
        del checkpoint
        torch.cuda.empty_cache()

    def init_mask_prompt_pts(self, mask_prompt, ckpt_name):
        input_group = [
            self.get_attribute("xyz"),
            self.get_attribute("scale"),
            self.get_attribute("rotate"),
            self.get_attribute("opacity"),
            self.get_attribute("rgb"),
            self.intr,
            # self.extr,
            self.get_extr(),
            self.bg,
            self.W,
            self.H,
        ]

        return_dict = render.render_multiple(
            input_group,
            ["uv", "center"]
        )
        uv = return_dict["uv"].detach() # (N, 2)
        uv_within = (uv[:,0] > 0) & (uv[:,0] < self.W-1) & (uv[:,1] > 0) & (uv[:,1] < self.H-1) # (N,)
        uv = uv[uv_within] # (N_within, 2)
        # print(uv[:,1][:10])
        y_coords = uv[:,1].long() # (N_within,)
        x_coords = uv[:,0].long() # (N_within,)
        # print(y_coords[:10])
        print(y_coords.max(), y_coords.min())
        print(x_coords.max(), x_coords.min())
        print(mask_prompt.shape)

        mask_np = mask_prompt.detach().cpu().numpy() * 255
        mask_np = mask_np.astype(np.uint8)
        # save
        os.makedirs(os.path.join(self.dir, "images_seg"), exist_ok=True)
        imageio.imwrite(os.path.join(self.dir, "images_seg", f"propagate_mask_{ckpt_name}.png"), mask_np)


        # select the points that are within the mask
        mask_prompt_pts = mask_prompt[y_coords, x_coords] # (N_within,)
        self.mask_prompt_pts = uv_within.clone()
        self.mask_prompt_pts[uv_within] = mask_prompt_pts

    def train(self, iterations=500, lr=1e-2, lr_camera=0., lambda_rgb=1.,
              lambda_depth=0., lambda_flow=0., lambda_var=0., lambda_still=0., lambda_scale=0.,
              save_imgs=False, save_videos=False, save_ckpt=False, move_mask=None,
              ckpt_name="ckpt", densify_interval=500, densify_times=1, densify_iter=0,
              grad_threshold=5e-3, mask=None, camera_only=False, eps=10, min_samples=20,
              densify_occ_percent=0.1, densify_err_thre=1e-2, densify_err_percent=0.2):
        frames = []
        frames_depth = []
        frames_center = []
        progress_bar = tqdm(range(1, iterations), desc="Training")
        l1_loss = nn.SmoothL1Loss(reduce="none")
        mse_loss = nn.MSELoss()
        mse_loss_pixel = nn.MSELoss(reduction='none')
        utils.SSIM_loss = utils.SSIM()

        """pre-update prosessing"""
        if not camera_only and hasattr(self, 'still_mask'):
            uv_move = self.last_uv[:self.last_still_mask.shape[0]][~self.last_still_mask] # (N_move, 2)
            # within the image
            uv_within_index = (uv_move[:,0] > 0) & (uv_move[:,0] < self.W-1) & (uv_move[:,1] > 0) & (uv_move[:,1] < self.H-1) # (N_move,)
            uv_move = uv_move[uv_within_index] # (N_move_within, 2)
            y_coords = uv_move[:,1].long() # (N_move_within,)
            x_coords = uv_move[:,0].long() # (N_move_within,)
            move_flow = self.gt_flow[y_coords, x_coords] # (N_move_within, 2)
            uv_move = uv_move + move_flow  # (N_move_within, 2)
            y_coords_move = uv_move[:,1].long() # (N_move_within,)
            x_coords_move = uv_move[:,0].long() # (N_move_within,)
            # clamp the coords
            y_coords_move = torch.clamp(y_coords_move, 0, self.H-1)
            x_coords_move = torch.clamp(x_coords_move, 0, self.W-1)
            depth_move = self.gt_depth[y_coords_move, x_coords_move] # (N_move_within,)
            # xyz_move = geometry.pix2world(uv_move, depth_move, self.intr, self.extr) # (N_move_within, 3)
            xyz_move = geometry.pix2world(uv_move, depth_move, self.intr, self.get_extr()) # (N_move_within, 3)
            # assign the new xyz to the moving part
            xyz = self._attributes["xyz"].clone()

            temp_1 = xyz[:self.last_still_mask.shape[0]][~self.last_still_mask].clone()
            temp_1[uv_within_index] = xyz_move.detach()

            temp_2 = xyz[:self.last_still_mask.shape[0]].clone()
            temp_2[~self.last_still_mask] = temp_1

            xyz[:self.last_still_mask.shape[0]] = temp_2

            self._attributes["xyz"] = xyz
            # xyz = self._attributes["xyz"].clone()
            # temp_xyz = xyz[:self.last_still_mask.shape[0]].detach()
            # temp_xyz[~self.last_still_mask] = xyz_move.detach()
            # xyz[:self.last_still_mask.shape[0]] = temp_xyz
            # self._attributes["xyz"] = xyz

        self.add_optimizer(lr, lr_camera, exclude_key=None, camera_only=camera_only, depth_invariant=True)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=iterations)

        """iterate the optimization"""
        for iteration in range(0, iterations):
            loss = 0.

            input_group = [
                self.get_attribute("xyz"),
                self.get_attribute("scale"),
                self.get_attribute("rotate"),
                self.get_attribute("opacity"),
                self.get_attribute("rgb"),
                self.intr,
                self.get_extr(),
                # self.extr,
                self.bg,
                self.W,
                self.H,
            ]

            return_dict = render.render_multiple(
                input_group,
                ["rgb", "uv", "depth", "depth_map", "depth_map_color", "center"]
            )

            # render image
            rendered_rgb, uv, depth = return_dict["rgb"], return_dict["uv"], return_dict["depth"]

            # render depth map
            rendered_depth_map = return_dict["depth_map"]

            # render colorful depth map
            rendered_depth_map_color = return_dict["depth_map_color"]
            
            # render center
            rendered_center = return_dict["center"]

            progress_dict = {}

            # valid_uv = uv within the a HxW range
            valid_uv_index = ((uv[:,0] > 0) & (uv[:,0] < self.W-1) & (uv[:,1] > 0) & (uv[:,1] < self.H-1)) # (N,)
            self.within_index = valid_uv_index
            # [ ] TODO do not calculate loss on the moving part
            if hasattr(self, 'still_mask_tentative') and camera_only: # if camera_only, combine the input move mask and mask of moving gs.
                # """
                input_group_temp = [
                    self.get_attribute("xyz").detach()[:self.still_mask_tentative.shape[0]][~self.still_mask_tentative],
                    self.get_attribute("scale").detach()[:self.still_mask_tentative.shape[0]][~self.still_mask_tentative],
                    self.get_attribute("rotate").detach()[:self.still_mask_tentative.shape[0]][~self.still_mask_tentative],
                    self.get_attribute("opacity").detach()[:self.still_mask_tentative.shape[0]][~self.still_mask_tentative],
                    self.get_attribute("rgb").detach()[:self.still_mask_tentative.shape[0]][~self.still_mask_tentative],
                    self.intr,
                    self.get_extr(),
                    # self.extr,
                    self.bg,
                    self.W,
                    self.H,
                ]

                return_dict_temp = render.render_multiple(
                    input_group_temp,
                    ["rgb"]
                )
                move_gs_rgb = return_dict_temp["rgb"] # shape (3, H, W)
                # print(move_gs_rgb.max(), move_gs_rgb.min())
                move_gs_grey = 0.299 * move_gs_rgb[0, :, :] + 0.587 * move_gs_rgb[1, :, :] + 0.114 * move_gs_rgb[2, :, :] # shape (H, W)
                move_gs_mask = move_gs_grey > 0.
                move_mask = move_gs_mask | move_mask
            if lambda_rgb > 0:
                if camera_only:
                    rendered_rgb = rendered_rgb * ~move_mask.unsqueeze(0)
                    gt_image = self.gt_image * ~move_mask.unsqueeze(-1)
                else:
                    gt_image = self.gt_image

                loss_rgb_pixel = mse_loss_pixel(rendered_rgb.permute(1, 2, 0), gt_image).mean(dim=2)
                loss_rgb = loss_rgb_pixel.mean()
                loss_ssim = 1-utils.SSIM_loss(rendered_rgb.unsqueeze(0), gt_image.permute(2, 0, 1).unsqueeze(0))
                loss_rgb = loss_rgb + loss_ssim
                loss += lambda_rgb * loss_rgb
                progress_dict["rgb"] = f"{loss_rgb.item():.6f}"

            rendered_depth_map = rendered_depth_map.permute(1, 2, 0)
            if hasattr(self, 'still_mask'):
                if camera_only: # only optimize the still part
                    valid_uv_index[:self.still_mask.shape[0]] = self.still_mask & valid_uv_index[:self.still_mask.shape[0]]
                else: # only optimize the move part
                    valid_uv_index[:self.still_mask.shape[0]] = ~self.still_mask & valid_uv_index[:self.still_mask.shape[0]]
            valid_uv = uv[valid_uv_index]
            depth_point_gt = self.gt_depth[valid_uv[:,1].long(), valid_uv[:,0].long()] # shape (N, 1)
            depth_point = depth[valid_uv_index] # shape (N, 1)

            if lambda_depth > 0:

                # rendered_depth_map_norm = rendered_depth_map
                rendered_depth_map_norm = self.depth_a * rendered_depth_map + self.depth_b # scale and shift invariant depth loss
                gt_depth_norm = self.gt_depth

                loss_depth = mse_loss_pixel(rendered_depth_map_norm, gt_depth_norm) / (rendered_depth_map_norm + gt_depth_norm)
                if camera_only:
                    loss_depth = loss_depth * ~move_mask.unsqueeze(-1)
                loss_depth = loss_depth.mean()

                loss += lambda_depth * loss_depth
                progress_dict["depth"] = f"{loss_depth.item():.6f}"

            if lambda_var: # penalize the scales with large variance, to avoid needle-like artifacts
                loss_var = torch.mean(torch.std(self.get_attribute("scale"), dim=1))
                loss += lambda_var * loss_var
                progress_dict["var"] = f"{loss_var.item():.6f}"

            if lambda_scale: # penalize the gaussian points with large scales, by L2 loss
                loss_scale = torch.norm(self.get_attribute("scale")[self.within_index], dim=1)
                depth_norm = 1 / depth_point
                # import pdb; pdb.set_trace()
                loss_scale = loss_scale * depth_norm.squeeze()
                loss_scale = loss_scale.mean()
                loss += lambda_scale * loss_scale
                progress_dict["scale"] = f"{loss_scale.item():.6f}"


            if lambda_still and hasattr(self, 'still_mask'): # pushing still gaussians to be still
                still_shape = self.last_still_mask.shape[0]
                loss_still = torch.norm(self.get_attribute("xyz")[:still_shape][self.last_still_mask] - self.last_xyz[:still_shape][self.last_still_mask], dim=1).mean()
                loss += lambda_still * loss_still
                progress_dict["still"] = f"{loss_still.item():.6f}"

            if lambda_flow and hasattr(self, "gt_flow"): # ensure local flow consistency
                and_mask = (self.last_uv[:,0] > 0) & (self.last_uv[:,0] < self.W-1) & (self.last_uv[:,1] > 0) & (self.last_uv[:,1] < self.H-1)
                if hasattr(self, 'still_mask'):
                    if camera_only: # only optimize the still part
                        and_mask[:self.still_mask.shape[0]] = self.still_mask & and_mask[:self.still_mask.shape[0]]
                    else: # only optimize the move part
                        and_mask[:self.still_mask.shape[0]] = ~self.still_mask & and_mask[:self.still_mask.shape[0]]
                and_mask = and_mask.detach()

                pred_flow = uv[:self.last_num][and_mask] - self.last_uv[and_mask]

                y_coords_last = self.last_uv[and_mask][:,1].long()
                x_coords_last = self.last_uv[and_mask][:,0].long()

                gt_flow = self.gt_flow[y_coords_last, x_coords_last]


                loss_flow = mse_loss(pred_flow, gt_flow)
                loss += lambda_flow * loss_flow
                progress_dict["flow"] = f"{loss_flow.item():.6f}"

            self.optimizer.zero_grad()
            loss.backward()

            """gradient control"""
            # control all gaussians's gradients to be zero: rgb 
            if hasattr(self, 'last_xyz'): # second frame and after
                for name, param in self._attributes.items():
                    if name == "rgb" and param.grad is not None:
                        param.grad = 0. * param.grad
            
            # control still gaussians's gradients to be zero: all
            if hasattr(self, 'still_mask'): # second frame and after
                for name, param in self._attributes.items():
                    if name == "xyz" and param.grad is not None:
                        param.grad[:self.still_mask.shape[0]][self.still_mask] = 0. * param.grad[:self.still_mask.shape[0]][self.still_mask]

            if camera_only:
                for name, param in self._attributes.items():
                    if param.grad is not None:
                        param.grad = 0. * param.grad

            """update the parameters"""
            self.optimizer.step()
            self.scheduler.step()
            progress_dict["total"] = loss.item()
            progress_bar.set_postfix(progress_dict)
            progress_bar.update(1)

            """densification"""
            # desify the occluded part
            if not camera_only and iteration == 0 and hasattr(self, 'last_xyz'): # second frame and after
                if mask.sum() > 0: # if there exists occluded points
                    self.densify_by_pixels(torch.ones_like(loss_rgb_pixel), error_threshold=0., percent=densify_occ_percent, mask=mask)

            # densify the detail-lacking part
            if not camera_only and densify_interval and (iteration+1) % densify_interval == 0 and (iteration+1) // densify_interval <= densify_times:
                if not hasattr(self, 'last_xyz'): # the first frame
                    self.densify_by_pixels(loss_rgb_pixel, error_threshold=densify_err_thre, percent=densify_err_percent, mask=None)
                else:
                    self.densify_by_pixels(loss_rgb_pixel, error_threshold=densify_err_thre, percent=densify_err_percent, mask=None) # [ ] TODO check densify hyperparameters

            if iteration % 10 == 0:
                # rgb
                rendered_rgb_np = render.render2img(rendered_rgb)
                frames.append(rendered_rgb_np)
                # depth map
                rendered_depth_map_color_np = render.render2img(rendered_depth_map_color)
                frames_depth.append(rendered_depth_map_color_np)
                # center
                rendered_center_np = render.render2img(rendered_center)
                frames_center.append(rendered_center_np)
        
        progress_bar.close()
        # import pdb; pdb.set_trace()
        
        """post-update prosessing"""
        if not camera_only:
            #### update still_mask
            within_mask = (uv[...,0] > 0) & (uv[...,0] < self.W-1) & (uv[...,1] > 0) & (uv[...,1] < self.H-1) # (N,)
            y_coords = uv[within_mask][:,1].long()
            x_coords = uv[within_mask][:,0].long()
            labels = ~move_mask[y_coords, x_coords] # (N_within,)

            self.still_mask = torch.ones(self.current_pts_num(), dtype=torch.bool).to(self.device).requires_grad_(False) # (N,)
            self.still_mask[within_mask] = labels
            self.still_mask_tentative = self.still_mask.detach().clone()
            if hasattr(self, 'last_still_mask'):
                self.still_mask[:self.last_still_mask.shape[0]] = self.last_still_mask

            # print mask ratio
            print("\t[still] mask ratio is", self.still_mask.sum().item() / self.still_mask.size(0))
            
            if uv[within_mask & ~self.still_mask].size(0) > 5:
                moving_seg_cluster = utils.FastConcaveHull2D(uv[within_mask & ~self.still_mask])
                ### get the convex hull segmentation
                self.move_seg = moving_seg_cluster.mask(self.W, self.H)
                self.move_seg = (self.move_seg * 255).astype(np.uint8)
                self.move_seg_erode = cv2.erode(self.move_seg, np.ones((20,20),np.uint8), iterations = 1)

            if hasattr(self, 'mask_prompt_pts'):
                propagate_uv = uv[:self.mask_prompt_pts.shape[0]][self.mask_prompt_pts]
                propagate_uv_within = (propagate_uv[:,0] > 0) & (propagate_uv[:,0] < self.W-1) & (propagate_uv[:,1] > 0) & (propagate_uv[:,1] < self.H-1)
                propagate_uv = propagate_uv[propagate_uv_within]
                if propagate_uv.size(0) > 4:
                    propagate_seg_cluster = utils.FastConcaveHull2D(propagate_uv)
                    self.propagate_seg = propagate_seg_cluster.mask(self.W, self.H)
                    self.propagate_seg = (self.propagate_seg * 255).astype(np.uint8)
                
            ### save current variables as last_variables for future use
            self.last_still_mask = self.still_mask.detach()
            self.last_uv = uv.detach()
            self.last_depth = depth.detach()
            self.last_xyz = self.get_attribute("xyz").detach()
            self.last_num = self.last_xyz.shape[0]
        
        ### render still points and moving points
        still_rgb_np = None
        still_center_np = None
        move_rgb_np = None
        move_center_np = None
        if hasattr(self, 'still_mask'):
            # render still points
            input_group = [
                self.get_attribute("xyz")[:self.still_mask.shape[0]][self.still_mask],
                self.get_attribute("scale")[:self.still_mask.shape[0]][self.still_mask],
                self.get_attribute("rotate")[:self.still_mask.shape[0]][self.still_mask],
                self.get_attribute("opacity")[:self.still_mask.shape[0]][self.still_mask],
                self.get_attribute("rgb")[:self.still_mask.shape[0]][self.still_mask],
                self.intr,
                # self.extr,
                self.get_extr(),
                self.bg,
                self.W,
                self.H,
            ]

            return_dict = render.render_multiple(
                input_group,
                ["rgb", "center"]
            )

            still_rgb_np = render.render2img(return_dict["rgb"])
            still_center_np = render.render2img(return_dict["center"])

            # render moving points
            input_group = [
                self.get_attribute("xyz")[:self.still_mask.shape[0]][~self.still_mask],
                self.get_attribute("scale")[:self.still_mask.shape[0]][~self.still_mask],
                self.get_attribute("rotate")[:self.still_mask.shape[0]][~self.still_mask],
                self.get_attribute("opacity")[:self.still_mask.shape[0]][~self.still_mask],
                self.get_attribute("rgb")[:self.still_mask.shape[0]][~self.still_mask],
                self.intr,
                # self.extr,
                self.get_extr(),
                self.bg,
                self.W,
                self.H,
            ]

            return_dict = render.render_multiple(
                input_group,
                ["rgb", "center"]
            )

            move_rgb_np = render.render2img(return_dict["rgb"])
            move_center_np = render.render2img(return_dict["center"])

        if save_imgs:
            os.makedirs(os.path.join(self.dir, "images"), exist_ok=True)
            imageio.imwrite(os.path.join(self.dir, "images", f"img_{ckpt_name}.png"), rendered_rgb_np)
            imageio.imwrite(os.path.join(self.dir, "images", f"img_center_{ckpt_name}.png"), rendered_center_np)
            imageio.imwrite(os.path.join(self.dir, "images", f"img_depth_{ckpt_name}.png"), rendered_depth_map_color_np)
            if hasattr(self, 'still_mask'):
                imageio.imwrite(os.path.join(self.dir, "images", f"img_still_{ckpt_name}.png"), still_rgb_np)
                imageio.imwrite(os.path.join(self.dir, "images", f"img_still_center_{ckpt_name}.png"), still_center_np)
                imageio.imwrite(os.path.join(self.dir, "images", f"img_move_{ckpt_name}.png"), move_rgb_np)
                imageio.imwrite(os.path.join(self.dir, "images", f"img_move_center_{ckpt_name}.png"), move_center_np)
            if hasattr(self, 'move_seg'):
                os.makedirs(os.path.join(self.dir, "images_seg"), exist_ok=True)
                imageio.imwrite(os.path.join(self.dir, "images_seg", f"move_mask_{ckpt_name}.png"), self.move_seg)
            if hasattr(self, 'move_seg_erode'):
                os.makedirs(os.path.join(self.dir, "images_seg"), exist_ok=True)
                imageio.imwrite(os.path.join(self.dir, "images_seg", f"move_mask_erode_{ckpt_name}.png"), self.move_seg_erode)
            if hasattr(self, 'propagate_seg'):
                os.makedirs(os.path.join(self.dir, "images_seg"), exist_ok=True)
                imageio.imwrite(os.path.join(self.dir, "images_seg", f"propagate_mask_{ckpt_name}.png"), self.propagate_seg)

        if save_videos:
            # save them as a video with imageio
            frames_np = np.stack(frames, axis=0)
            imageio.mimwrite(os.path.join(self.dir, "training_rgb.mp4"), frames_np, fps=30)
            frames_center_np = np.stack(frames_center, axis=0)
            imageio.mimwrite(os.path.join(self.dir, "training_center.mp4"), frames_center_np, fps=30)
            frames_depth_np = np.stack(frames_depth, axis=0)
            imageio.mimwrite(os.path.join(self.dir, "training_depth.mp4"), frames_depth_np, fps=30)

        if save_ckpt:
            self.save_checkpoint(ckpt_name=ckpt_name, camera_only=camera_only)

        return frames, frames_center, frames_depth, still_rgb_np, still_center_np, move_rgb_np, move_center_np, self.move_seg

    def eval(self, traj_index=None, line_scale=0.1, point_scale=0.3, alpha=0.5, split_interval=None):
        # line_scale = 0.6
        # point_scale = 2.
        # alpha = 0.6 # changing opacity
        num_traj = len(traj_index)
        # check if this class has a attribute traj_points
        if not hasattr(self, 'traj_xyz'): # the first frame
            self.traj_xyz = self.get_attribute("xyz")[traj_index].to(self.device).float()
            self.traj_scale = torch.ones((num_traj, 3), device=self.device).float()
            # create a no rotation quaternion
            self.traj_rotate = torch.tensor([1, 0, 0, 0], device=self.device).repeat(num_traj, 1).float()

            self.traj_opacity = self._activations_inv["opacity"](0.99*torch.ones((num_traj, 1), device=self.device).float())

            if split_interval is None or num_traj==split_interval:
                traj_rgb = torch.arange(0, 1, 1/num_traj, device=self.device).float().unsqueeze(1)
            else:
                traj_rgb_still = torch.arange(0, 1, 1/split_interval, device=self.device).float().unsqueeze(1)
                traj_rgb_move = torch.arange(0, 1, 1/(num_traj-split_interval), device=self.device).float().unsqueeze(1)
                traj_rgb = torch.cat([traj_rgb_still, traj_rgb_move], dim=0)
            traj_rgb = utils.apply_float_colormap(traj_rgb, colormap="gist_rainbow")
            self.traj_rgb = self._activations_inv["rgb"](traj_rgb)

            self.last_traj_xyz = self.traj_xyz
            self.last_traj_rgb = self.traj_rgb
        else: # the following frames
            current_xyz = self.get_attribute("xyz")[traj_index].to(self.device).float()
            line_xyz, line_rgb = utils.gen_line_set(self.last_traj_xyz, current_xyz, self.last_traj_rgb, device=self.device)
            num_in_line = line_xyz.shape[0]
            self.traj_xyz = torch.cat([self.traj_xyz, line_xyz], dim=0)
            num_total = self.traj_xyz.shape[0]
            # print(num_total)

            self.traj_scale = torch.ones((num_total, 3), device=self.device).float() * 1e-6 # too big will cause memory overflow

            self.traj_rotate = torch.tensor([1, 0, 0, 0], device=self.device).repeat(num_total, 1).float()

            # gradually fade out the opacity
            self.traj_opacity *= alpha
            # traj_opacity = torch.ones((num_in_line, 1), device=self.device)
            traj_opacity = self._activations_inv["opacity"](0.99*torch.ones((num_in_line, 1), device=self.device))
            self.traj_opacity = torch.cat([self.traj_opacity, traj_opacity], dim=0)
            # self.point_opacity = torch.ones((num_traj, 1), device=self.device)

            self.traj_rgb = torch.cat([self.traj_rgb, line_rgb], dim=0)
            # self.point_rgb = self.last_traj_rgb

            self.last_traj_xyz = self.get_attribute("xyz")[traj_index].to(self.device).float()
        input_group = [
            self.get_attribute("xyz"),
            self.get_attribute("scale"),
            self.get_attribute("rotate"),
            self.get_attribute("opacity"),
            self.get_attribute("rgb"),
            # self.intr, self.extr,
            self.intr, self.get_extr(),
            self.bg, self.W, self.H,
        ]

        output_dict = render.render_multiple(
            input_group,
            ["rgb", "center", "depth_map_color"]
        )

        out_img = render.render2img(output_dict["rgb"])
        out_img_center = render.render2img(output_dict["center"])
        out_img_depth = render.render2img(output_dict["depth_map_color"])
        
        traj_group = [
            self.traj_xyz,
            self.traj_scale,
            self.traj_rotate,
            self.traj_opacity,
            self.traj_rgb,
            # self.intr, self.extr,
            self.intr, self.get_extr(),
            self.bg, self.W, self.H,
        ]

        out_traj = render.render_traj(
            traj_group, num_traj, line_scale, point_scale
        )

        out_img_traj = render.render2img(out_traj)

        # screen blending mode
        arr1 = np.array(out_img) / 255.0
        arr2 = np.array(out_img_traj) / 255.0

        # 应用滤色混色模式
        result = 1 - (1 - arr1) * (1 - arr2)

        # 将结果转换回 0-255 范围并转换为整数
        out_img_traj_upon = (result * 255).astype(np.uint8)
        # import pdb
        # pdb.set_trace()


        return out_img, out_img_center, out_img_depth, out_img_traj, out_img_traj_upon

    def render(self, xyz, scale, rotate, opacity, rgb):
        input_group = [
            xyz,
            scale,
            rotate,
            opacity,
            rgb,
            self.intr,
            self.extr,
            # self.extr,
            self.get_extr(),
            self.bg,
            self.W,
            self.H,
        ]

        output_dict = render.render_multiple(
            input_group,
            ["rgb", "center", "depth_map_color"]
        )
        
        out_img = render.render2img(output_dict["rgb"])
        out_img_center = render.render2img(output_dict["center"])
        out_img_depth = render.render2img(output_dict["depth_map_color"])
        
        return out_img, out_img_center, out_img_depth


    def _prune_optimizer(self, mask, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] not in tensors_dict.keys():
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True))).contiguous()
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True)).contiguous()
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask, tensors_dict=None):
        if tensors_dict is None:
            tensors_dict = {
                "means": self.means,
                "scales": self.scales,
                "quats": self.quats,
                "rgbs": self.rgbs,
                "opacities": self.opacities,
            }
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask, tensors_dict)
        self.means = optimizable_tensors["means"]
        self.scales = optimizable_tensors["scales"]
        self.quats = optimizable_tensors["quats"]
        self.rgbs = optimizable_tensors["rgbs"]
        self.opacities = optimizable_tensors["opacities"]

    def densify_by_pixels(self, error_map, error_threshold=1e-3, percent=0.1, mask=None, ):
        num_before = self.get_attribute("xyz").shape[0]
        error_map = error_map.detach().cpu().numpy()
        # print(error_map.shape)
        print("\n\t[densify] error_map's max, min, mean:", error_map.max().item(), error_map.min().item(), error_map.mean().item())
        # add uniform magnitude, to avoid zero probability
        error_map = error_map + np.nanmin(error_map[error_map>0])

        if mask is None:
            mask = (error_map > error_threshold).squeeze()
            print(f"\t[densify] No specify mask, generating it by error threshold: {error_threshold}")
        else:
            mask = mask.detach().cpu().numpy().squeeze()

        # transfer to bool type, then apply the mask
        mask = mask > 0
        error_map = error_map*mask[:,:error_map.shape[1]]

        mask_ratio = np.sum(mask) / np.size(mask)

        # make positive elements to be 1, and negative elements to be 0
        print("\t[densify] mask_ratio:", mask_ratio)
        probability_distribution = error_map / np.sum(error_map)
        densify_num = int(self.num_points * mask_ratio * percent)

        if densify_num > 0:
            # sample points from the probability distribution
            sampled_points = np.random.choice(a=np.arange(self.H*self.W), size=densify_num, p=probability_distribution.flatten())

            # convert the sampled points to coordinates on the image
            sampled_coordinates = np.unravel_index(sampled_points, (self.H, self.W))

            xys = np.array(sampled_coordinates).T[:,::-1].copy()
            depths = self.gt_depth[sampled_coordinates]
            scales = 1 / probability_distribution[sampled_coordinates]
            scales = np.ones_like(scales) * (1/self.num_points)

            scales = scales * (depths.cpu().numpy()/depths.cpu().numpy().min()).squeeze()
            rgbs = self.gt_image[sampled_coordinates]


            xys = torch.from_numpy(xys).to(self.device).float()
            depths = depths.to(self.device).float()

            new_xyz = geometry.pix2world(xys, depths, self.intr, self.get_extr())
            scales = torch.from_numpy(scales).float().unsqueeze(1).repeat(1, 3).to(self.device)


            new_scale = self._activations_inv["scale"](scales)
            rgbs = rgbs.contiguous()
            eps = 1e-15  # avoid logit function input 0 or 1
            rgbs = torch.clamp(rgbs, min=eps, max=1-eps)
            # calculate the inverse of sigmoid function, i.e., logit function
            new_rgb = self._activations_inv["rgb"](rgbs)
            new_rotate = torch.tensor([1, 0, 0, 0], device=self.device).repeat(new_xyz.shape[0], 1).float()
            new_opacity = 0.99*torch.ones((new_xyz.shape[0], 1), device=self.device).float()
            new_opacity = self._activations_inv["opacity"](new_opacity)

            self.densification_postfix(new_xyz, new_scale, new_rotate, new_opacity, new_rgb)

        num_after = self.get_attribute("xyz").shape[0]
        print(f"\t[densify / split] number of gaussians: {num_before} -> {num_after}")

    def densification_postfix(self, new_xyz, new_scale, new_rotate, new_opacity, new_rgb):
        self._attributes["xyz"] = torch.cat((self._attributes["xyz"], new_xyz), dim=0)
        self._attributes["scale"] = torch.cat((self._attributes["scale"], new_scale), dim=0)
        self._attributes["rotate"] = torch.cat((self._attributes["rotate"], new_rotate), dim=0)
        self._attributes["opacity"] = torch.cat((self._attributes["opacity"], new_opacity), dim=0)
        self._attributes["rgb"] = torch.cat((self._attributes["rgb"], new_rgb), dim=0)

        for attribute_name in self._attributes.keys():
            self._attributes[attribute_name] = nn.Parameter(self._attributes[attribute_name]).requires_grad_(True)

        self.optimizer = torch.optim.Adam(list(self._attributes.values()), lr=self.lr)

    def project_points(self, points):
        # return msplat.project_point(points, self.intr, self.extr, self.W, self.H)
        return msplat.project_point(points, self.intr, self.get_extr(), self.W, self.H)
    