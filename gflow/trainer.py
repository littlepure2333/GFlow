import math
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import imageio
import msplat
import utils
from init import complex_texture_sampling
from datetime import datetime
import os
import render
import cv2
import matplotlib

class SimpleGaussian:
    def __init__(self, gt_image, gt_depth=None, num_points=100000, background="black", depth_scale=1.):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gt_image = gt_image.to(self.device)
        self.gt_depth = gt_depth.to(self.device) if gt_depth is not None else None
        self.num_points = num_points
        self.depth_scale = depth_scale

        H, W, C = gt_image.shape 
        self.H = H
        self.W = W
        self.bg = 0. if background == "black" else 1.
        fov = math.pi / 2.0
        fx = 0.5 * float(W) / math.tan(0.5 * fov)
        fy = 0.5 * float(H) / math.tan(0.5 * fov)
        self.intr = torch.Tensor([fx, fy, float(W) / 2, float(H) / 2]).cuda().float()
        intr_matrix = torch.Tensor([[fx,  0.0, float(W) / 2],
                                    [0.0, fy,  float(H) / 2],
                                    [0.0, 0.0, 1.0         ]]).cuda().float()
        self.t3 = t3 = 2.5 # translation along z-axis
        self.extr = torch.Tensor([[1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, t3 ]]).cuda().float()
        
        N = int(num_points)
        self._attributes = {
            "xyz":      torch.rand((N, 3), dtype=torch.float32).cuda() * 2 - 1,
            "scale":    torch.rand((N, 3), dtype=torch.float32).cuda(),
            "rotate":   torch.rand((N, 4), dtype=torch.float32).cuda(),
            "opacity":  torch.rand((N, 1), dtype=torch.float32).cuda(),
            "rgb":      torch.rand((N, 3), dtype=torch.float32).cuda()
        }
        
        self._activations = {
            "scale": lambda x: torch.abs(x) + 1e-8,
            "rotate": torch.nn.functional.normalize,
            "opacity": torch.sigmoid,
            "rgb": torch.sigmoid
        }
        
        # the inverse of the activation functions
        self._activations_inv = {
            "scale": lambda x: torch.abs(x),
            "rotate": torch.nn.functional.normalize,
            "opacity": torch.logit,
            "rgb": torch.logit
        }

        # if gt_image is not None:
        #     self.init_gaussians_from_image(gt_image=gt_image, gt_depth=gt_depth, num_points=num_points, t3=t3)
        
        # Get current date and time as string
        now = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

        # Create a directory with the current date and time as its name
        directory = f"logs/{now}"
        os.makedirs(directory, exist_ok=True)
        # make a new directory, always soft link to the current run
        abs_directory = os.path.abspath(directory)
        os.system(f"rm logs/0_latest")
        os.system(f"ln -s {abs_directory} logs/0_latest")
        self.dir = directory
    
    def add_optimizer(self, lr=1e-2):
        self.lr = lr
        for attribute_name in self._attributes.keys():
            self._attributes[attribute_name] = nn.Parameter(self._attributes[attribute_name]).requires_grad_(True)
        
        self.optimizer = torch.optim.Adam(list(self._attributes.values()), lr=lr)
    
    def add_attribute(self, name, value, clear_grad=True):
        if clear_grad:
            self.optimizer.zero_grad()
        self._attributes[name] = nn.Parameter(value).requires_grad_(True)
        self.optimizer = torch.optim.Adam(list(self._attributes.values()), lr=self.lr)

    def set_gt_image(self, gt_image):
        self.gt_image = gt_image.to(self.device)

    def set_gt_depth(self, gt_depth):
        self.gt_depth = gt_depth.to(self.device)

    def set_gt_flow(self, gt_flow):
        self.gt_flow = gt_flow.to(self.device)
    
    def init_gaussians_from_image(self, gt_image, gt_depth=None, num_points=None, t3=1.):
        # gt_image (H, W, C) in [0, 1]
        if num_points is None:
            num_points = self.num_points
        # gt_image = resize_by_divide(gt_image, 16)
        xys, depths, scales, rgbs, gt_depth = complex_texture_sampling(gt_image, gt_depth, num_points=num_points, device=self.device)
        '''
        for depth, 1 is near, 0 is far!!!
        '''
        xys = torch.from_numpy(xys).to(self.device).float() * t3
        self.gt_depth = gt_depth.to(self.device).float() * self.depth_scale
        # print(xys.shape)
        # depths = depths.to(self.device).float() * self.depth_scale
        depths = torch.ones_like(depths).to(self.device).float() * 0.
        xyd = torch.concatenate((xys, depths), dim=1)
        self._attributes["xyz"] = xyd
        # self._attributes["xyz"] = utils.camera2world(xyd, self.intr, self.extr)
        fn = lambda x: np.power(x, 0.6)
        # fn = lambda x: np.sqrt(x)
        # fn = lambda x: x
        self.fn = fn
        scales = self.fn(scales) * (depths.cpu().numpy()+0.5).squeeze() # * [0.5,1.5] decrease near's scales, increase far's scales
        scales = torch.from_numpy(scales).float().unsqueeze(1).repeat(1, 3).to(self.device)
        self._attributes["scale"] = self._activations_inv["scale"](scales)

        rgbs = torch.from_numpy(rgbs).float().contiguous().to(self.device)
        eps = 1e-15  # avoid logit function input 0 or 1
        rgbs = torch.clamp(rgbs, min=eps, max=1-eps)
        # calculate the inverse of sigmoid function, i.e., logit function
        self._attributes["rgb"] = self._activations_inv["rgb"](rgbs)
        
    def step(self):
        self.optimizer.zero_grad()
        self.optimizer.step()
    
    def get_attribute(self, name):
        try:
            if name in self._activations.keys() and self._activations[name] is not None:
                return self._activations[name](self._attributes[name])
            else:
                return self._attributes[name]
        except:
            raise ValueError(f"Attribute or activation for {name} is not VALID!")
    
    def save_checkpoint(self, ckpt_name=None):
        checkpoint = self._attributes
        if ckpt_name is None:
            ckpt_name = 'ckpt'
        # make directory
        os.makedirs(os.path.join(self.dir, "ckpt"), exist_ok=True)
        self.checkpoint_path = os.path.join(self.dir, "ckpt", f"{ckpt_name}.tar")
        torch.save(checkpoint, self.checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self._attributes = checkpoint
    
    def train(self, iterations=500, lr=1e-2, 
              lambda_depth=0., lambda_flow=0., lambda_var=0.,
              save_imgs=False, save_videos=False, save_ckpt=False,
              ckpt_name="ckpt", densify_interval=500, densify_times=1, 
              grad_threshold=5e-3, mask=None):
        frames = []
        frames_depth = []
        frames_center = []
        progress_bar = tqdm(range(1, iterations), desc="Training")
        mse_loss = nn.SmoothL1Loss()
        self.add_optimizer(lr)
        self.reset_densification_stats()
        
        for iteration in range(0, iterations):
            loss = 0.

            input_group = [
                self.get_attribute("xyz"),
                self.get_attribute("scale"),
                self.get_attribute("rotate"),
                self.get_attribute("opacity"),
                self.get_attribute("rgb"),
                self.intr,
                self.extr,
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
            
            loss_rgb = mse_loss(rendered_rgb.permute(1, 2, 0), self.gt_image)
            loss += loss_rgb
            progress_dict = {"rgb": f"{loss_rgb.item():.6f}"}

            rendered_depth_map = rendered_depth_map.permute(1, 2, 0) - self.t3
            # import pdb
            # pdb.set_trace()
            if lambda_depth > 0:
                loss_depth = mse_loss(rendered_depth_map, self.gt_depth)
                loss += lambda_depth * loss_depth
                progress_dict["depth"] = f"{loss_depth.item():.6f}"

            if lambda_var: # penaliza the scales with large variance
                loss_var = torch.mean(torch.var(self.get_attribute("xyz"), dim=0))
                    # print("\t[loss] loss_var", loss_var.item())
                loss += lambda_var * loss_var
                progress_dict["depth"] = f"{loss_var.item():.6f}"


            if lambda_flow: # ensure local flow consistency
                num_last_uv = self.last_uv.shape[0]

                # select uv that uv within the a HxW range
                within_mask = (uv[:num_last_uv,0] > 0) & (uv[:num_last_uv,0] < self.W-1) & (uv[:num_last_uv,1] > 0) & (uv[:num_last_uv,1] < self.H-1)

                pred_flow = uv[:num_last_uv][within_mask] - self.last_uv[within_mask]
                # y_coords = torch.clamp(uv.detach()[:,1].long(), 0, self.H-1)
                # x_coords = torch.clamp(uv.detach()[:,0].long(), 0, self.W-1)
                y_coords = uv.detach()[:num_last_uv][within_mask][:,1].long()
                x_coords = uv.detach()[:num_last_uv][within_mask][:,0].long()
                # print(y_coords.max(), y_coords.min())
                # print(x_coords.max(), x_coords.min())

                gt_flow = self.gt_flow[y_coords, x_coords]

                # print(pred_flow.shape, gt_flow.shape)

                loss_flow = mse_loss(pred_flow, gt_flow)
                loss += lambda_flow * loss_flow
                progress_dict["flow"] = f"{loss_flow.item():.6f}"

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            progress_dict["total"] = loss.item()
            progress_bar.set_postfix(progress_dict)
            # progress_bar.set_description(progress_dict)
            
            # progress_bar.set_postfix({"Loss": f"{loss:.{7}f}"})
            progress_bar.update(1)

            if densify_interval:
                grad_dict = {
                    "xyz": self._attributes["xyz"],
                    # "scales": self.scales,
                    # "quats": self.quats,
                }
                self.add_densification_stats(grad_dict)


            if densify_interval and (iteration+1) % densify_interval == 0 and (iteration+1) // densify_interval <= densify_times:
                grad_norm = self.acquire_densification_stats()
                print("\n\t[densify] grad_norm's max, min:", grad_norm.max().item(), grad_norm.min().item())
                print("\t[densify] grad_norm's mean, threshold:", grad_norm.mean().item(), grad_threshold)
                scene_extent = (self.H+self.W)/2
                self.densify_and_clone(grad_norm, grad_threshold, scene_extent, 0.00005, N=2, mask=mask)
                self.densify_and_split(grad_norm, grad_threshold*1, scene_extent, 0.0001, N=None, mask=mask, there=True)
                # self.densify_and_split(grad_norm, grad_threshold*1, scene_extent, 0.0001, N=2, mask=None, there=True)
                torch.cuda.empty_cache()
                self.reset_densification_stats()


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
        self.last_uv = uv.detach()
        self.last_depth = depth.detach()

        if save_imgs:
            os.makedirs(os.path.join(self.dir, "images"), exist_ok=True)
            imageio.imwrite(os.path.join(self.dir, "images", f"img_{ckpt_name}.png"), rendered_rgb_np)
            imageio.imwrite(os.path.join(self.dir, "images", f"img_center_{ckpt_name}.png"), rendered_center_np)
            imageio.imwrite(os.path.join(self.dir, "images", f"img_depth_{ckpt_name}.png"), rendered_depth_map_color_np)

        if save_videos:
            # save them as a video with imageio
            frames_np = np.stack(frames, axis=0)
            imageio.mimwrite(os.path.join(self.dir, "training_rgb.mp4"), frames_np, fps=30)
            frames_center_np = np.stack(frames_center, axis=0)
            imageio.mimwrite(os.path.join(self.dir, "training_center.mp4"), frames_center_np, fps=30)
            frames_depth_np = np.stack(frames_depth, axis=0)
            imageio.mimwrite(os.path.join(self.dir, "training_depth.mp4"), frames_depth_np, fps=30)

        if save_ckpt:
            self.save_checkpoint(ckpt_name=ckpt_name)

        return frames, frames_center, frames_depth
    
    def eval(self, traj_index=None, line_scale=0.6, point_scale=2., alpha=0.5):
        # line_scale = 0.6
        # point_scale = 2.
        # alpha = 0.6 # changing opacity
        num_traj = len(traj_index)
        # check if this class has a attribute traj_points
        if not hasattr(self, 'traj_xyz'): # the first frame
            self.traj_xyz = self.get_attribute("xyz")[traj_index].to(self.device).float()
            self.traj_scale = torch.ones((num_traj, 3), device=self.device).float() * point_scale
            # create a no rotation quaternion
            self.traj_rotate = torch.tensor([1, 0, 0, 0], device=self.device).repeat(num_traj, 1).float()

            self.traj_opacity = torch.ones((num_traj, 1), device=self.device).float()

            traj_rgb = torch.arange(0, 1, 1/num_traj, device=self.device).float().unsqueeze(1)
            traj_rgb = utils.apply_float_colormap(traj_rgb, colormap="turbo")
            self.traj_rgb = self._activations_inv["rgb"](traj_rgb)

            self.last_traj_xyz = self.traj_xyz
            self.last_traj_rgb = self.traj_rgb
        else: # the following frames
            current_xyz = self.get_attribute("xyz")[traj_index].to(self.device).float()
            line_xyz, line_rgb = utils.gen_line_set(self.last_traj_xyz, current_xyz, self.last_traj_rgb, device=self.device)
            num_in_line = line_xyz.shape[0]
            self.traj_xyz = torch.cat([self.traj_xyz, line_xyz], dim=0)
            num_total = self.traj_xyz.shape[0]

            self.traj_scale = torch.ones((num_total, 3), device=self.device).float() * point_scale

            self.traj_rotate = torch.ones((num_total, 4), device=self.device).float()

            # gradually fade out the opacity
            self.traj_opacity *= alpha
            traj_opacity = torch.ones((num_in_line, 1), device=self.device)
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
            self.intr, self.extr,
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
            self.intr, self.extr,
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

        # self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        # self.denom = self.denom[valid_points_mask]
        # self.max_radii2D = self.max_radii2D[valid_points_mask]

    
    def densify_and_split(self, grads, grad_threshold, scene_extent=None, percent_dense=0.01, N:int=2, mask=None, there=False):
        n_init_points = self.get_attribute("xyz").shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device=self.device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        print("\t[densify / split] max scale, min scale:", self.get_attribute("scale").max().item(), self.get_attribute("scale").min().item())
        print("\t[densify / split] scale threshold:", percent_dense*scene_extent)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_attribute("scale"), dim=1).values > percent_dense*scene_extent)
        
        if mask is not None:
            (uv, depth) = msplat.project_point(
                self.get_attribute("xyz"), 
                self.intr, self.extr, self.W, self.H
            )
            # find which points of xys are inside the mask, mask is a tensor of shape [H, W], xys is a tensor of shape [num_points, 2]
            mask = mask.to(self.device).squeeze()
            H, W = mask.shape
            # need to consider the value of xys may not in the range of [0, W] and [0, H]
            in_range_mask = torch.where((uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H), True, False)
            selected_pts_mask = torch.logical_and(selected_pts_mask, in_range_mask)
            # print("in_range_mask.shape", in_range_mask.shape)
            masked_indices = torch.where(mask[uv[:, 1].clamp(0, H-1).long(), uv[:, 0].clamp(0, W-1).long()]>0, True, False)
            selected_pts_mask = torch.logical_and(selected_pts_mask, masked_indices)
            # print("masked_indices.shape", len(masked_indices))
            if N is None:
                # calculate the true ratio of points inside the mask
                mask_ratio = mask.sum() / (H * W)
                # print("\t[densify] mask ratio is", mask_ratio)
                add_num = mask_ratio * len(self.get_attribute("xyz"))
                # print("\t[densify] len(self.means) is", len(self.means))
                # round up N using math.ceil
                add_num = math.ceil(add_num)
                # print("\t[densify] add_num is", add_num)
                # print("\t[densify] selected_pts_mask.sum() is", selected_pts_mask.sum())
                N = max(2, add_num // selected_pts_mask.sum())
                print("\t[densify / split] split N is", N)
        else:
            N = 1

        if there:
            new_xyz = self._attributes["xyz"][selected_pts_mask].repeat(N, 1)
            new_scale = self._attributes["scale"][selected_pts_mask].repeat(N, 1) / 1.6
            new_rotate = self._attributes["rotate"][selected_pts_mask].repeat(N, 1)
            new_opacity = self._attributes["opacity"][selected_pts_mask].repeat(N, 1)
            new_rgb = self._attributes["rgb"][selected_pts_mask].repeat(N, 1)

            num_before = self.get_attribute("xyz").shape[0]
            self.densification_postfix(new_xyz, new_scale, new_rotate, new_opacity, new_rgb)
            num_after = self.get_attribute("xyz").shape[0]
            print(f"\t[densify / split] number of gaussians: {num_before} -> {num_after}")

        else:
            stds = torch.abs(self.get_attribute("scale")[selected_pts_mask]).repeat(N,1)
            means = torch.zeros((stds.size(0), 3), device=self.device)
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(self.quats[selected_pts_mask]).repeat(N,1,1)
            new_means = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.means[selected_pts_mask].repeat(N, 1)
            new_scales = self.scales_inv(self.get_attribute("scale")[selected_pts_mask].repeat(N,1) / 1.6)
            new_quats = self.quats[selected_pts_mask].repeat(N,1)
            new_rgbs = self.rgbs[selected_pts_mask].repeat(N,1)
            new_opacities = self.opacities[selected_pts_mask].repeat(N,1)
            # new_opacities = torch.ones_like(self.opacities[selected_pts_mask].repeat(N,1))
            print("\t[densify] before split, the number of gaussians is", self.means.shape[0])
            self.densification_postfix(new_xyz, new_scale, new_rotate, new_opacity, new_rgb)
            prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device=self.device, dtype=bool)))
            self.prune_points(prune_filter, {"means": self.means, "scales": self.scales, "quats": self.quats, "rgbs": self.rgbs, "opacities": self.opacities})
            print("\t[densify] after split, the number of gaussians is", self.means.shape[0])
            

    def densify_and_clone(self, grads, grad_threshold, scene_extent=None, percent_dense=0.01, N:int=1, mask=None):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_attribute("scale"), dim=1).values <= percent_dense*scene_extent)
        print("\t[densify / clone] max scale, min scale:", self.get_attribute("scale").max().item(), self.get_attribute("scale").min().item())
        print("\t[densify / clone] scale threshold:", percent_dense*scene_extent)

        if mask is not None:
            (uv, depth) = msplat.project_point(
                self.get_attribute("xyz"), 
                self.intr, self.extr, self.W, self.H
            )
            # find which points of xys are inside the mask, mask is a tensor of shape [H, W], xys is a tensor of shape [num_points, 2]
            mask = mask.to(self.device).squeeze()
            H, W = mask.shape
            # need to consider the value of xys may not in the range of [0, W] and [0, H]
            in_range_mask = torch.where((uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H), True, False)
            selected_pts_mask = torch.logical_and(selected_pts_mask, in_range_mask)
            # print("in_range_mask.shape", in_range_mask.shape)
            masked_indices = torch.where(mask[uv[:, 1].clamp(0, H-1).long(), uv[:, 0].clamp(0, W-1).long()]>0, True, False)
            selected_pts_mask = torch.logical_and(selected_pts_mask, masked_indices)
            # print("masked_indices.shape", len(masked_indices))
        
        new_xyz = self._attributes["xyz"][selected_pts_mask].repeat(N, 1)
        new_scale = self._attributes["scale"][selected_pts_mask].repeat(N, 1)
        new_rotate = self._attributes["rotate"][selected_pts_mask].repeat(N, 1)
        new_opacity = self._attributes["opacity"][selected_pts_mask].repeat(N, 1)
        new_rgb = self._attributes["rgb"][selected_pts_mask].repeat(N, 1)

        num_before = self.get_attribute("xyz").shape[0]
        # reset optimizer
        self.densification_postfix(new_xyz, new_scale, new_rotate, new_opacity, new_rgb)
        num_after = self.get_attribute("xyz").shape[0]
        print(f"\t[densify / clone] number of gaussians: {num_before} -> {num_after}")

    def densification_postfix(self, new_xyz, new_scale, new_rotate, new_opacity, new_rgb):

        self._attributes["xyz"] = torch.cat((self._attributes["xyz"], new_xyz), dim=0)
        self._attributes["scale"] = torch.cat((self._attributes["scale"], new_scale), dim=0)
        self._attributes["rotate"] = torch.cat((self._attributes["rotate"], new_rotate), dim=0)
        self._attributes["opacity"] = torch.cat((self._attributes["opacity"], new_opacity), dim=0)
        self._attributes["rgb"] = torch.cat((self._attributes["rgb"], new_rgb), dim=0)

        for attribute_name in self._attributes.keys():
            self._attributes[attribute_name] = nn.Parameter(self._attributes[attribute_name]).requires_grad_(True)

        self.optimizer = torch.optim.Adam(list(self._attributes.values()), lr=self.lr)

        # self.opacities.requires_grad_(True)
        # optimizable_tensors = self.cat_tensors_to_optimizer(d)
        # self.means = optimizable_tensors["means"]
        # self.scales = optimizable_tensors["scales"]
        # self.rgbs = optimizable_tensors["rgbs"]
        # self.opacities = optimizable_tensors["opacities"]
        # self.quats = optimizable_tensors["quats"]
        # self.opacities.requires_grad_(False)

        # self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        # self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        # self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.device)

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            # print(group.keys())
            # print(group["name"])
            if group["name"] not in tensors_dict.keys():
                continue
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0).contiguous()
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0).contiguous()

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)).contiguous()
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)).contiguous()
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
    
    def add_densification_stats(self, tensor_dict):
        for name, tensor in tensor_dict.items():
            if name not in self.grad_accum:
                self.grad_accum[name] = torch.norm(tensor.grad, dim=-1, keepdim=True)
                self.denom[name] = 1
            else:
                self.grad_accum[name] += torch.norm(tensor.grad, dim=-1, keepdim=True)
                self.denom[name] += 1
   
    def reset_densification_stats(self):
        self.grad_accum = {}
        self.denom = {}
    
    def acquire_densification_stats(self):
        grad_sum = 0
        for name, tensor in self.grad_accum.items():
            grad_sum += self.grad_accum[name] / self.denom[name]
        return grad_sum
        

if __name__ == "__main__":
    seed = 123
    torch.manual_seed(seed)
    
    # img = imageio.imread("./data/stanford-bunny.jpg") # H, W, C
    img = imageio.imread("./data/face.png") # H, W, C
    img = img.astype(np.float32) / 255.0
    gt = torch.from_numpy(img).cuda().permute(2, 0, 1) # C, H, W
    
    C, H, W = gt.shape 
    
    gaussians = SimpleGaussian(gt_image=gt.permute(1, 2, 0), num_points=10000, depth_scale=1.) # H, W, C
    gaussians.train(iterations=1000, lr=1e-2, lambda_depth=0.1, 
                    save_imgs=True, save_videos=True, save_ckpt=True, 
                    ckpt_name="ckpt", 
                    densify_interval=500, densify_times=1, grad_threshold=5e-6, mask=None)
    