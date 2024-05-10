import math
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import imageio
import msplat
import utils
from utils import read_depth
from init import complex_texture_sampling, image_sampling
from datetime import datetime
import os
import render
import cv2
import matplotlib
import geometry
from io import BytesIO
import json
from plyfile import PlyData, PlyElement
from sklearn.cluster import KMeans, AgglomerativeClustering

class SimpleGaussian:
    def __init__(self, gt_image, gt_depth=None, num_points=100000, background="black", depth_scale=1., depth_offset=1.):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gt_image = gt_image.to(self.device)
        self.gt_depth = gt_depth.to(self.device) if gt_depth is not None else None
        self.num_points = num_points
        self.depth_scale = depth_scale
        self.depth_offset = depth_offset
        # self.clustering = AgglomerativeClustering(
        #     # distance_threshold=0.1,
        #     linkage='ward', 
        #     n_clusters=2,
        #     metric='euclidean')
        self.clustering = KMeans(n_clusters=2, random_state=0)

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
        # self.t3 = t3 = 2.5 # translation along z-axis
        self.extr = torch.Tensor([[1.0, 0.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0, 0.0],
                                  [0.0, 0.0, 1.0, 0.00 ]]).cuda().float()
        
        N = int(num_points)
        self._attributes = {
            "xyz":      torch.rand((N, 3), dtype=torch.float32).cuda() * 2 - 1,
            "scale":    torch.rand((N, 3), dtype=torch.float32).cuda(),
            "rotate":   torch.rand((N, 4), dtype=torch.float32).cuda(),
            "opacity":  torch.rand((N, 1), dtype=torch.float32).cuda(),
            "rgb":      torch.rand((N, 3), dtype=torch.float32).cuda()
        }

        def _2dgs(x):
            x = (torch.abs(x) + 1e-8).repeat(1, 2)
            # add the third channel to be zero
            x = torch.cat((x, torch.zeros_like(x[:, :1])), dim=1)
            return x
        
        def _isotropic(x):
            return (torch.abs(x) + 1e-8).repeat(1, 3)
        
        self._activations = {
            "scale": lambda x: torch.abs(x) + 1e-8,
            # "scale": _isotropic,
            "rotate": torch.nn.functional.normalize,
            "opacity": torch.sigmoid,
            "rgb": torch.sigmoid
        }
        
        # the inverse of the activation functions
        self._activations_inv = {
            "scale": lambda x: torch.abs(x),
            # "scale": lambda x: torch.abs(x)[..., [0]],
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
        os.makedirs("logs/0_latest", exist_ok=True)
        # make a new directory, always soft link to the current run
        abs_directory = os.path.abspath(directory)
        os.system(f"rm logs/0_latest/*")
        os.system(f"ln -s {abs_directory} logs/0_latest/{now}")
        self.dir = directory
    
    def add_optimizer(self, lr=1e-2, lr_camera=0., exclude_key="rotate"):
        self.lr = lr
        self.lr_camera = lr_camera
        for attribute_name in self._attributes.keys():
            if attribute_name != exclude_key:
                self._attributes[attribute_name] = nn.Parameter(self._attributes[attribute_name]).requires_grad_(True)
        self.extr = nn.Parameter(self.extr).requires_grad_(True)

        # assign different lr to attributes and extr
        optim_params=[
            {"params": self._attributes.values(), "lr": lr, "name": "attributes"},
            {"params": self.extr, "lr": lr_camera, "name": "extr"},
        ]
        
        self.optimizer = torch.optim.Adam(optim_params)
    
    def add_attribute(self, name, value, clear_grad=True):
        if clear_grad:
            self.optimizer.zero_grad()
        self._attributes[name] = nn.Parameter(value).requires_grad_(True)
        self.optimizer = torch.optim.Adam(list(self._attributes.values()), lr=self.lr)

    def set_gt_image(self, gt_image):
        self.gt_image = gt_image.to(self.device)

    def set_gt_depth(self, gt_depth):
        self.gt_depth = gt_depth.to(self.device) * self.depth_scale

    def set_gt_flow(self, gt_flow):
        self.gt_flow = gt_flow.to(self.device)

    def load_camera(self, focal=None, pp=None, extr=None):
        if (focal is not None) and (pp is not None):
            self.intr = torch.tensor([focal, focal, pp[0], pp[1]], dtype=torch.float32).to(self.device)
        if extr is not None:
            self.extr = torch.tensor(extr, dtype=torch.float32).to(self.device)
        print("[camera] intr: ", self.intr)
        print("[camera] extr: ", self.extr)
    
    def init_gaussians_from_image(self, gt_image, gt_depth=None, num_points=None):
        # gt_image (H, W, C) in [0, 1]
        if num_points is None:
            num_points = self.num_points
        # gt_image = resize_by_divide(gt_image, 16)
        xys, depths, scales, rgbs, gt_depth = complex_texture_sampling(gt_image, gt_depth, num_points=num_points, device=self.device)
        # _ = image_sampling(gt_image, gt_depth, num_points=num_points, device=self.device)
        """depths: 0 is near, 1 is far"""
        xys = torch.from_numpy(xys).to(self.device).float()
        # depths = depths.to(self.device).float() * self.depth_scale + self.depth_offset
        # self.gt_depth = gt_depth.to(self.device).float() * self.depth_scale + self.depth_offset
        self.depth_scale = self.intr[0] / 20
        depths = depths.to(self.device).float() * self.depth_scale
        self.gt_depth = gt_depth.to(self.device).float() * self.depth_scale
        # print(self.gt_depth.shape)
        # print(self.gt_depth.max())
        # print(xys.shape)
        # depths = torch.ones_like(depths).to(self.device).float() * 1.
        # xyd = torch.concatenate((xys, depths), dim=1)
        # self._attributes["xyz"] = xyd

        # using dust3r outputs

        # self._attributes["xyz"]= geometry.depth_to_pts3d(depths, xys, self.intr[:2], self.intr[2:])
        # # self._attributes["xyz"][:,1] = self._attributes["xyz"][:,1] - 0.01
        # # self._attributes["xyz"][:,2] = self._attributes["xyz"][:,2] + 0.1
        # print(self._attributes["xyz"].shape)
        # print(self._attributes["xyz"][:,0].min(), self._attributes["xyz"][:,0].max())
        # print(self._attributes["xyz"][:,1].min(), self._attributes["xyz"][:,1].max())
        # print(self._attributes["xyz"][:,-1].min(), self._attributes["xyz"][:,-1].max())

        self._attributes["xyz"] = geometry.pix2world(xys, depths, self.intr, self.extr)
        # self._attributes["xyz"][:,1] = self._attributes["xyz"][:,1] - 0.02
        # self._attributes["xyz"][:,2] = self._attributes["xyz"][:,2] + 0.025
        # print(self._attributes["xyz"].shape)
        print("[init] x range: ", self._attributes["xyz"][:,0].min().item(), self._attributes["xyz"][:,0].max().item())
        print("[init] y range: ", self._attributes["xyz"][:,1].min().item(), self._attributes["xyz"][:,1].max().item())
        print("[init] z range: ", self._attributes["xyz"][:,2].min().item(), self._attributes["xyz"][:,2].max().item())


        # fn = lambda x: np.power(x, 0.6)
        # fn = lambda x: np.sqrt(x)
        # fn = lambda x: x
        # self.fn = fn
        # scales = self.fn(scales)
        scales = scales * (depths/depths.min()).squeeze().cpu().numpy()
        scales = torch.from_numpy(scales).float().unsqueeze(1).repeat(1, 3).to(self.device)
        # scales = torch.clamp(scales, max=1e-2)
        self._attributes["scale"] = self._activations_inv["scale"](scales)

        rgbs = torch.from_numpy(rgbs).float().contiguous().to(self.device)
        eps = 1e-15  # avoid logit function input 0 or 1
        rgbs = torch.clamp(rgbs, min=eps, max=1-eps)
        # calculate the inverse of sigmoid function, i.e., logit function
        self._attributes["rgb"] = self._activations_inv["rgb"](rgbs)

        # opacity = torch.ones((num_points, 1), device=self.device).float() - eps
        # self._attributes["opacity"] = self._activations_inv["opacity"](opacity)
        
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
    
    def get_splat_buffer(self):
        sorted_indices = np.argsort(-np.prod(self.get_attribute("scale").detach().cpu().numpy(), axis=1) / (1 + np.exp(-self.get_attribute("opacity").detach().cpu().T.numpy())))
        buffer = BytesIO()
        for idx in sorted_indices[0]:
            position = self.get_attribute("xyz")[idx].detach().cpu().numpy()
            scales = self.get_attribute("scale")[idx].detach().cpu().numpy()
            rot = self.get_attribute("rotate")[idx].detach().cpu().numpy()
            color = self.get_attribute("rgb")[idx].detach().cpu().numpy()
            color = np.append(color, 1 / (1 + np.exp(-self.get_attribute("opacity")[idx].detach().cpu().numpy())))
            buffer.write(position.astype(np.float32).tobytes())
            buffer.write(scales.astype(np.float32).tobytes())
            buffer.write((color * 255).clip(0, 255).astype(np.uint8).tobytes())
            buffer.write((rot * 128 + 128).clip(0, 255).astype(np.uint8).tobytes())
        return buffer.getvalue()

    def get_ply_elements(self):
        dtype_full = [(attribute, 'f4') for attribute in utils.construct_list_of_attributes()]
        elements = np.empty(self.get_attribute("xyz").shape[0], dtype=dtype_full)
        for idx in range(self.get_attribute("xyz").shape[0]):
            xyz = self.get_attribute("xyz")[idx].detach().cpu().numpy()
            normals = np.zeros_like(xyz)
            f_dc = (self.get_attribute("rgb")[idx].detach().cpu().numpy()-0.5)/0.28209479177387814
            opacities = self.get_attribute("opacity")[idx].detach().cpu().numpy()
            scales = np.log(self.get_attribute("scale")[idx].detach().cpu().numpy())
            rot = self.get_attribute("rotate")[idx].detach().cpu().numpy()
            elements[idx] = (xyz[0], xyz[1], xyz[2], normals[0], normals[1], normals[2], f_dc[0], f_dc[1], f_dc[2], opacities, scales[0], scales[1], scales[2], rot[0], rot[1], rot[2], rot[3])
        return elements

    def save_camera_json(self, file_name):
        camera_json = utils.extract_camera_parameters(self.intr, self.extr)
        with open(file_name, "w") as f:
            json.dump(camera_json, f)


    def save_checkpoint(self, ckpt_name=None):
        checkpoint = {
            "attributes": self._attributes,
            "intr": self.intr,
            "extr": self.extr,
        }
        if ckpt_name is None:
            ckpt_name = 'ckpt'
        # make directory
        os.makedirs(os.path.join(self.dir, "ckpt"), exist_ok=True)
        self.checkpoint_path = os.path.join(self.dir, "ckpt", f"{ckpt_name}.tar")
        torch.save(checkpoint, self.checkpoint_path)
        splat_data = self.get_splat_buffer()
        os.makedirs(os.path.join(self.dir, "splat"), exist_ok=True)
        self.splat_file_path = os.path.join(self.dir, "splat", f"{ckpt_name}.splat")
        utils.save_splat_file(splat_data, self.splat_file_path)
        os.makedirs(os.path.join(self.dir, "ply"), exist_ok=True)
        self.ply_file_path = os.path.join(self.dir, "ply", f"{ckpt_name}.ply")
        ply_elements = self.get_ply_elements()
        ply_data = PlyData([PlyElement.describe(ply_elements, 'vertex')], text=True)
        os.makedirs(os.path.join(self.dir, "json"), exist_ok=True)
        self.json_file_path = os.path.join(self.dir, "json", f"{ckpt_name}.json")
        self.save_camera_json(self.json_file_path)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self._attributes = checkpoint["attributes"]
        self.intr = checkpoint["intr"]
        self.extr = checkpoint["extr"]
        del checkpoint
        torch.cuda.empty_cache()
    
    def train(self, iterations=500, lr=1e-2, lr_camera=0.,
              lambda_depth=0., lambda_flow=0., lambda_var=0., lambda_still=0.,
              save_imgs=False, save_videos=False, save_ckpt=False,
              ckpt_name="ckpt", densify_interval=500, densify_times=1, 
              grad_threshold=5e-3, mask=None, camera_only=False):
        frames = []
        frames_depth = []
        frames_center = []
        progress_bar = tqdm(range(1, iterations), desc="Training")
        l1_loss = nn.SmoothL1Loss(reduce="none")
        mse_loss = nn.MSELoss()
        mse_loss_pixel = nn.MSELoss(reduction='none')
        self.add_optimizer(lr, lr_camera, exclude_key="xyz")
        self.reset_densification_stats()
        # if hasattr(self, 'last_xyz'): # not the first frame
            # initial number of gaussians in the training
            # num_g_init = self.get_attribute("xyz").shape[0] 
        #     # if If xyz of gaussians present nearly no change compared to last xyz, 
        #     # we consider them as the still gaussians
        #     xyz_diff = torch.norm(self.get_attribute("xyz") - self.last_xyz, dim=1)
        #     print("xyz_diff", xyz_diff.min(), xyz_diff.max())
        #     still_mask = xyz_diff < 1e-9
        
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
            # print("before xy" , self.get_attribute("xyz")[:,:2].min(), self.get_attribute("xyz")[:,:2].max())
            # print("before z" , self.get_attribute("xyz")[:,2].min(), self.get_attribute("xyz")[:,2].max())
            # print("before xyz" , self.get_attribute("xyz")[:5])
            # print("uv", uv.min(), uv.max())
            # print("depth", depth.min(), depth.max())
            # print("uv", uv[:5])
            # print("depth", depth[:5])
            xyz = geometry.pix2world(uv, depth, self.intr, self.extr)
            # print("after xy",xyz[:,:2].min(), xyz[:,:2].max())
            # print("after z",xyz[:,2].min(), xyz[:,2].max())
            # print("after xyz",xyz[:5])
            
            # render depth map
            rendered_depth_map = return_dict["depth_map"]
            # print("rendered_depth_map", rendered_depth_map.min(), rendered_depth_map.max())

            # render colorful depth map
            rendered_depth_map_color = return_dict["depth_map_color"]
            
            # render center
            rendered_center = return_dict["center"]
            
            # loss_rgb = mse_loss(rendered_rgb.permute(1, 2, 0), self.gt_image)
            loss_rgb_pixel = mse_loss_pixel(rendered_rgb.permute(1, 2, 0), self.gt_image).mean(dim=2)
            loss_rgb = loss_rgb_pixel.mean()
            loss += loss_rgb
            progress_dict = {"rgb": f"{loss_rgb.item():.6f}"}

            rendered_depth_map = rendered_depth_map.permute(1, 2, 0)

            # valid_uv = uv within the a HxW range
            valid_uv_index = (uv[:,0] > 0) & (uv[:,0] < self.W-1) & (uv[:,1] > 0) & (uv[:,1] < self.H-1)
            valid_uv = uv[valid_uv_index]
            depth_point_gt = self.gt_depth[valid_uv[:,1].long(), valid_uv[:,0].long()] # shape (N, 1)
            depth_point = depth[valid_uv_index] # shape (N, 1)

            if lambda_depth > 0:
                # print("rendered_depth_map", rendered_depth_map.min(), rendered_depth_map.max())
                # print("gt_depth", self.gt_depth.min(), self.gt_depth.max())
                # loss_depth = l1_loss(depth_point, depth_point_gt) / (depth_point_gt + depth_point)
                # loss_depth = loss_depth.mean()

                loss_depth = mse_loss_pixel(rendered_depth_map, self.gt_depth) / (self.gt_depth + rendered_depth_map)
                loss_depth = loss_depth.mean()

                # loss_depth = mse_loss(rendered_depth_map, self.gt_depth)
                # loss_depth = mse_loss(rendered_depth_map+self.t3, self.gt_depth)
                loss += lambda_depth * loss_depth
                progress_dict["depth"] = f"{loss_depth.item():.6f}"

            if lambda_var: # penaliza the scales with large variance
                loss_var = torch.mean(torch.var(self.get_attribute("xyz"), dim=0))
                    # print("\t[loss] loss_var", loss_var.item())
                loss += lambda_var * loss_var
                progress_dict["var"] = f"{loss_var.item():.6f}"

            if lambda_still and hasattr(self, 'still_mask'): # pushing all gaussians to be still
                still_shape = self.still_mask.shape[0]
                loss_still = torch.norm(self.get_attribute("xyz")[:still_shape][self.still_mask] - self.last_xyz[:still_shape][self.still_mask], dim=1).mean()
                # print(loss_still.shape)
                # loss_still = loss_still / (depth_point_gt + depth_point)
                # loss_still = loss_still.mean()
                loss += lambda_still * loss_still
                progress_dict["still"] = f"{loss_still.item():.6f}"

            if lambda_flow: # ensure local flow consistency
                # select uv that uv within the a HxW range
                and_mask = (uv[:self.last_num,0] > 0) & (uv[:self.last_num,0] < self.W-1) & (uv[:self.last_num,1] > 0) & (uv[:self.last_num,1] < self.H-1)
                if hasattr(self, 'still_mask'):
                    and_mask[:self.still_mask.shape[0]] = ~self.still_mask & and_mask[:self.still_mask.shape[0]]

                pred_flow = uv[:self.last_num][and_mask] - self.last_uv[and_mask]
                # y_coords = torch.clamp(uv.detach()[:,1].long(), 0, self.H-1)
                # x_coords = torch.clamp(uv.detach()[:,0].long(), 0, self.W-1)
                y_coords = uv.detach()[:self.last_num][and_mask][:,1].long()
                x_coords = uv.detach()[:self.last_num][and_mask][:,0].long()
                # print(y_coords.max(), y_coords.min())
                # print(x_coords.max(), x_coords.min())

                gt_flow = self.gt_flow[y_coords, x_coords]

                # print(pred_flow.shape, gt_flow.shape)

                loss_flow = mse_loss(pred_flow, gt_flow)
                loss += lambda_flow * loss_flow
                progress_dict["flow"] = f"{loss_flow.item():.6f}"

            self.optimizer.zero_grad()
            loss.backward()
            # if hasattr(self, 'last_xyz'): # not the first frame
            #     # for every gaussian, if the gradient of xyz is small, then consider them as still points
            #     grad_xyz = torch.norm(self.get_attribute("xyz").grad, dim=1)
            #     print("\t[grad] grad_xyz's max, min:", grad_xyz.max().item(), grad_xyz.min().item())
            #     grad_mask = grad_xyz < 1e-7
            #     for name, param in self._attributes.items():
            #         if name == "xyz":
            #             param.grad[grad_mask] = 0.
            #         if name == "rgb":
            #             param.grad[grad_mask] = 0.

            # Make the new densified gaussians's gradients to be zero: rgb 
            if hasattr(self, 'last_xyz'): # second frame and after
                for name, param in self._attributes.items():
                    if name == "rgb":
                        # param.grad[self.last_num:] = 0.
                        param.grad = 0. * param.grad
                    # if name == "opacity":
                    #     param.grad = 0. * param.grad


            # if hasattr(self, 'last_xyz'): # second frame and after
            #     # Make the first num_g_init gaussians's rgb gradients to be zero
            #     for name, param in self._attributes.items():
            #         if name == "rgb":
            #             param.grad[:self.last_num] = 0.

            # Make the gaussians in still_mask to be still
            if hasattr(self, 'still_mask'): # third frame and after
                for name, param in self._attributes.items():
                    if name == "xyz":
                        param.grad[:self.still_mask.shape[0]][self.still_mask] = 0.

            #     # If xyz of gaussians is nearly no change, we consider them as the still gaussians
            #     # and make their gradients to be zero
            #     for name, param in self._attributes.items():
            #         if name == "xyz":
            #             param.grad[still_mask] = 0.
            #         if name == "rgb":
            #             param.grad[still_mask] = 0.

            self.optimizer.step()

            progress_dict["total"] = loss.item()
            progress_bar.set_postfix(progress_dict)
            # progress_bar.set_description(progress_dict)
            
            # progress_bar.set_postfix({"Loss": f"{loss:.{7}f}"})
            progress_bar.update(1)

            # if densify_interval:
            #     grad_dict = {
            #         "xyz": self._attributes["xyz"],
            #         # "scales": self.scales,
            #         # "quats": self.quats,
            #     }
            #     self.add_densification_stats(grad_dict)


            if not camera_only and densify_interval and (iteration+1) % densify_interval == 0 and (iteration+1) // densify_interval <= densify_times:
                # grad_dict = {
                #     "xyz": self._attributes["xyz"],
                #     # "scales": self.scales,
                #     # "quats": self.quats,
                # }
                # self.add_densification_stats(grad_dict)
                
                # grad_norm = self.acquire_densification_stats()
                # print("\n\t[densify] grad_norm's max, min:", grad_norm.max().item(), grad_norm.min().item())
                # print("\t[densify] grad_norm's mean, threshold:", grad_norm.mean().item(), grad_threshold)
                # scene_extent = (self.H+self.W)/2
                # self.densify_and_clone(grad_norm, grad_threshold, scene_extent, 0.00005, N=2, mask=mask)
                # self.densify_and_split(grad_norm, grad_threshold*1, scene_extent, 0.0001, N=2, mask=mask, there=True)
                # # self.densify_and_split(grad_norm, grad_threshold*1, scene_extent, 0.0001, N=2, mask=None, there=True)
                # torch.cuda.empty_cache()
                # self.reset_densification_stats()
                # if hasattr(self, 'still_mask'): # update the still_mask
                #     still_mask_pad = torch.zeros(self.get_attribute("xyz").shape[0]-num_g_init, dtype=torch.bool).to(self.device)
                #     self.still_mask = torch.cat((self.still_mask, still_mask_pad), dim=0)
                if not hasattr(self, 'last_xyz'): # the first frame
                    self.densify_by_pixels(loss_rgb_pixel, error_threshold=1e-2, percent=0.1, mask=mask)
                else:
                    self.densify_by_pixels(torch.ones_like(loss_rgb_pixel), error_threshold=0., percent=0.03, mask=mask)
                    self.densify_by_pixels(loss_rgb_pixel, error_threshold=1e-1, percent=0.1, mask=None)


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
        if not camera_only:
            if hasattr(self, 'last_xyz'): # not the first frame
                # if If xyz of gaussians present nearly no change compared to last xyz, 
                # we consider them as the still gaussians
                xyz_diff = self.get_attribute("xyz")[:self.last_num] - self.last_xyz # shape (N, 3)
                # print("xyz_diff.shape", xyz_diff.shape)
                xyz_diff_norm = torch.norm(xyz_diff, dim=1)
                # print("xyz_diff_norm.shape", xyz_diff_norm.shape)
                self.still_mask = xyz_diff_norm < 0.09
                print("\n\t[still] xyz_diff_norm", xyz_diff_norm[xyz_diff_norm>0].min().item(), xyz_diff_norm.max().item())

                # print(xyz_diff.shape)
                # print(depth_point.shape)
                # print(self.last_num)
                # xyz_diff_norm_scaled = xyz_diff_norm / depth_point[:self.last_num].squeeze()
                # print("\t[still] xyz_diff_norm", xyz_diff_norm[xyz_diff_norm>0].min().item(), xyz_diff_norm.max().item())
                # self.still_mask = xyz_diff_norm < 0.01


                within_mask = (uv[:self.last_num,0] > 0) & (uv[:self.last_num,0] < self.W-1) & (uv[:self.last_num,1] > 0) & (uv[:self.last_num,1] < self.H-1)
                y_coords = uv.detach()[:self.last_num][within_mask][:,1].long()
                x_coords = uv.detach()[:self.last_num][within_mask][:,0].long()
                gt_flow_point = self.gt_flow[y_coords, x_coords]
                # print(uv.shape)
                # print(self.last_num)
                # print(within_mask.shape)
                # print(y_coords.shape)
                # print(x_coords.shape)
                # print(gt_flow_point.shape)

                # rigid_feat = torch.concat((xyz_diff,self.get_attribute("xyz")[:self.last_num]),dim=1).detach().squeeze().cpu().numpy()
                # rigid_feat = xyz_diff.detach().squeeze().cpu().numpy()
                rigid_feat = gt_flow_point.detach().squeeze().cpu().numpy()
                labels = self.clustering.fit_predict(rigid_feat)
                labels = torch.from_numpy(labels).to(self.device)
                # print(labels.shape)
                # turn to bool tensor
                cluster0 = xyz_diff_norm[within_mask][labels == 0].mean()
                cluster1 = xyz_diff_norm[within_mask][labels == 1].mean()
                larger_cluster_label = 0 if cluster0 > cluster1 else 1 # movement is larger, means move part
                self.still_mask = (labels != larger_cluster_label)
                # print("self.still_mask.shape", self.still_mask.shape)
                self.still_mask = torch.ones(self.last_num, dtype=torch.bool).to(self.device)
                self.still_mask[within_mask] = (labels != larger_cluster_label)
                # inheret the still mask
                if hasattr(self, 'last_still_mask'):
                    self.still_mask[:self.last_still_mask.shape[0]] = self.still_mask[:self.last_still_mask.shape[0]] | self.last_still_mask
                # print("self.still_mask.shape", self.still_mask.shape)
                # print(labels.min(), labels.max())
                # print("label mask ratio is ", (labels.sum() / labels.shape[0]).item())

                # print mask ratio
                print("\t[still] mask ratio is", self.still_mask.sum().item() / self.last_num)
                self.last_still_mask = self.still_mask
            self.last_uv = uv.detach()
            self.last_depth = depth.detach()
            self.last_xyz = self.get_attribute("xyz").detach()
            self.last_num = self.last_xyz.shape[0]
        
        progress_bar.close()

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
                self.extr,
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
                self.extr,
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

        return frames, frames_center, frames_depth, still_rgb_np, still_center_np, move_rgb_np, move_center_np
    
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
            # print(num_total)

            self.traj_scale = torch.ones((num_total, 3), device=self.device).float() * 1e-6 # too big will cause memory overflow

            self.traj_rotate = torch.tensor([1, 0, 0, 0], device=self.device).repeat(num_total, 1).float()

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

    def densify_by_pixels(self, error_map, error_threshold=1e-3, percent=0.1, mask=None, ):
        num_before = self.get_attribute("xyz").shape[0]
        error_map = error_map.detach().cpu().numpy()
        # print(error_map.shape)
        print("\n\t[densify] error_map's max, min, mean:", error_map.max().item(), error_map.min().item(), error_map.mean().item())
        # add uniform magnitude, to avoid zero probability
        error_map = error_map + np.nanmin(error_map[error_map>0])
        if mask is not None:
            mask = mask.detach().cpu().numpy().squeeze()
            # transfer to bool type
            mask = mask > 0
            error_map = error_map*mask

        # make positive elements to be 1, and negative elements to be 0
        # error_map = np.where(error_map > error_threshold, 1, 0)

        probability_distribution = error_map / np.sum(error_map)
        densify_num = int(np.sum(error_map > error_threshold) * percent)

        # sample points from the probability distribution
        sampled_points = np.random.choice(a=np.arange(self.H*self.W), size=densify_num, p=probability_distribution.flatten())

        if sampled_points.size > 0:
            # convert the sampled points to coordinates on the image
            sampled_coordinates = np.unravel_index(sampled_points, (self.H, self.W))

            xys = np.array(sampled_coordinates).T[:,::-1].copy()
            depths = self.gt_depth[sampled_coordinates]
            scales = 1 / probability_distribution[sampled_coordinates]
            scales = 0.1 * scales / np.max(scales)
            # scales = scales / 1e3
            scales = scales * (depths.cpu().numpy()/depths.cpu().numpy().min()).squeeze()
            rgbs = self.gt_image[sampled_coordinates]
            # print("\t[densify] sampled rgbs' max, min:", rgbs.max().item(), rgbs.min().item())
            # rgbs = rgbs / 255.

            xys = torch.from_numpy(xys).to(self.device).float()
            depths = depths.to(self.device).float()
            new_xyz = geometry.pix2world(xys, depths, self.intr, self.extr)
            scales = torch.from_numpy(scales).float().unsqueeze(1).repeat(1, 3).to(self.device)
            new_scale = self._activations_inv["scale"](scales)
            rgbs = rgbs.contiguous()
            eps = 1e-15  # avoid logit function input 0 or 1
            rgbs = torch.clamp(rgbs, min=eps, max=1-eps)
            # calculate the inverse of sigmoid function, i.e., logit function
            new_rgb = self._activations_inv["rgb"](rgbs)
            new_rotate = torch.tensor([1, 0, 0, 0], device=self.device).repeat(new_xyz.shape[0], 1).float()
            # new_opacity = torch.rand((new_xyz.shape[0], 1), device=self.device).float()
            new_opacity = torch.ones((new_xyz.shape[0], 1), device=self.device).float() - eps
            new_opacity = self._activations_inv["opacity"](new_opacity)

            self.densification_postfix(new_xyz, new_scale, new_rotate, new_opacity, new_rgb)

        num_after = self.get_attribute("xyz").shape[0]
        print(f"\t[densify / split] number of gaussians: {num_before} -> {num_after}")

    
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
    # img = imageio.imread("./images/car-turn/car-turn/00000.jpg") # H, W, C
    img = img.astype(np.float32) / 255.0
    gt_image = torch.from_numpy(img).cuda().permute(2, 0, 1) # C, H, W
    
    depth = imageio.imread("./data/face_depth.png") # H, W, C
    depth = np.expand_dims(depth, axis=-1)
    # depth = imageio.imread("./images/car-turn/car-turn/00000.jpg") # H, W, C
    depth = depth.astype(np.float32) / 255.0
    gt_depth = torch.from_numpy(depth).cuda().permute(2, 0, 1) # C, H, W
    gt_depth = gt_depth.mean(dim=0, keepdim=True)
    print(gt_image.shape, gt_depth.shape)
    
    C, H, W = gt_image.shape 
    
    gaussians = SimpleGaussian(gt_image=gt_image.permute(1, 2, 0), gt_depth=gt_depth.permute(1,2,0), num_points=10000, depth_scale=10.) # H, W, C
    gaussians.init_gaussians_from_image(gt_image=gt_image.permute(1, 2, 0), gt_depth=gt_depth.permute(1,2,0), num_points=10000, t3=gaussians.t3)
    gaussians.train(iterations=500, lr=1e-2, lambda_depth=0.01, 
                    save_imgs=True, save_videos=True, save_ckpt=True, 
                    ckpt_name="ckpt", 
                    densify_interval=500, densify_times=0, grad_threshold=5e-6, mask=None)
    