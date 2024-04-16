
import math
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import imageio
import msplat
import utils
from init import complex_texture_sampling


class SimpleGaussian:
    def __init__(self, num_points=100000, intr=None, extr=None, img=None, t3=1.):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.intr = intr
        self.extr = extr
        
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

        if img is not None:
            self.init_gaussians_from_image(img, num_points=num_points, t3=t3)
        
        for attribute_name in self._attributes.keys():
            self._attributes[attribute_name] = nn.Parameter(self._attributes[attribute_name]).requires_grad_(True)
        
        self.optimizer = torch.optim.Adam(list(self._attributes.values()), lr=1e-2)
    
    def add_attribute(self, name, value, clear_grad=True):
        if clear_grad:
            self.optimizer.zero_grad()
        self._attributes[name] = nn.Parameter(value).requires_grad_(True)
        self.optimizer = torch.optim.Adam(list(self._attributes.values()), lr=1e-3)
    
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
        # xys = torch.from_numpy(xys).to(self.device).float()
        self.gt_depth = gt_depth.to(self.device).float()
        # print(xys.shape)
        depths = depths.to(self.device).float()
        # depths = torch.ones_like(depths).to(self.device).float() * 0.
        xyd = torch.concatenate((xys, depths), dim=1)

        import pdb
        input_xyd = xyd.clone()
        input_xyd[:, :2] = input_xyd[:, :2] / t3
        input_xyd[:, 0] = (input_xyd[:, 0] + 1) * W / 2.
        input_xyd[:, 1] = (input_xyd[:, 1] + 1) * H / 2.
        input_xyd[:, :2] = input_xyd[:, :2] / input_xyd[:, 2:3]

        dummy_xyz = utils.camera2world(input_xyd, self.intr, self.extr)
        xyd[:10, :] = dummy_xyz[:10, :]
        self._attributes["xyz"] = xyd
        pdb.set_trace()
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
        self.optimizer.step()
        self.optimizer.zero_grad()
    
    def get_attribute(self, name):
        try:
            if name in self._activations.keys() and self._activations[name] is not None:
                return self._activations[name](self._attributes[name])
            else:
                return self._attributes[name]
        except:
            raise ValueError(f"Attribute or activation for {name} is not VALID!")

if __name__ == "__main__":
    seed = 123
    torch.manual_seed(seed)
    
    # img = imageio.imread("./data/stanford-bunny.jpg") # H, W, C
    img = imageio.imread("./data/face.png") # H, W, C
    img = img.astype(np.float32) / 255.0
    gt = torch.from_numpy(img).cuda().permute(2, 0, 1) # C, H, W
    
    C, H, W = gt.shape 
    
    bg = 1
    fov = math.pi / 2.0
    fx = 0.5 * float(W) / math.tan(0.5 * fov)
    fy = 0.5 * float(H) / math.tan(0.5 * fov)
    intr = torch.Tensor([fx, fy, float(W) / 2, float(H) / 2]).cuda().float()
    intr_matrix = torch.Tensor([[fx,  0.0, float(W) / 2],
                                [0.0, fy,  float(H) / 2],
                                [0.0, 0.0, 1.0         ]]).cuda().float()
    t3 = 2.5 # translation along z-axis
    extr = torch.Tensor([[1.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0, t3 ]]).cuda().float()
    
    gaussians = SimpleGaussian(num_points=10000, intr=intr_matrix, extr=extr, img=gt.permute(1, 2, 0), t3=t3) # H, W, C
    
    max_iter = 2000
    frames = []
    progress_bar = tqdm(range(1, max_iter), desc="Training")
    cal_loss = nn.SmoothL1Loss()
    
    for iteration in range(0, max_iter):
        # project points
        (uv, depth) = msplat.project_point(
            gaussians.get_attribute("xyz"), 
            intr, extr, W, H
        )
        # uv_pdb
        import pdb 
        pdb.set_trace()
        # uv[:10]
        visible = depth != 0

        # compute cov3d
        cov3d = msplat.compute_cov3d(
            gaussians.get_attribute("scale"), 
            gaussians.get_attribute("rotate"), 
            visible
        )

        # ewa project
        (conic, radius, tiles_touched) = msplat.ewa_project(
            gaussians.get_attribute("xyz"), 
            cov3d, 
            intr, extr, uv, 
            W, H, visible
        )

        # sort
        (gaussian_ids_sorted, tile_range) = msplat.sort_gaussian(
            uv, depth, W, H, radius, tiles_touched
        )

        # alpha blending
        rendered_feature = msplat.alpha_blending(
            uv, conic, 
            gaussians.get_attribute("opacity"),
            gaussians.get_attribute("rgb"), 
            gaussian_ids_sorted, tile_range, 
            bg, W, H,
        )
        
        # rendered_feature = msplat.rasterization(
        #     gaussians.get_attribute("xyz"),
        #     gaussians.get_attribute("scale"),
        #     gaussians.get_attribute("rotate"), 
        #     gaussians.get_attribute("opacity"),
        #     gaussians.get_attribute("rgb"),
        #     intr,
        #     extr,
        #     W, H, bg)
        
        loss = cal_loss(rendered_feature, gt)
        loss.backward()
        gaussians.step()
        
        progress_bar.set_postfix({"Loss": f"{loss:.{7}f}"})
        progress_bar.update(1)
        
        if iteration % 20 == 0:
            render = rendered_feature.detach().permute(1, 2, 0)
            render = torch.clamp(render, 0.0, 1.0)
            render = (render.cpu().numpy() * 255).astype(np.uint8)
            
            empty = np.ones((render.shape[0], 2, 3), dtype=np.uint8)
            show_data = np.hstack((render, empty, (img * 255).astype(np.uint8)))
            frames.append(show_data)
    
    # optimize camera extr
    max_iter = 500
    progress_bar = tqdm(range(1, max_iter), desc="Training")
    for attribute_name in gaussians._attributes.keys():
        gaussians._attributes[attribute_name] = nn.Parameter(gaussians._attributes[attribute_name]).requires_grad_(False)
    gaussians._attributes["xyz"][:,:] += 0.3
    gaussians.add_attribute("extr", extr)
    for iteration in range(0, max_iter):
        rendered_feature = msplat.rasterization(
            gaussians.get_attribute("xyz"),
            gaussians.get_attribute("scale"),
            gaussians.get_attribute("rotate"), 
            gaussians.get_attribute("opacity"),
            gaussians.get_attribute("rgb"),
            intr,
            gaussians.get_attribute("extr"),
            W, H, bg)
        
        if iteration % 20 == 0:
            render = rendered_feature.detach().permute(1, 2, 0)
            render = torch.clamp(render, 0.0, 1.0)
            render = (render.cpu().numpy() * 255).astype(np.uint8)
            
            empty = np.ones((render.shape[0], 2, 3), dtype=np.uint8)
            show_data = np.hstack((render, empty, (img * 255).astype(np.uint8)))
            frames.append(show_data)
        
        loss = cal_loss(rendered_feature, gt)
        loss.backward()
        gaussians.step()
        
        progress_bar.set_postfix({"Loss": f"{loss:.{7}f}"})
        progress_bar.update(1)
        
        if iteration % 20 == 0:
            render = rendered_feature.detach().permute(1, 2, 0)
            render = torch.clamp(render, 0.0, 1.0)
            render = (render.cpu().numpy() * 255).astype(np.uint8)
            
            empty = np.ones((render.shape[0], 2, 3), dtype=np.uint8)
            show_data = np.hstack((render, empty, (img * 255).astype(np.uint8)))
            frames.append(show_data)
    
    progress_bar.close()
    
    # save them as a video with imageio
    frames = np.stack(frames, axis=0)
    imageio.mimwrite("tutorial_2d.mp4", frames, fps=30)
    