"""A simple example to render a (large-scale) Gaussian Splats

```bash
python gflow/viewer.py --gpu 0 --port 8082 --folder "logs/2024_08_03-21_37_27"
```
"""

import argparse
import os
import time

import torch
from tqdm import tqdm
import viser
import os

import utils.render as render
import time
import roma
from utils.geometry import inv

def sensitive_sigmoid(x, scale=10.):
    return torch.sigmoid(x * scale)

def sensitive_logit(x, scale=10.):
    return torch.logit(x) / scale

activations = {
    "scale": lambda x: torch.abs(x),
    # "scale": lambda x: torch.abs(x) + 1e-8,
    # "scale": _isotropic,
    "rotate": torch.nn.functional.normalize,
    # "opacity": torch.sigmoid,
    "opacity": sensitive_sigmoid,
    "rgb": torch.sigmoid
}

def log_loader(log_folder):
    # Load the log folder
    ckpt_folder = os.path.join(log_folder, "ckpt")

    ckpt_files = sorted([os.path.join(ckpt_folder, f) for f in os.listdir(ckpt_folder) if f.endswith('.tar')])

    data = torch.load(ckpt_files[0], map_location="cpu")
    width = int(data['width'])
    height = int(data['height'])

    return ckpt_files, width, height

def load_ckpt(ckpt_file, device):
    # load checkpoint
    checkpoint = torch.load(ckpt_file, map_location=device)
    attributes = checkpoint["attributes"]
    xyz = attributes["xyz"]
    scale = activations["scale"](attributes["scale"])
    rotate = activations["rotate"](attributes["rotate"])
    opacity = activations["opacity"](attributes["opacity"])
    rgb = activations["rgb"](attributes["rgb"])
    intr = checkpoint["intr"]
    extr = checkpoint["extr"]

    print("Number of Gaussians:", len(xyz))

    return xyz, scale, rotate, opacity, rgb, intr, extr

def extr_to_quan_pos(extr):
    # extr (w2c): 3x4
    extr = extr.detach()
    # 3x4 -> 4x4
    extr = torch.cat([extr, torch.tensor([[0., 0., 0., 1.]], device=extr.device)], dim=0)
    c2w = inv(extr)
    q = roma.rotmat_to_unitquat(c2w[:3, :3]).cpu().numpy().tolist()
    pos = c2w[:3, 3].cpu().numpy().tolist()
    return q, pos

def quan_pos_to_extr(q, pos):
    c2w = torch.eye(4)
    c2w[:3, :3] = roma.unitquat_to_rotmat(torch.tensor(q))
    c2w[:3, 3] = torch.tensor(pos)
    w2c = inv(c2w) # world2cam
    extr = w2c[:3, :].detach() # 3x4
    return extr

def main(args):
    torch.manual_seed(42)
    device = torch.device("cuda", args.gpu)
    print("Using device:", device)

    print("Loading log folder: ", args.folder)

    ckpt_files, width, height = log_loader(args.folder)

    # bg = 0. # background color (black)
    bg = 1. # background color (white)

    # load checkpoints
    input_group_list = []
    original_extr = []
    wxyz_pos_list = []
    for ckpt_file in tqdm(ckpt_files, desc="Loading checkpoints"):
        xyz, scale, rotate, opacity, rgb, intr, extr = load_ckpt(ckpt_file, device=device)
        input_group = [xyz, scale, rotate, opacity, rgb, intr, extr, bg, width, height]
        input_group_list.append(input_group)
        original_extr.append(extr)

        q, pos = extr_to_quan_pos(extr)
        wxyz = (q[3], q[0], q[1], q[2])
        pos = (pos[0], pos[1], pos[2])
        wxyz_pos_list.append([wxyz, pos])


    server = viser.ViserServer(port=args.port, verbose=False)

    with server.gui.add_folder("Rendering"):
        fps_gui = server.gui.add_number(
            label="fps",
            initial_value=0,
            disabled=True,
            hint="frames per second",
        )

        n_gaussians = server.gui.add_number(
            label="Number of Gaussians",
            initial_value=len(xyz),
            disabled=True,
            hint="Number of Gaussians",
        )

    with server.gui.add_folder("Frames"):
        current_frame = server.gui.add_number(
            label="Current frame",
            initial_value=0,
            disabled=True,
            hint="Current frame number",
        )

        prev_button = server.gui.add_button(
            label="Prev",
            hint="Go to previous frame",
        )

        next_button = server.gui.add_button(
            label="Next",
            hint="Go to next frame",
        )

    with server.gui.add_folder("Camera"):
        camera_rot = server.gui.add_text(
            label="Rotation",
            initial_value="1, 0, 0, 0",
            disabled=True,
            hint="Rotation quaternion",
        )
        camera_pos = server.gui.add_text(
            label="Position",
            initial_value="0, 0, 0",
            disabled=True,
            hint="Camera position",
        )

        following = server.gui.add_checkbox("Follow training view", False)

        back_button = server.gui.add_button(
            label="Back to origin camera",
            hint="Reset camera to origin",
        )
    
    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = (0., 0., 0.)
        client.camera.wxyz = (1., 0., 0., 0.)
        # extr = original_extr[current_frame.value] # world2cam
        # q, pos = extr_to_quan_pos(extr)
        # client.camera.wxyz = (q[3], q[0], q[1], q[2])
        # client.camera.position = (pos[0], pos[1], pos[2])

        @back_button.on_click
        def _(_) -> None:
            extr = original_extr[current_frame.value] # world2cam
            q, pos = extr_to_quan_pos(extr)
            client.camera.wxyz = (q[3], q[0], q[1], q[2])
            client.camera.position = (pos[0], pos[1], pos[2])
        

        # This will run whenever we get a new camera!
        @client.camera.on_update
        def _(_: viser.CameraHandle) -> None:
            camera_rot.value = f"{client.camera.wxyz[0]:.3f}, \n{client.camera.wxyz[1]:.3f}, \n{client.camera.wxyz[2]:.3f}, \n{client.camera.wxyz[3]:.3f}"
            camera_pos.value = f"{client.camera.position[0]:.4f}, \n{client.camera.position[1]:.4f}, \n{client.camera.position[2]:.4f}"

        @prev_button.on_click
        def _(_) -> None:
            current_frame.value = max(current_frame.value - 1, 0)

        @next_button.on_click
        def _(_) -> None:
            current_frame.value = min(current_frame.value + 1, len(ckpt_files) - 1)


    print("Viewer running... Ctrl+C to exit.")
    while True:
        for client in server.get_clients().values():
            t_start = time.time()
            camera = client.camera # cam2world
            wxyz = camera.wxyz
            postion = camera.position
            extr = quan_pos_to_extr([wxyz[1], wxyz[2], wxyz[3], wxyz[0]], [postion[0], postion[1], postion[2]]).to(device)
            
            current_input_group = input_group_list[current_frame.value]
            if following.value: # follow the training view
                current_input_group[-4] = original_extr[current_frame.value]

                client.camera.wxyz = wxyz_pos_list[current_frame.value][0]
                client.camera.position = wxyz_pos_list[current_frame.value][1]
            else:
                current_input_group[-4] = extr

            # print("render number of gaussians: ", len(current_input_group[0]))
            n_gaussians.value = len(current_input_group[0])

            rendered_dict = render.render_multiple(current_input_group, ["rgb"])
            rendered_rgb_np = render.render2img(rendered_dict["rgb"])

            t_end = time.time()
            fps = 1. / (t_end - t_start)
            fps_gui.value = fps

            client.scene.set_background_image(rendered_rgb_np, format="jpeg")
    
    # time.sleep(100000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder", type=str, help="path to the logs folder"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="port for the viewer server"
    )
    parser.add_argument(
        "--gpu", type=int, default=0, help="which GPU to use"
    )
    args = parser.parse_args()

    main(args)