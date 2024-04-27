import torch
import argparse
import os
import numpy as np
from io import BytesIO
import json
from datetime import datetime
import shutil

def load_tar(tar_file_path):
    checkpoint = torch.load(tar_file_path, map_location="cpu")
    attributes = checkpoint["attributes"]
    intr = checkpoint["intr"]
    extr = checkpoint["extr"]
    return attributes, intr, extr

activations = {
    "scale": lambda x: torch.abs(x) + 1e-8,
    "rotate": torch.nn.functional.normalize,
    "opacity": torch.sigmoid,
    "rgb": torch.sigmoid
}

def get_attribute(attributes, name):
    try:
        if name in activations.keys() and activations[name] is not None:
            return activations[name](attributes[name])
        else:
            return attributes[name]
    except:
        raise ValueError(f"Attribute or activation for {name} is not VALID!")


def process_attributes(attributes_origin):
    attributes = {}
    for key in attributes_origin.keys():
        attributes[key] = get_attribute(attributes_origin, key)

    sorted_indices = np.argsort(-np.prod(attributes["scale"].detach().cpu().numpy(), axis=1) / (1 + np.exp(-attributes["opacity"].detach().cpu().T.numpy())))
    buffer = BytesIO()
    for idx in sorted_indices[0]:
        position = attributes["xyz"][idx].detach().numpy()
        scales = attributes["scale"][idx].detach().numpy()
        rot = attributes["rotate"][idx].detach().numpy()
        color = attributes["rgb"][idx].detach().numpy()
        color = np.append(color, 1 / (1 + np.exp(-attributes["opacity"][idx].detach().numpy())))
        buffer.write(position.astype(np.float32).tobytes())
        buffer.write(scales.astype(np.float32).tobytes())
        buffer.write((color * 255).clip(0, 255).astype(np.uint8).tobytes())
        # buffer.write(np.)
        # print(len((rot * 128 + 128).clip(0, 255).astype(np.uint8).tobytes()))
        buffer.write((rot * 128 + 128).clip(0, 255).astype(np.uint8).tobytes())
    return buffer.getvalue()

def save_splat_file(splat_data, output_path):
    with open(output_path, "wb") as f:
        f.write(splat_data)
    # print("Data written to {}".format(output_path))

def extract_camera_parameters(intrinsic_matrix, extrinsic_matrix):
    # Extract focal lengths and principal point from the intrinsic matrix
    [fx, fy, cx, cy] = intrinsic_matrix.detach().numpy().tolist()

    # Extract rotation matrix and translation vector from the extrinsic matrix
    R = extrinsic_matrix[:, :3]
    t = extrinsic_matrix[:, 3]

    # Calculate camera position in world coordinates
    camera_position = -np.linalg.inv(R.detach().numpy()).dot(t.detach().numpy())

    # Return all extracted parameters
    return [{
        "id": 0,
        "img_name": "00001",
        "width": 864,
        "height": 480,
        "position": camera_position.tolist(),
        "rotation": R.tolist(),
        "fx": fx,
        "fy": fy
    }]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tar_file", type=str, required=True, help="Path to the tar file.")
    parser.add_argument("--output_folder", type=str, default="", help="Output folder name. If not provided, the current time will be used. (all the outputs will be saved in the logs folder)")
    args = parser.parse_args()
    directory = ""
    if args.output_folder:
        directory = os.path.join("logs", args.output_folder)
    else:
        now = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        directory = f"logs/{now}_vg"
    os.makedirs(directory, exist_ok=True)
    tar_output_path = os.path.join(directory, "ckpt")
    os.makedirs(tar_output_path, exist_ok=True)
    splat_output_path = os.path.join(directory, "splat")
    os.makedirs(splat_output_path, exist_ok=True)
    json_output_path = os.path.join(directory, "json")
    os.makedirs(json_output_path, exist_ok=True)

    tar_file = args.tar_file
    attributes, intr, extr = load_tar(tar_file)
    # Copy the original file into the directory using shutil
    shutil.copy(tar_file, os.path.join(tar_output_path, os.path.basename(tar_file)))


    tar_basename = "".join(os.path.basename(tar_file).split(".")[:-1])

    buffer = process_attributes(attributes)
    save_splat_file(buffer, os.path.join(splat_output_path, "{}.splat".format(tar_basename)))

    parameters = extract_camera_parameters(intr, extr)
    # import pdb; pdb.set_trace()
    with open(os.path.join(json_output_path, "{}.json".format(tar_basename)), 'w') as f:
        json.dump(parameters, f)
    
    print("Finished processing the tar file. Outputs are saved in the {} folder.".format(directory))