import pickle
from PIL import Image
from tqdm import tqdm
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--tapvid_path", type=str)
parser.add_argument("--davis_path", type=str, default="./data/davis")
args = parser.parse_args()

pkl_path = os.path.join(args.tapvid_path, "tapvid_davis.pkl")

f = pickle.load(open(pkl_path, 'rb'))
for id in tqdm(f.keys()):
    if not os.path.exists(f"{args.davis_path}/{id}/{id}/"):
        print("Path does not exist, creating folder and extract images from tapvid davis for scene: ", id)
        os.makedirs(f"{args.davis_path}/{id}/{id}/")
        for i in tqdm(range(f[id]["video"].shape[0])):
            result = Image.fromarray(f[id]["video"][i])
            result.save(f"{args.davis_path}/{id}/{id}/{i:05d}.jpg")
    pickle.dump({"points":f[id]["points"], "occluded":f[id]["occluded"]}, open(f"{args.davis_path}/{id}/{id}/tracking.pkl", "wb"))