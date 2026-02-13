##################################################################
# Author: anh-nn01; Feb 09, 2026
# Description: 
#   Generate 3D point cloud from monocular image
##################################################################

import sys
import os
import argparse
pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{pwd}/../../Depth-Anything-3/")

import torch
from depth_anything_3.api import DepthAnything3

# Load model from Hugging Face Hub
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DepthAnything3.from_pretrained("depth-anything/da3nested-giant-large")
model = model.to(device=device)


def generate_3Dpc(images, export_dir):
    global model
    # Run inference on images and save to export_dir
    prediction = model.inference(
        images,
        export_dir=export_dir,
        export_format="glb",  # Options: glb, npz, ply, mini_npz, gs_ply, gs_video,
        show_cameras=False,
    )

    # remove unnecessary files generated during inference
    os.remove(os.path.join(export_dir, "scene.jpg"))

    # # Access results
    # print(prediction.depth.shape)        # Depth maps: [N, H, W] float32
    # print(prediction.conf.shape)         # Confidence maps: [N, H, W] float32
    # print(prediction.extrinsics.shape)   # Camera poses (w2c): [N, 3, 4] float32
    # print(prediction.intrinsics.shape)   # Camera intrinsics: [N, 3, 3] float32


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 3D point cloud from monocular image")
    parser.add_argument("--img_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--export_dir", type=str, required=True, help="Directory to save output 3D point cloud")
    args = parser.parse_args()

    generate_3Dpc([args.img_path], args.export_dir)