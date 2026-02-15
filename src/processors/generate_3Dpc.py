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

class PointCloud3DGenerator:
    def __init__(self):
        # Model initialization is done globally to avoid reloading the model for every image
        # Load model from Hugging Face Hub
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DepthAnything3.from_pretrained("depth-anything/da3nested-giant-large")
        self.model = self.model.to(device=self.device)

    def generate_3Dpc(self, images, export_dir):
        """ 
        Args:
            images: list of image paths 
            (for multiview; for monocular, use list of [single_image_path])
        """
        # Run inference on images and save to export_dir
        prediction = self.model.inference(
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


""" This is for testing on a single image only
    To process on the whole folder, import the above functions
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 3D point cloud from monocular image")
    parser.add_argument("--img_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--export_dir", type=str, required=True, help="Directory to save output 3D point cloud")
    args = parser.parse_args()

    # generate 3D point cloud and save to export_dir
    generator = PointCloud3DGenerator()
    generator.generate_3Dpc([args.img_path], args.export_dir)