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
import numpy as np
from PIL import Image

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

        Outputs:
            depth: estimated metric depth map
            K: intrinsic parameters
        """
        # Run inference on images and save to export_dir
        #   => export_dir=None, not saving any 3D point cloud results
        #   => save only the (1) intrinsics and (2) depth map instead
        prediction = self.model.inference(
            images,
            export_dir=None, # export_dir: not saving any final PC results.
            # export_format="glb",  # Options: glb, npz, ply, mini_npz, gs_ply, gs_video,
            show_cameras=False,
            # process_res=max(sample_img.size),
        )

        # # remove unnecessary files generated during inference
        # if export_dir is not None:
        #     os.remove(os.path.join(export_dir, "scene.jpg"))

        ###################################
        ### [DEVELOPER] Extract info    ###
        ###################################
        # image = np.asarray(Image.open(images[0]))
        depth = prediction.depth[0]  # single first frame
        K = prediction.intrinsics[0] # single first frame
        # print('Image shape      :', image.shape)
        # print('Depth shape      :', depth.shape)
        # print('Intrinsics shape :', K.shape)
        
        # save depth and intrinsics
        os.makedirs(export_dir, exist_ok=True)
        np.save(os.path.join(export_dir, 'depth.npy'), depth)
        np.save(os.path.join(export_dir, 'intrinsics.npy'), K)

        # # Access results
        # print('Depth shape:', prediction.depth.shape)        # Depth maps: [N, H, W] float32
        # print(prediction.conf.shape)         # Confidence maps: [N, H, W] float32
        # print('Extrinsics :', prediction.extrinsics.shape)   # Camera poses (w2c): [N, 3, 4] float32
        # print('Intrinsics :', prediction.intrinsics.shape)   # Camera intrinsics: [N, 3, 3] float32

        # P = prediction.extrinsics
        # K = prediction.intrinsics[0]
        # D = prediction.depth[0]
        # s = prediction.scale_factor
        # print(P.round(5))
        # print(K.round(5))
        # print(s)
        # print(np.linalg.inv(K))
        return depth, K


""" This is for testing on a single image only
    To process on the whole folder, import the above functions

    Example:
    python generate_3Dpc.py \
    --img_path=../examples/nusc/img_4/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151605512404.jpg \
    --export_path=../examples/nusc/img_4/da3_output/
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 3D point cloud from monocular image")
    parser.add_argument("--img_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--export_path", type=str, default=None, help="Directory to save output 3D point cloud")
    args = parser.parse_args()

    # generate 3D point cloud and save to export_dir
    generator = PointCloud3DGenerator()
    generator.generate_3Dpc([args.img_path], args.export_dir)