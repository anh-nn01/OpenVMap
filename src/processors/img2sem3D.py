##################################################################
# Author: anh-nn01; March 11, 2026
# Description: 
#   Generete sematic 3D point clouds directly from input image
#   (1)  Clear occlusion  (img -> img_clear)    
#           => SAM3 + ObjectClear
#   (2a) 2D semantic mask (img_clear -> sem2d)    
#           => SAM3 
#   (2b) Metric Depth + Intrinsincs (img_clear -> K,D)
#           => DA3
#   (3)  3D unprojection: (img_clear, sem2d, K, D -> points3D)
#           => geometry
##################################################################

import os
import sys
import argparse
pwd = os.path.dirname(os.path.abspath(__file__))
import numpy as np
from PIL import Image
import cv2
import time
import uuid
from termcolor import colored

# configs
sys.path.append(f"{pwd}/../configs")
import cfg_bev

# 2D occlusion clearance
from clear_occlusion import OcclusionFreeGenerator
# 2D semantic info
from generate_semantics import Semantic2DGenerator
# 3D unprojection methods
from generate_3Dpc import PointCloud3DGenerator
# 2D info to 3D points
from generate_bev_grid import img_reconstruct_3D_points, visualize_outputs


# Initialize generators
generator_clear = OcclusionFreeGenerator(steps=15) # diffusion steps
generator_sem2d = Semantic2DGenerator(threshold=cfg_bev.SEG_THRESHOLD)
generator_3d = PointCloud3DGenerator()

""" TODO: list of img paths"""
img_path = '/fs/nexus-projects/open_vectormap/src/examples/nusc/img_8/sample_nuscene_87.png'
vis_path = f'{pwd}/debug_outputs/img2sem3d/' # None: no output

img = Image.open(img_path)
##############################################
# (1a) occlusion masking
##############################################
start_total = time.time()
start = time.time()
occ_masks = generator_sem2d.generate_2Dsem(
    img, semantic_classes=cfg_bev.OCCLUSION_CLASSES
)
occ_mask = np.logical_or.reduce(list(occ_masks.values()))
occ_mask = (occ_mask * 255).astype(np.uint8) # Convert to 0-255 uint8 format
occ_mask = occ_mask[0] # (1,H,W) => (H,W)
occ_mask = Image.fromarray(occ_mask, mode='L')
end = time.time()
print(colored(f'Occlusion Mask: {round(end-start, 3)} s', 'green'))
##############################################
# (1b) occlusion clearance: diffusion-based
##############################################
start = time.time()
images = [img]
masks = [occ_mask]
occ_free_images = generator_clear.clear_2d_occlusion(images, masks)
img = occ_free_images[0]
# save occlusion-free image to tmp dir
tmp_dir = f'{pwd}/../../tmp/'
os.makedirs(tmp_dir, exist_ok=True)
unique = uuid.uuid4().hex
path_img_clear = os.path.join(tmp_dir, f"{unique}.jpg")
img.save(path_img_clear) # temp
end = time.time()
print(colored(f'Diffusion: {round(end-start, 3)} s', 'green'))

##############################################
# (2a) 2D semantic mask (img_clear -> sem2d)  
##############################################
start = time.time()
mask_dict = generator_sem2d.generate_2Dsem(
    img, semantic_classes=cfg_bev.SEMANTIC_CLASSES
)
end = time.time()
print(colored(f'Semantic 2D runtime: {round(end-start, 3)} s', 'green'))

##############################################
# (2b) Metric Depth + Intrinsincs (img_clear -> K,D)
##############################################
start = time.time()
D, K = generator_3d.generate_3Dpc([path_img_clear])
H_d, W_d = D.shape
end = time.time()
print(colored(f'Metric Depth runtime: {round(end-start, 3)} s', 'green'))
os.remove(path_img_clear)


##############################################
# (3) 3D unprojection: 
#       (img_clear, sem2d, K, D -> points3D)
##############################################
img = np.asarray(img)
img = cv2.resize(img, (W_d, H_d), interpolation=cv2.INTER_LINEAR) # resize to depth
num_classes = len(mask_dict)
assert len(mask_dict) == len(cfg_bev.SEMANTIC_CLASSES)
# Match shape of the mask to the depth
for idx, semantic_class in enumerate(mask_dict):
    # (1,H,W) -> (H,W) -> (H_d, W_d)
    print(f'\tSemantic class {idx+1}: {semantic_class}')
    mask = mask_dict[semantic_class].astype(np.float32)[0]
    mask = cv2.resize(mask, (W_d, H_d), interpolation=cv2.INTER_LINEAR)
    mask_dict[semantic_class] = (mask >= cfg_bev.SEG_THRESHOLD).astype(np.float32)
# 3D point reconstruction
pc_grid, sem_grid, valid_indexes = img_reconstruct_3D_points(
    img, mask_dict, K, D,
    xlim=cfg_bev.xlim, zlim=cfg_bev.zlim, 
    # ylim (height) is determined automatically based on cfg_bev.BEV_HEIGHT
)
# extract points of interest
points = pc_grid.reshape(-1,3)[valid_indexes]
semantics = sem_grid.reshape(-1,1)[valid_indexes]
end_total = time.time()
print(colored(f'Total E2E: {end_total-start_total} s.', 'green'))

# visualization
if vis_path is not None:
    os.makedirs(vis_path, exist_ok=True)
    # original image
    img_orig = Image.open(img_path).resize((W_d, H_d))
    img_orig.save(os.path.join(vis_path, '0_original.jpg'))
    # intermediate outputs
    visualize_outputs(
        img, D, 
        pc_grid, sem_grid, None,
        num_classes, valid_indexes,
        vis_path,
    )

