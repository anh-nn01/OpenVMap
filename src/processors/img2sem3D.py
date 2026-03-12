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
import time
import uuid

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
from generate_bev_grid import img_reconstruct_3D_points


# Initialize generators
generator_clear = OcclusionFreeGenerator()
generator_sem2d = Semantic2DGenerator(threshold=cfg_bev.SEG_THRESHOLD)
generator_3d = PointCloud3DGenerator()

""" TODO: list of img paths"""
img_path = '/fs/nexus-projects/open_vectormap/src/examples/nusc/img_2/n008-2018-05-21-11-06-59-0400__CAM_FRONT__1526915292912465.jpg'

img = Image.open(img_path)

# (1a) occlusion masking
start = time.time()
occ_masks = generator_sem2d.generate_2Dsem(
    img, semantic_classes=cfg_bev.OCCLUSION_CLASSES
)
occ_mask = np.logical_or.reduce(list(occ_masks.values()))
occ_mask = (occ_mask * 255).astype(np.uint8) # Convert to 0-255 uint8 format
occ_mask = occ_mask[0] # (1,H,W) => (H,W)
occ_mask = Image.fromarray(occ_mask, mode='L')
end = time.time()
print(f'Occlusion Mask: {round(end-start, 3)} s')
# (1b) occlusion clearance: diffusion-based
start = time.time()
images = [img]
masks = [occ_mask]
occ_free_images = generator_clear.clear_2d_occlusion(images, masks)
img = occ_free_images[0]
# save occlusion-free image to tmp dir
os.makedirs(f'{pwd}/tmp/', exist_ok=True)
unique = uuid.uuid4().hex
path_img_clear = os.path.join(f'{pwd}/tmp/', f"{unique}.jpg")
img.save(path_img_clear) # temp
end = time.time()
print(f'Diffusion: {round(end-start, 3)} s')

# (2a) 2D semantic mask (img_clear -> sem2d)  
start = time.time()
masks = generator_sem2d.generate_2Dsem(
    img, semantic_classes=cfg_bev.SEMANTIC_CLASSES
)
end = time.time()
print(f'Semantic 2D runtime: {round(end-start, 3)} s')

# (2b) Metric Depth + Intrinsincs (img_clear -> K,D)
start = time.time()
depth, K = generator_3d.generate_3Dpc([path_img_clear])
end = time.time()
print(f'Metric Depth runtime: {round(end-start, 3)} s')
os.remove(path_img_clear)