##################################################################
# Author: anh-nn01; Feb 09, 2026
# Description: 
#   Generate BEV grid voxels from monocular image
#   combining 2D semantics and 3D point cloud
##################################################################

# [NOTE] use conda venv for DA3 for this script

import os
import sys
import argparse
pwd = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import torch
import trimesh
import cv2
import matplotlib.pyplot as plt
from PIL import Image

sys.path.append(f"{pwd}/../configs")
from bev import SEMANTIC_CLASSES

def load_semantic_masks(path):
    data = np.load(path, allow_pickle=True)
    semantic_masks = {}
    # fill in masks
    for k in data.files:
        m = data[k]
        semantic_masks[k] = m
        # print(semantic_masks[k].shape)
    return semantic_masks

def load_points_from_glb(path):
    scene = trimesh.load(path)
    pcd = scene.geometry[list(scene.geometry.keys())[0]]
    points = np.asarray(pcd.vertices)
    colors = pcd.visual.vertex_colors[:, :3] # ignore alpha

    return points, colors

def load_intrinsics(path):
    K = np.load(path)
    return K

def load_depth(path):
    depth = np.load(path)
    return depth



def extract_points(points, xlim=None, ylim=None, zlim=None):
    valid = True
    if xlim: # x limit: lateral
        xmin, xmax = xlim
        valid = valid & (points[:,0] >= xmin) & (points[:,0] <= xmax)
    if ylim: # y limit: height
        ymin, ymax = ylim
        valid = valid & (points[:,1] >= ymin) & (points[:,1] <= ymax)
    if zlim: # z limit: forward
        zmin, zmax = zlim
        valid = valid & (points[:,2] >= zmin) & (points[:,2] <= zmax)
    return valid



# ###############################################
# # Match shape of the mask to the depth
# ###############################################
# for semantic_class in mask_dict:
#     # (1,H,W) -> (H,W) -> (H_d, W_d)
#     mask = mask_dict[semantic_class].astype(np.float32)[0]
#     mask = cv2.resize(mask, (W_d, H_d), interpolation=cv2.INTER_LINEAR)
#     mask_dict[semantic_class] = mask
# ###############################################
# # Unproject semantic mask to 3D
# ###############################################
# # debug: focus on "drivable road area" only
# semantic_class = "drivable road area"
# mask = mask_dict[semantic_class]


    


# points, colors = load_points_from_glb(path_points)
# print(points.shape)
# print('Dimension 0:', points[:,0].min(), points[:,0].max())
# print('Dimension 1:', points[:,1].min(), points[:,1].max())
# print('Dimension 2:', points[:,2].min(), points[:,2].max())