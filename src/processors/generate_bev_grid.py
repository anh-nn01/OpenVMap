##################################################################
# Author: anh-nn01; Feb 09, 2026
# Description: 
#   Generate BEV grid voxels from monocular image
#   combining 2D semantics and 3D point cloud
##################################################################

# [NOTE] use conda venv for DA3 for this script
# [NOTE] adjust 3D coordinates by avg ground height (drivable lane index = 1)
#        => avg Y of (all points associated with (1) drivable lane and (2) <= 5 meters of depth)
#        => more generalizable to different sensor setups

import os
import sys
import argparse
pwd = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import torch
import trimesh
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from PIL import Image

sys.path.append(f"{pwd}/../configs")
import cfg_bev

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


def debug_func(xlim=None, ylim=None, zlim=None):
    #############################
    # Debugging
    #############################
    eg_id = 3
    # path_img = '../examples/nusc/eg_4/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151605512404.jpg'
    path_img = '../examples/nusc/eg_3/n008-2018-08-01-15-52-19-0400__CAM_FRONT__1533153350162404.jpg'
    path_masks = f'{pwd}/../examples/nusc/eg_{eg_id}/semantic_masks.npz'
    path_intrinsic =  f'{pwd}/../examples/nusc/eg_{eg_id}/da3_output/intrinsics.npy'
    path_depth =  f'{pwd}/../examples/nusc/eg_{eg_id}/da3_output/depth.npy' 
    # path_points = f'{pwd}/../examples/nusc/eg_5/da3_output/scene.glb'

    # ============================================================
    # (1a) Load camera intrinsics
    #   K: (3, 3) camera intrinsic matrix
    # ============================================================
    K = load_intrinsics(path_intrinsic)
    # ============================================================
    # (1b) Load depth map
    #   D: (H_d, W_d) depth in meters (Z-axis depth)
    # ============================================================
    D = load_depth(path_depth)
    H_d, W_d = D.shape
    print('Depth spatial shape:', H_d, W_d)
    print('Depth min/max [m]:', D.min(), D.max())
    # ============================================================
    # (1c) Load semantic masks
    #   mask_dict: { <class_name> : np.array }
    # ============================================================
    mask_dict = load_semantic_masks(path_masks)
    num_classes = len(mask_dict)
    assert len(mask_dict) == len(cfg_bev.SEMANTIC_CLASSES)
    first_key = next(iter(mask_dict))
    H_m, W_m = mask_dict[first_key].shape[1:]
    print('Mask spatial shape :', H_m, W_m)
    # Match shape of the mask to the depth
    for idx, semantic_class in enumerate(mask_dict):
        # (1,H,W) -> (H,W) -> (H_d, W_d)
        print(f'\tSemantic class {idx+1}: {semantic_class}')
        mask = mask_dict[semantic_class].astype(np.float32)[0]
        mask = cv2.resize(mask, (W_d, H_d), interpolation=cv2.INTER_LINEAR)
        mask_dict[semantic_class] = (mask >= 0.5).astype(np.float32)
    # ============================================================
    # (1d) Load RGB image (resized to depth resolution)
    # ============================================================
    img = Image.open(path_img)
    img = np.asarray(img)
    img = cv2.resize(img, (W_d, H_d), interpolation=cv2.INTER_LINEAR)
    
    
    
    # ============================================================
    # (2a) Build pixel coordinate grid
    #   uv_grid: (H, W, 2)
    #   uv_grid[v, u] = [u, v]
    #
    # Convention:
    #   u -> horizontal axis (width, x-direction in image)
    #   v -> vertical axis   (height, y-direction in image)
    # ============================================================
    u = np.arange(W_d) # X-axis
    v = np.arange(H_d) # Y-axis
    u_grid, v_grid = np.meshgrid(u, v)
    # uv_grid[v, u] -> [u, v]
    uv_grid = np.stack([u_grid, v_grid], axis=-1) # # each pixel store (u,v) pixel coordinate
    # print(uv_grid.shape)
    # print(uv_grid[95,98])
    # ============================================================
    # (2b) Back-project depth map to 3D camera coordinates
    #
    # Formula:
    #   [X,Y,Z]^T = D(u,v) * (K^{-1} @ [u,v,1]^T)
    #
    # Steps:
    #   1. Convert (u,v) -> homogeneous coordinates
    #   2. Compute normalized camera rays via K^{-1}
    #   3. Scale rays by depth
    # ============================================================
    # homogeneous coordinate
    uv_grid = np.concatenate([uv_grid, np.ones((H_d, W_d, 1))], axis=-1)
    uv_rays = np.linalg.inv(K) @ uv_grid.reshape(-1, 3).T # (u,v) 3D rays
    pc_grid = D.reshape(1,-1) * uv_rays # depth unprojection
    pc_grid = pc_grid.T # (3,N) -> (N,3)
    pc_grid = pc_grid.reshape((H_d, W_d, 3)) # each pixel store unprojected (X,Y,Z) coordinate
    # invert X-axis (lateral) and Y-axis (height) to virtual image space
    pc_grid[:,:,0] *=-1
    pc_grid[:,:,1] *=-1
    # ============================================================
    # (2c) Assign semantic labels to points
    #   sem_grids: (H,W,1)
    #   sem_grids[u,v] = <semantic class index>
    # 
    # Step: (1) match each point to a semantic class
    #           => tie break: assign highest index
    #       (2) +=1: valid semantic index starts at 1
    #                => 0 = unassigned index
    # ============================================================
    # Stack all mask across classes: (N_class, H_d, W_d)
    masks = np.stack([mask_dict[semantic_class] for semantic_class in mask_dict], axis=0)
    # (N_class, H_d, W_d) -> (1, H_d, W_d)
    # semantic class tie break: use the class at the higher index
    #   e.g. a pixel having both  class 0 (drivable) and class 3 (crosswalk), use class 3 (crosswalk)
    sem_grid_reverse_idx = np.argmax(masks[::-1], axis=0) 
    sem_grid = (num_classes-1) - sem_grid_reverse_idx # use original index order
    sem_grid = masks.max(0).astype(np.uint8) * (sem_grid + 1) # semantic index starts at 1; 0 = unmatched class
    # print(sem_grids.shape)
    # ============================================================
    # (2d) Filter out 3D points based on point cloud range
    # ============================================================
    points = pc_grid.reshape(-1,3) # points in (HxW, 3)
    semantics = sem_grid.reshape(-1,1) # semantic class in (HxW, 1)
    # filter points based on point cloud range
    valid_indexes = extract_points(points, xlim, ylim, zlim)
    # **************************************
    # Visualization: must be here
    # **************************************
    visualize_outputs(
        img, D, 
        sem_grid, num_classes,
        points, valid_indexes
    )
    # filter points
    points = points[valid_indexes]
    semantics = semantics[valid_indexes]




    # ============================================================
    # (3a) Compute average ground height in camera pose
    #       => define ylim
    # ============================================================
    # filter ground points
    points_ground = points[semantics[:,0] == 1] # drivable lanes
    # points_ground = points_ground[points_ground[:,2] <= 10] # within 5 meter depth
    avg_ground_height = points_ground[:,1].mean() # average ground height
    print('\nAvg ground height in cam frame:', avg_ground_height.round(2), '[m]')
    # Construct height limit
    if not ylim:
        ylim = (avg_ground_height-1, avg_ground_height + cfg_bev.BEV_HEIGHT)
    # ============================================================
    # (3b) Construct semantic BEV semantic voxels from 3D pc
    # ============================================================
    bev_voxels = construct_bev_voxels(
        points, semantics, grid_size=cfg_bev.voxel_size,
        xlim=xlim, ylim=ylim, zlim=zlim
    )

def construct_bev_voxels(
        points, semantics, grid_size, xlim=None, ylim=None, zlim=None
    ):
    """
        Construct BEV grid voxels from semantic point cloud

            points: 3D point cloud sets
            semantics: associated semantic class sets
            grid_size: voxel size in metric [m]
            xlim: point cloud range in metric [m]
            ylim: point cloud range in metric [m]
            zlim: point cloud range in metric [m]
            
        Output: bev voxels filled with semantic class indexes
                default: -1 = no point matched (e.g. occlusion)
    """
    bev_grid = None
    pass

    
def visualize_outputs(
        img, depth, 
        sem_grid, num_classes,
        points, valid_indexes
    ):
    """ 
        Visualization:
        
            img: input image (H,W,3)
            depth: estimated metric depth (H,W,1)
            sem_grid: multi-class semantic mask (H,W,1)
            points: unprojected 3D points (HxW,3)
            valid_indexes: extracted points of interest 
    """
    colors = img.reshape(-1, 3) # Point colors from RGB image
    semantics = sem_grid.reshape(-1,1) # Point semantic classes
    
    # semantic colors for visualization
    cmap = plt.get_cmap('tab20b', num_classes+1)
    palette = (cmap(np.arange(num_classes + 1))[:, :3] * 255).astype(np.uint8)
    palette[0] = np.array([128, 128, 128], dtype=np.uint8) # index 0 = gray: unmatched class
    palette[1] = np.array([10, 128, 10], dtype=np.uint8) # index 1 = green: drivable areas
    sem_colors = palette[semantics].reshape(-1,3)  # (N, 3)
    H_d, W_d = depth.shape[:2]
    sem_visualization = sem_colors.reshape((H_d, W_d, 3))
    
    # filter points of interest
    points = points[valid_indexes]
    colors = colors[valid_indexes]
    semantics = semantics[valid_indexes]
    sem_colors = sem_colors[valid_indexes]
    # print(points.shape)
    # print(semantics.shape)

    # **************************************
    # Visualize input image
    # **************************************
    vis = plt.imshow(img)
    plt.savefig('debug_outputs/1_example_obs.png', bbox_inches='tight')
    plt.close()
    # **************************************
    # Visualize depth map
    # **************************************
    vis = plt.imshow(depth, cmap='Spectral', vmin=5, vmax=50)
    plt.colorbar(vis, label='Depth [meters]', orientation='horizontal',)
    plt.savefig('debug_outputs/2_example_depth.png', bbox_inches='tight')
    plt.close()
    # **************************************
    # Visualize semantic map
    # **************************************
    custom_cmap = ListedColormap(palette / 255.0)
    vis = plt.imshow(sem_visualization, cmap=custom_cmap)
    plt.colorbar(vis, ticks=np.arange(num_classes + 1), label='semantic colors', orientation='horizontal',)
    plt.savefig('debug_outputs/3_example_semantic.png', bbox_inches='tight')
    plt.close()
    # **************************************
    # visualize unprojected 3D point clouds
    # **************************************
    os.system('mkdir -p debug_outputs')
    print('Point shape (after filtering):', points.shape)
    print('\tX range [m] (lateral):', points[:,0].min().round(2), points[:,0].max().round(2))
    print('\tY range [m] (height) :', points[:,1].min().round(2), points[:,1].max().round(2))
    print('\tZ range [m] (forward):', points[:,2].min().round(2), points[:,2].max().round(2))
    # point cloud
    pc = trimesh.points.PointCloud(vertices=points)
    pc.export('debug_outputs/4_example_pc.glb')
    pc = trimesh.points.PointCloud(vertices=points, colors=colors)
    pc.export('debug_outputs/5_example_pc_colored.glb')
    pc = trimesh.points.PointCloud(vertices=points, colors=sem_colors)
    pc.export('debug_outputs/6_example_pc_semantic.glb')


debug_func(xlim=cfg_bev.xlim, zlim=cfg_bev.zlim)

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