##################################################################
# Author: anh-nn01; Feb 09, 2026
# Description: 
#   Generate BEV grid voxels from monocular image
#   combining 2D semantics and 3D point cloud
##################################################################

# [NOTE] use conda venv for DA3 for this script

""" Some notes on the designs of local processing"""
# [NOTE] adjust 3D coordinates by avg ground height (drivable lane index = 1)
#        => avg Y of (all points associated with (1) drivable lane and (2) in point cloud of interest)
#        => more generalizable to different sensor setups
# [NOTE] points: filter out noisy 3D points
# [NOTE] bev_voxel: max pool along Y (height) first, then along XZ
# [NOTE] bev_voxel: post-processing: use ZX kernel size = (5)



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
import open3d as o3d

sys.path.append(f"{pwd}/../configs")
import cfg_bev

# valid class index +=1 (0 = unknown class)
road_class_idx = cfg_bev.SEMANTIC_CLASSES.index('road')+1 # 'drivable road area'
lane_marking_class_idx = cfg_bev.SEMANTIC_CLASSES.index('lane marking')+1
crosswalk_class_idx = cfg_bev.SEMANTIC_CLASSES.index('crosswalk area')+1

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

    
def statistical_filter_3Dnoises(points, nb_neighbors=20, std_ratio=2.0):
    """ NOTE: segmentation fault when np>=2.0"""
    # Convert numpy to Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    points = np.ascontiguousarray(points, dtype=np.float64)
    pcd.points = o3d.utility.Vector3dVector(points)
    # cl: filtered pcd, ind: list of inlier indices
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, 
                                             std_ratio=std_ratio)
    # Create boolean mask (default False)
    mask = np.zeros(len(points), dtype=bool)
    mask[ind] = True
    return mask

def get_radius_filter_mask(points, nb_points=8, radius=0.5):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # cl: filtered pcd, ind: list of inlier indices
    cl, ind = pcd.remove_radius_outlier(nb_points=nb_points, 
                                        radius=radius)
    mask = np.zeros(len(points), dtype=bool)
    mask[ind] = True
    return mask



def img_reconstruct_3D_points(
        img, mask_dict, K, D, xlim=None, ylim=None, zlim=None,
    ):
    """
    Function to reconstruct a 3D point cloud from a monocular image.

    Inputs:
        + img: input monocular image
        + mask_dict: dictionary of semantic masks
            format: {<class_name>: np.array(H, W)}
        + K: camera intrinsic matrix
        + D: metric depth map
        + xlim: lateral range of interest for the point cloud (x-axis)
        + ylim: height range of interest for the point cloud (y-axis)
        + zlim: forward range of interest for the point cloud (z-axis)

    Outputs: (pc_grid, sem_grid, valid_indexes)

        + pc_grid: 3D position corresponding to each pixel
            shape: (H, W, 3)

        + sem_grid: semantic class index for each pixel
            shape: (H, W, 1)
            note:
                class indices start at 1
                index 0 = pixels without a semantic class

        + valid_indexes: indices of pixels whose reconstructed 3D points
            fall within the specified (xlim, ylim, zlim) ranges

            usage:
                points = pc_grid.reshape(-1, 3)[valid_indexes]
                semantics = sem_grid.reshape(-1, 1)[valid_indexes]
    """
    num_classes = len(mask_dict)
    H_d, W_d = D.shape

    """ (I) Construct 3D point cloud and semantic maps """
    # ============================================================
    # (1a) Build pixel coordinate grid
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
    # (1b) Back-project depth map to 3D camera coordinates
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
    # (1c) Assign semantic labels to points
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




    """ (II) Extract points of interest """
    points = pc_grid.reshape(-1,3) # points in (HxW, 3)
    semantics = sem_grid.reshape(-1,1) # semantic class in (HxW, 1)
    # ============================================================
    # (2a) Compute average ground height in camera pose
    #       => adaptively define ylim (height)
    # ============================================================
    points_ground = points[semantics[:,0] == road_class_idx] # drivable lanes
    # points_ground = points_ground[points_ground[:,2] <= 10] # within 5 meter depth
    avg_ground_height = points_ground[:,1].mean() # average ground height
    print('\nAvg ground height in cam frame:', avg_ground_height.round(2), '[m]')
    # Construct height limit
    if not ylim:
        ylim = (avg_ground_height-1, avg_ground_height + cfg_bev.BEV_HEIGHT)
    # ============================================================
    # (2b) Filter out 3D points based on point cloud range
    # ============================================================
    # filter points based on point cloud range
    valid_indexes = extract_points(points, xlim, ylim, zlim)
    # ============================================================
    # (2c) Filter out noisy 3D points
    # ============================================================
    # # filter_noise_indexes = statistical_filter_3Dnoises(points)
    # filter_noise_indexes = get_radius_filter_mask(points)
    # valid_indexes = valid_indexes & filter_noise_indexes

    return pc_grid, sem_grid, valid_indexes

def construct_bev_voxels(
        points, semantics, voxel_size,
        xlim, zlim,
    ):
    """
        Construct BEV grid voxels from semantic point cloud

            points: 3D point cloud sets (N_,3)
            semantics: associated semantic class sets (N_,1)
            voxel_size: voxel size in metric [m]
            xlim: lateral limit [m]
            zlim: longitudinal (forward) limit [m]
            
        Output: bev voxels filled with semantic class indexes
                default: -1 = no point matched (e.g. occlusion)
                
    """
    # 1. Filter points within bounds
    mask = (points[:, 0] >= xlim[0]) & (points[:, 0] < xlim[1]) & \
           (points[:, 2] >= zlim[0]) & (points[:, 2] < zlim[1])
    pts = points[mask].copy()
    labels = semantics[mask].flatten()

    if len(pts) == 0:
        width = int(np.ceil((xlim[1] - xlim[0]) / voxel_size))
        depth = int(np.ceil((zlim[1] - zlim[0]) / voxel_size))
        return np.full((width, depth), -1, dtype=np.int32)

    # 2. Map to discrete grid indices
    # We calculate indices FIRST before inversion to avoid math errors
    x_indices = ((pts[:, 0] - xlim[0]) / voxel_size).astype(np.int32)
    z_indices = ((pts[:, 2] - zlim[0]) / voxel_size).astype(np.int32)

    # 3. Calculate Dimensions
    width = int(np.ceil((xlim[1] - xlim[0]) / voxel_size))
    depth = int(np.ceil((zlim[1] - zlim[0]) / voxel_size))

    # 4. Max Pooling along Height (Axis=1)
    # lexsort sorts by x, then z, then labels (labels is the primary sort key)
    # This places the HIGHEST label at the end of each (x, z) group
    sort_idx = np.lexsort((labels, z_indices, x_indices))
    x_s, z_s, l_s = x_indices[sort_idx], z_indices[sort_idx], labels[sort_idx]
    # Use unique on combined (x, z) to find the LAST occurrence (the max label)
    combined_idx = x_s.astype(np.int64) * depth + z_s
    _, first_unique_reversed = np.unique(combined_idx[::-1], return_index=True)
    unique_indices = (len(combined_idx) - 1) - first_unique_reversed

    # 5. Fill Grid
    bev_grid = np.full((width, depth), -1, dtype=np.int32)
    bev_grid[x_s[unique_indices], z_s[unique_indices]] = l_s[unique_indices]
    # 6. Apply Lateral Inversion 
    #   (original 3D PC: x-axis points to the left => invert to the right)
    bev_grid = np.flip(bev_grid, axis=0)
    return bev_grid


    
def visualize_outputs(
        img, depth, 
        pc_grid, sem_grid, bev_voxels,
        num_classes, valid_indexes,
        output_path,
    ):
    """ 
        Visualization:
        
            img: input image (H,W,3) (rgb)
            depth: estimated metric depth (H,W,1) (metric depth)
            pc_grid: unprojected 3D point cloud grids (H,W,3) ([X,Y,Z])
            sem_grid: multi-class semantic mask (H,W,1) (sematic class)
            valid_indexes: extracted points of interest 
            output_path: output visualization path
    """
    points = pc_grid.reshape(-1,3)
    colors = img.reshape(-1, 3) # Point colors from RGB image
    semantics = sem_grid.reshape(-1,1) # Point semantic classes
    
    # semantic colors for visualization
    cmap = plt.get_cmap('tab20b', num_classes+1)
    palette = (cmap(np.arange(num_classes + 1))[:, :3] * 255).astype(np.uint8)
    palette[0] = np.array([128, 128, 128], dtype=np.uint8)              # index 0 = gray: unmatched class
    palette[road_class_idx] = np.array([10, 128, 10], dtype=np.uint8)           # index 1 = green: drivable areas
    palette[lane_marking_class_idx] = np.array([255, 165, 10], dtype=np.uint8)  # index 3 = orange: lane marking
    palette[crosswalk_class_idx] = np.array([10, 10, 255], dtype=np.uint8)     # index 4 = orange: cross walk area
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
    os.system(f'mkdir -p {output_path}')
    vis = plt.imshow(img)
    plt.savefig(f'{output_path}/1_example_obs.png', bbox_inches='tight')
    plt.close()
    # **************************************
    # Visualize depth map
    # **************************************
    vis = plt.imshow(depth, cmap='Spectral', vmin=5, vmax=50)
    plt.colorbar(vis, label='Depth [meters]', orientation='horizontal',)
    plt.savefig(f'{output_path}/2_example_depth.png', bbox_inches='tight')
    plt.close()
    # **************************************
    # Visualize semantic map
    # **************************************
    custom_cmap = ListedColormap(palette / 255.0)
    vis = plt.imshow(sem_visualization, cmap=custom_cmap)
    plt.colorbar(vis, ticks=np.arange(num_classes + 1), label='semantic colors', orientation='horizontal',)
    plt.savefig(f'{output_path}/3_example_semantic.png', bbox_inches='tight')
    plt.close()
    # **************************************
    # visualize unprojected 3D point clouds
    # **************************************
    print('Point shape (after filtering):', points.shape)
    print('\tX range [m] (lateral):', points[:,0].min().round(2), points[:,0].max().round(2))
    print('\tY range [m] (height) :', points[:,1].min().round(2), points[:,1].max().round(2))
    print('\tZ range [m] (forward):', points[:,2].min().round(2), points[:,2].max().round(2))
    # point cloud
    pc = trimesh.points.PointCloud(vertices=points)
    pc.export(f'{output_path}/4_example_pc.glb')
    pc = trimesh.points.PointCloud(vertices=points, colors=colors)
    pc.export(f'{output_path}/5_example_pc_colored.glb')
    pc = trimesh.points.PointCloud(vertices=points, colors=sem_colors)
    pc.export(f'{output_path}/6_example_pc_semantic.glb')
    # **************************************
    # visualize BEV voxels
    # **************************************
    if bev_voxels is not None:
        bev_voxels = np.ma.masked_where(bev_voxels == -1, bev_voxels)
        vis = plt.imshow(
            bev_voxels.T, # Transpose to align with (X, Z) expectations
            origin='lower',
            extent=[cfg_bev.xlim[0], cfg_bev.xlim[1], cfg_bev.zlim[0], cfg_bev.zlim[1]],
            cmap=custom_cmap,
            interpolation='nearest'
        )
        plt.colorbar(vis, ticks=np.arange(num_classes + 1), label='semantic colors', orientation='horizontal',)
        plt.axis('equal')
        plt.savefig(f'{output_path}/7_example_bev.png', bbox_inches='tight')
        plt.close()

    


if __name__ == '__main__': # for demo purposes
    # debug_func(xlim=cfg_bev.xlim, zlim=cfg_bev.zlim)
    #############################
    # Debugging
    #############################
    # output_path = 'debug_outputs/initial/'
    # eg_id = 6
    # path_img = '../examples/nusc/img_2/n008-2018-05-21-11-06-59-0400__CAM_FRONT__1526915292912465.jpg'
    # path_img = '../examples/nusc/img_4/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151605512404.jpg'
    # path_img = '../examples/nusc/img_3/n008-2018-08-01-15-52-19-0400__CAM_FRONT__1533153350162404.jpg'
    # path_img = '../examples/nusc/img_5/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402942162460.jpg'
    # path_img = '../examples/nusc/img_6/val_front_275.jpg'
    # path_masks = f'{pwd}/../examples/nusc/img_{eg_id}/semantic_masks.npz'
    # path_intrinsic =  f'{pwd}/../examples/nusc/img_{eg_id}/da3_output/intrinsics.npy'
    # path_depth =  f'{pwd}/../examples/nusc/img_{eg_id}/da3_output/depth.npy' 

    
    output_path = 'debug_outputs/occfree/'
    eg_id = 5
    path_img = f'../examples/nusc/img_{eg_id}/img_occlusion_free.jpg'
    path_masks = f'{pwd}/../examples/nusc/img_{eg_id}/semantic_masks.npz'
    path_intrinsic =  f'{pwd}/../examples/nusc/img_{eg_id}/da3_output_occfree/intrinsics.npy'
    path_depth =  f'{pwd}/../examples/nusc/img_{eg_id}/da3_output_occfree/depth.npy' 

    """ (I) Load perception inputs """
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
        mask_dict[semantic_class] = (mask >= cfg_bev.SEG_THRESHOLD).astype(np.float32)
    # ============================================================
    # (1d) Load RGB image (resized to depth resolution)
    # ============================================================
    img = Image.open(path_img)
    img = np.asarray(img)
    img = cv2.resize(img, (W_d, H_d), interpolation=cv2.INTER_LINEAR)



    # ============================================================
    # (2) 3D point cloud reconstruction
    # ============================================================
    pc_grid, sem_grid, valid_indexes = img_reconstruct_3D_points(
        img, mask_dict, K, D,
        xlim=cfg_bev.xlim, zlim=cfg_bev.zlim, 
        # ylim (height) is determined automatically based on cfg_bev.BEV_HEIGHT
    )
    # extract points of interest
    points = pc_grid.reshape(-1,3)[valid_indexes]
    semantics = sem_grid.reshape(-1,1)[valid_indexes]

    # ============================================================
    # (3) Construct semantic BEV semantic voxels from 3D pc
    # ============================================================
    bev_voxels = construct_bev_voxels(
        points, semantics, voxel_size=cfg_bev.voxel_size,
        xlim=cfg_bev.xlim, zlim=cfg_bev.zlim,
    )

    # **************************************
    # Visualization: must be here
    # **************************************
    visualize_outputs(
        img, D, 
        pc_grid, sem_grid, bev_voxels,
        num_classes, valid_indexes,
        output_path,
    )

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