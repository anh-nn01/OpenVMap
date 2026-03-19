##################################################################
# Author: anh-nn01; March 11, 2026
# Description: 
#   Vectorize semantic 3D point cloud
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
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import ConvexHull, Delaunay, KDTree, cKDTree
import alphashape
from shapely.geometry import MultiPoint

# configs
sys.path.append(f"{pwd}/../configs")
import cfg_bev

# valid class index +=1 (0 = unknown class)
road_class_idx = cfg_bev.road_class_idx # 'drivable road area'
lane_div_class_idx = cfg_bev.lane_div_class_idx
lane_marking_class_idx = cfg_bev.lane_marking_class_idx
crosswalk_class_idx = cfg_bev.crosswalk_class_idx



# def vectorize_map(points, semantics):
#     """ Vectorize BEV maps given semantic 3D point cloud
#         TODO: point cloud sparsification for speed up 

#         Args:
#             points: 3D point cloud
#             semantics: corresponding semantics
#         Outputs:
#             polyline_dict: vectorized polyline for each type
#     """
#     polylines_dict = {
#         'boundary': None,
#         'crosswalk': None,
#     }
    

#     #############################
#     # Vectorize crosswalk area  #
#     #############################
#     start = time.time()
#     polylines = vectorize_convex_crosswalk(points, semantics)
#     polylines_dict['crosswalk'] = polylines
#     end = time.time()
#     print(colored(f'Crosswalk vectorization: {round(end-start, 5)} s', 'green'))

#     #############################
#     # Vectorize drivable area   #
#     #############################
#     start = time.time()
#     polylines = vectorize_drivable_boundaries(points, semantics)
#     polylines_dict['boundary'] = polylines
#     end = time.time()
#     print(colored(f'Drivable areas vectorization: {round(end-start, 5)} s', 'green'))


#     return polylines_dict


def sparsify_points(points, grid_size):
    """
    Downsample points by keeping one point per grid cell.
    
    Args:
        points: (N, 2) or (N, 3) array of coordinates
        grid_size: float, the size of each grid cell (e.g., 0.1 for 10cm)
    Outputs:
        downsampled_points: (M, D) array where M <= N
    """
    # 1. Quantize the coordinates by dividing by grid_size and flooring
    grid_indices = np.floor(points / grid_size).astype(int)
    # 2. Find unique grid cells. return_index=True gives the first point in each cell.
    _, unique_indices = np.unique(grid_indices, axis=0, return_index=True)
    return points[unique_indices]

def densify_points(points, semantics, grid_size, k=5):
    """ 
    Semantic-grounded points upsampling using K-nearest neighbor (KD-Tree)

    Args: 
        points: (N, 2) or (N, 3) array of coordinates
        semantics: (N, 1) or (N,) array of semantic index
        grid_size: float, the size of each grid cell (e.g., 0.1 for 10cm)
        k: number of neighbors used for semantic voting

    Outputs:
        upsample_points: (M, D) array where M >= N
        upsample_semantics: (M, 1) array of semantic labels
    """
    semantics = semantics.squeeze() # (N,)
    # 1. Define the bounding box of the points
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    # 2. Create a dense grid of coordinates within that box
    axes = [np.arange(min_b, max_b + grid_size, grid_size) 
            for min_b, max_b in zip(min_bound, max_bound)]
    grid_coords = np.stack(np.meshgrid(*axes), -1).reshape(-1, points.shape[1])
    # 3. KDTree: find K nearest original points for each grid coordinate
    tree = KDTree(points)
    _, knn_idx = tree.query(grid_coords, k=k)
    upsampled_points = grid_coords
    # 4. Semantic voting among the K nearest neighbors
    knn_labels = semantics[knn_idx]
    if k == 1:
        upsampled_semantics = knn_labels
    else:
        # majority vote
        upsampled_semantics = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), axis=1, arr=knn_labels
        )
    upsampled_semantics = upsampled_semantics.reshape(-1, 1) # (M,1)
    return upsampled_points, upsampled_semantics


# def transform_point_density(points, semantics, grid_size):
#     """ 
#     Semantic-grounded point density transformation using nearest neighbor

#     Args: 
#         points: (N, 2) or (N, 3) array of coordinates
#         semantics: (N, 1) array of semantic index
#         grid_size: float, the size of each grid cell (e.g., 0.1 for 10cm)
#     Outputs:
#         upsample_points: (M, D)
#     """
#     # 1. Snap points to the nearest grid center
#     # Adding 0.5 ensures we snap to the middle of the cell, not the corner
#     grid_indices = np.round(points / grid_size)
#     grid_centers = grid_indices * grid_size
#     # 2. Find unique grid centers to ensure uniform density (one point per cell)
#     # return_index=True picks the nearest original semantic label for that cell
#     _, unique_indices = np.unique(grid_indices, axis=0, return_index=True)
#     upsampled_points = grid_centers[unique_indices]
#     upsampled_semantics = semantics[unique_indices]

#     return upsampled_points, upsampled_semantics


def vectorize_convex_crosswalk(points, semantics):
    """ Vectorize crosswalk areas from the point cloud
        Geometry: DBSCAN + Chan's Convex Hull Algorithm

        TODO: minimize number of points
        TODO: should inherently cover non-convexity due to occlusions?
        TODO: test difficult cases (heavy occlusion)
        
        Args:
            points: 3D point cloud
            semantics: corresponding semantics
        Outputs:
            polylines: polyline of crosswalk areas
    """
    # # ===============================================
    # # (0) Transform point density for 
    # #   proper nearest-neighbor-based boundary matching
    # # ===============================================
    # points, semantics = transform_point_density(points, semantics, grid_size=cfg_bev.grid_size)

    polylines = []
    # 3D points of crosswalk areas
    valid_indexes = semantics[:,0] == crosswalk_class_idx
    if valid_indexes.sum() <= 20: # <= 20 crosswalk pointcloud
        return [] # empty: no crosswalk area

    # Extract points of semantic class
    points_crosswalk = points[valid_indexes]
    # Ignore height dimension
    points_crosswalk = points_crosswalk[:,[0,2]]
    # Point sparsification at dense areas for speed up
    points_crosswalk = sparsify_points(points_crosswalk, grid_size=cfg_bev.grid_size)

    # Produce convex hull
    clustering = DBSCAN(
        eps=cfg_bev.eps_cluster, min_samples=cfg_bev.min_points_crosswalk, n_jobs=-1
    ).fit(points_crosswalk)
    labels = clustering.labels_
    # Compute convex hull for each cluster (ignoring noise label -1)
    for label in set(labels):
        if label == -1: continue
        cluster_points = points_crosswalk[labels == label]
        if len(cluster_points) >= 3:
            hull = ConvexHull(cluster_points)
            # Extract the key points (vertices)
            polylines.append(cluster_points[hull.vertices])
    
    return polylines

# def vectorize_drivable_boundaries(points, semantics):
#     """ Vectorize drivable area boundaries
#         DBSCAN + Alpha Concave Hull
#         Args:
#             points: 3D point cloud
#             semantics: corresponding semantics
#         Outputs:
#             polylines: polyline of drivable areas
#     """
#     polylines = []
#     # 3D points of drivable areas (road + lane marking + crosswalk)
#     valid_indexes = np.isin(
#         semantics[:,0], 
#         [road_class_idx, lane_div_class_idx, lane_marking_class_idx, crosswalk_class_idx]
#     )
#     if valid_indexes.sum() <= 50: # <= 100 pointcloud
#         return [] # empty: no drivable area

#     # Extract points of semantic class
#     points_drivable = points[valid_indexes]
#     # Project to a single height plane (Y-axis)
#     points_drivable[:,1] = 0. # any constant => only care about X and Z
#     points_drivable = points_drivable[:,[0,2]]

#     # Cluster points in case drivable lanes are segregated
#     clustering = DBSCAN(
#         eps=cfg_bev.eps_cluster, min_samples=cfg_bev.min_points
#     ).fit(points_drivable)
#     labels = clustering.labels_
    
#     # Compute concave hull for each cluster (ignoring noise label -1)
#     for label in set(labels):
#         if label == -1: continue
#         cluster_points = points_drivable[labels == label]
#         if len(cluster_points) >= 3:
#             concave_hull = alphashape.alphashape(cluster_points, cfg_bev.alpha)
#             if concave_hull.geom_type == 'Polygon':
#                 # .coords returns (N, 2) including the closing point
#                 points = np.array(concave_hull.exterior.coords)
#                 polylines.append(points)
#             elif concave_hull.geom_type == 'MultiPolygon':
#                 # Handle MultiPolygons (if alpha is too high, the cluster might split)
#                 for poly in concave_hull.geoms:
#                     points = np.array(poly.exterior.coords)
#                     polylines.append(points)

#     return polylines


def vectorize_drivable_boundaries(points, semantics):
    """ Vectorize drivable area boundaries
        kNN (boundary points) + DBSCAN (group boundary points)
        Args:
            points: 3D point cloud
            semantics: corresponding semantics
                (expected: 2 semantic classes: drivable and non-drivable)
        Outputs:
            polylines: polyline of drivable boundaries
    """
    polylines = []
    # ===============================================
    # (0) Point densification at sparse areas
    #   proper nearest-neighbor-based boundary matching
    # ===============================================
    # points, semantics = transform_point_density(points, semantics, grid_size=cfg_bev.grid_size)
    # points, semantics = densify_points(points, semantics, grid_size=cfg_bev.max_dist_nn_phase1)

    # ===========================
    # 3D points of drivable areas (road + lane marking + crosswalk)
    indexes_drivable = semantics[:,0] == road_class_idx
    indexes_nondrivable = ~indexes_drivable
    if indexes_drivable.sum() <= 50: # <= 50 pointcloud
        return [] # empty: no drivable area


    # ===============================================
    # (1) Extract drivable and nondrivable points
    # ===============================================
    # Project to a single height plane (Y-axis)
    points[:,1] = 0. # any constant => only care about X and Z
    pts_drivable, pts_nondrivable = points[indexes_drivable], points[indexes_nondrivable]
    pts_drivable = pts_drivable[:,[0,2]] # X-axis and Z-axis
    pts_nondrivable = pts_nondrivable[:,[0,2]] # X-axis and Z-axis
    # ===============================================
    # (2) Point Sparsification at dense areas for speed up
    # ===============================================
    pts_drivable = sparsify_points(pts_drivable, grid_size=cfg_bev.grid_size)
    pts_nondrivable = sparsify_points(pts_nondrivable, grid_size=cfg_bev.grid_size)

    # ===============================================
    # (3) Search nondrivable points boundary to drivable points
    # ===============================================
    # nn = NearestNeighbors(n_neighbors=1).fit(pts_drivable)
    # dist, _ = nn.kneighbors(pts_nondrivable)
    # pts_boundary = pts_nondrivable[dist.flatten() <= cfg_bev.max_dist_nn_phase1]
    
    pts_boundary = []
    # Convert (M,2) -> (M,3) once: [X,Z] to [X,Y,Z]
    pts_drivable_3d = np.insert(pts_drivable, 1, 0.0, axis=1)
    pts_nondrivable_3d = np.insert(pts_nondrivable, 1, 0.0, axis=1)
    # Construct KDTree on drivable points
    tree = cKDTree(pts_drivable_3d)
    # Search boundary points based on k-nearest drivable neighbor
    dists, neighbors = tree.query(pts_nondrivable_3d, k=cfg_bev.k)
    # print(dists.shape) # (N_nondrivable, k)
    # Constrain max nearest neighbor distance
    mask_boundary = np.any(dists <= cfg_bev.max_dist_nn_phase1, axis=1)
    pts_boundary = pts_nondrivable[mask_boundary]
    
    # polylines.append(pts_boundary)
    
    # ===============================================
    # (4) Cluster points boundary points into polylines
    # ===============================================
    clustering = DBSCAN(
        eps=cfg_bev.eps_cluster, min_samples=cfg_bev.min_points_poly_boundary
    ).fit(pts_boundary)
    poly_groups = clustering.labels_
    print('Total Polylines:', len(set(poly_groups)))
    # Group boundary points into set of polylines (ignoring noise label -1)
    for label in set(poly_groups):
        if label == -1: continue
        cluster_points = pts_boundary[poly_groups == label]
        polylines.append(cluster_points)

        # # ===============================================
        # # (5) Filter noisy/false positive boundary points
        # #   => area of non-drivable lane >= min_area
        # # ===============================================
        # try: # valid convex points
        #     convex_hull = ConvexHull(cluster_points)
        #     convex_area = convex_hull.volume
        #     if convex_area < cfg_bev.min_area_boundary:
        #         continue
        #     polylines.append(cluster_points)
        # except: # coplanar points => invalid set, ignore
        #     continue

    return polylines




def visualize_map(polylines_dict, output_path):
    """ Visualize the vectorized polylines for each semantic type """
    plt.figure(figsize=(10, 8))
    
    # Define a color map for different types
    colors = {
        'boundary': 'green',
        'crosswalk': 'blue',
    }

    for label, polylines in polylines_dict.items():
        if polylines is None or len(polylines) == 0:
            continue
            
        color = colors.get(label, 'black')
        
        for i, poly in enumerate(polylines):
            # Plot polylines
            if label == 'crosswalk':
                # Plot crosswalk as closed polyline
                # Ensure the polyline is closed for visualization (last point -> first point)
                closed_poly = np.vstack([poly, poly[0]])
                plt.plot(closed_poly[:, 0], closed_poly[:, 1], color=color, 
                    linewidth=2, label=label if i == 0 else "")
                plt.scatter(closed_poly[:, 0], closed_poly[:, 1], color=color, s=10)
                plt.fill(closed_poly[:, 0], closed_poly[:, 1], color=color, alpha=0.3)
            else:
                # # plot other points as sequence of points
                # plt.plot(poly[:, 0], poly[:, 1], color=color, 
                #     linewidth=2, label=label if i == 0 else "")
                plt.scatter(poly[:, 0], poly[:, 1], color=color, s=10)

    plt.axis('equal') # Maintain real-world proportions
    plt.xlim(cfg_bev.xlim) # lateral
    plt.ylim(cfg_bev.zlim) # forward 
    plt.xlabel('X [m]', fontsize=12)
    plt.ylabel('Z [m]', fontsize=12)
    plt.gca().invert_xaxis() # flip x axis: 3D points x-direction points to the left
    # plt.title('Vectorized BEV Map', fontsize=16)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    # plt.show()
    plt.savefig(os.path.join(output_path, 'map_polylines.png'), bbox_inches='tight')