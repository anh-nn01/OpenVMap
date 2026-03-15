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
from scipy.spatial import ConvexHull, Delaunay, cKDTree
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
    polylines = []
    # 3D points of crosswalk areas
    valid_indexes = semantics[:,0] == crosswalk_class_idx
    if valid_indexes.sum() <= 20: # <= 20 crosswalk pointcloud
        return [] # empty: no crosswalk area

    # Extract points of semantic class
    points_crosswalk = points[valid_indexes]
    # Ignore height dimension
    points_crosswalk = points_crosswalk[:,[0,2]]

    # Produce convex hull
    clustering = DBSCAN(
        eps=cfg_bev.eps_cluster, min_samples=cfg_bev.min_points_crosswalk
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
    # 3D points of drivable areas (road + lane marking + crosswalk)
    indexes_drivable = semantics[:,0] == road_class_idx
    indexes_nondrivable = ~indexes_drivable
    if indexes_drivable.sum() <= 50: # <= 50 pointcloud
        return [] # empty: no drivable area

    # Project to a single height plane (Y-axis)
    points[:,1] = 0. # any constant => only care about X and Z
    # Extract drivable and nondrivable points
    pts_drivable, pts_nondrivable = points[indexes_drivable], points[indexes_nondrivable]
    pts_drivable = pts_drivable[:,[0,2]] # X-axis and Z-axis
    pts_nondrivable = pts_nondrivable[:,[0,2]] # X-axis and Z-axis

    # Search nondrivable points boundary to drivable points
    nn = NearestNeighbors(n_neighbors=1).fit(pts_drivable)
    dist, _ = nn.kneighbors(pts_nondrivable)
    pts_boundary = pts_nondrivable[dist.flatten() <= cfg_bev.max_dist_nn]
    
    # Cluster points boundary points to set of polylines
    clustering = DBSCAN(
        eps=cfg_bev.eps_cluster, min_samples=cfg_bev.min_points_poly_boundary
    ).fit(pts_boundary)
    labels = clustering.labels_
    # Group boundary points into set of polylines (ignoring noise label -1)
    for label in set(labels):
        if label == -1: continue
        cluster_points = pts_boundary[labels == label]
        polylines.append(pts_boundary)
    

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
            # Ensure the polyline is closed for visualization (last point -> first point)
            closed_poly = np.vstack([poly, poly[0]])
            # Plot the outline
            if label != 'boundary': # temporary debug
                plt.plot(closed_poly[:, 0], closed_poly[:, 1], color=color, 
                        linewidth=2, label=label if i == 0 else "")
            plt.scatter(closed_poly[:, 0], closed_poly[:, 1], color=color, s=10)
            # Fill crosswalks to make them look like real road markings
            if label == 'crosswalk':
                plt.fill(closed_poly[:, 0], closed_poly[:, 1], color=color, alpha=0.3)

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
    plt.savefig(os.path.join(output_path, 'map_polylines.png'))