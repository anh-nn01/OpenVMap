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
from scipy.spatial import ConvexHull, Delaunay, KDTree, cKDTree, Voronoi
from scipy.optimize import linear_sum_assignment
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
    


# def voronoi_based_road_boundary(
#         pts_driv_candidates, 
#         pts_nodriv_candidates,
#     ):
#     """ 
#         Extract true boundary points between drivable and nondrivable points
#         using Voronoi Diagram. One-to-one matching only.

#         Args:
#             pts_driv_candidates: candidate boundary drivable points
#             pts_nodriv_candidates: candidate boundary non-drivable points
#         Outputs:
#             pts_boundary: true non-drivable boundary points
#     """
#     offset = len(pts_driv_candidates)
#     all_pts = np.vstack([pts_driv_candidates, pts_nodriv_candidates])
#     voronoi_diagram = Voronoi(all_pts)  
    
#     # 1. Store distances for all valid Voronoi ridges (driv <-> nodriv)
#     # Use a dictionary to keep the cost matrix sparse
#     ridge_costs = {} 
#     for p1, p2 in voronoi_diagram.ridge_points:
#         is_p1_driv, is_p2_driv = p1 < offset, p2 < offset
#         if is_p1_driv != is_p2_driv:
#             d_idx = p1 if is_p1_driv else p2
#             n_idx = (p2 if is_p1_driv else p1) - offset
#             ridge_costs[(d_idx, n_idx)] = np.linalg.norm(all_pts[p1] - all_pts[p2])

#     if not ridge_costs: return np.array([])

#     # 2. Build cost matrix for the assignment problem
#     d_indices, n_indices = zip(*ridge_costs.keys())
#     d_unique, n_unique = np.unique(d_indices), np.unique(n_indices)
#     # Map original indices to matrix row/col indices
#     d_map = {idx: i for i, idx in enumerate(d_unique)}
#     n_map = {idx: i for i, idx in enumerate(n_unique)}
#     cost_matrix = np.full((len(d_unique), len(n_unique)), np.inf)
#     for (d, n), dist in ridge_costs.items():
#         cost_matrix[d_map[d], n_map[n]] = dist
        
#     # # 3. Solve for one-to-one matching
#     # row_ind, col_ind = linear_sum_assignment(cost_matrix)
#     # # 4. Filter out non-existent Voronoi edges (inf) and return matches
#     # valid = cost_matrix[row_ind, col_ind] != np.inf
#     # true_nodriv_indices = n_unique[col_ind[valid]]

#     # Voronoi extraction
#     true_nodriv_indices = np.unique([n for (_, n) in ridge_costs.keys()])
#     pts_boundary_candidates = pts_nodriv_candidates[true_nodriv_indices]
    
#     return pts_nodriv_candidates



def adaptive_LGAF(pts, k_neighbors=15, min_ratio=0.15, max_dist_mult=2.0):
    """
    Adaptive Local Geometric Anisotropy Filtering; 
    filter elongated/thin point areas with density awareness for sparse distant points
    
    Args:
        pts: (N, 2) array of [X, Z] points.
        k_neighbors: Number of neighbors to check (higher = more stable).
        min_ratio: How 'thick' a shape must be.
        max_dist_mult: Safety check. If neighbors are too far apart for the 
                       current range, it's likely noise.
    """
    tree = cKDTree(pts)
    
    # 1. Query the K nearest neighbors for every point
    # distances: distance to each neighbor, indices: indices of neighbors
    distances, indices = tree.query(pts, k=k_neighbors)
    mask = np.zeros(len(pts), dtype=bool)
    
    # Calculate distance from sensor (origin) for each point
    # This helps us understand the expected sparsity
    ranges = np.linalg.norm(pts, axis=1)

    for i in range(len(pts)):
        # 2. Sparsity Check
        # At further ranges, we expect points to be further apart.
        # We calculate a 'typical' spacing for this distance.
        avg_neighbor_dist = np.mean(distances[i, 1:])
        expected_spacing = (ranges[i] * 0.05) # Assume 5% angular spread as a rule of thumb
        
        # If the neighbors are ridiculously far away even for this range, skip it
        if avg_neighbor_dist > expected_spacing * max_dist_mult:
            continue

        # 3. Geometric Shape Analysis (PCA)
        neighbor_pts = pts[indices[i]]
        cov = np.cov(neighbor_pts.T)
        evals = np.linalg.eigvalsh(cov)
        
        # thickness_ratio = width / length
        ratio = np.sqrt(evals[0] / (evals[1] + 1e-6))
        
        # 4. Adaptive Threshold
        # We can be slightly less strict at a distance 
        # because sampling patterns can make thick things look thin.
        adaptive_tau = min_ratio * (1.0 - (ranges[i] / 100.0)) 
        adaptive_tau = max(adaptive_tau, 0.05) # min threshold: 0.05
        if ratio > adaptive_tau:
            mask[i] = True
            
    return pts[mask]



def voronoi_based_road_boundary(
    pts_driv_candidates,
    pts_nodriv_candidates,
    k=2,
    max_dist=None,
):
    """
    Extract non-drivable boundary candidates using Voronoi adjacency.
    For each drivable point, keep k closest Voronoi-adjacent non-drivable points.

    Args:
        pts_driv_candidates:   (N_d, 2)
        pts_nodriv_candidates: (N_n, 2)
        k: number of closest neighbors per drivable point
        max_dist: optional distance threshold (reject far pairs)
    Outputs:
        pts_boundary_candidates: (N_b, 2)
    """
    pts_driv_candidates = np.asarray(pts_driv_candidates, dtype=float)
    pts_nodriv_candidates = np.asarray(pts_nodriv_candidates, dtype=float)

    if len(pts_driv_candidates) == 0 or len(pts_nodriv_candidates) == 0:
        return np.empty((0, 2), dtype=float)

    offset = len(pts_driv_candidates)
    all_pts = np.vstack([pts_driv_candidates, pts_nodriv_candidates])

    vor = Voronoi(all_pts)

    # collect candidates per drivable point
    neigh = {}  # d_idx -> list of (dist, n_idx)
    for p1, p2 in vor.ridge_points:
        is_p1_driv = p1 < offset
        is_p2_driv = p2 < offset

        if is_p1_driv == is_p2_driv:
            continue

        d_idx = p1 if is_p1_driv else p2
        n_idx = (p2 if is_p1_driv else p1) - offset
        dist = np.linalg.norm(all_pts[p1] - all_pts[p2])

        if max_dist is not None and dist > max_dist:
            continue
        if d_idx not in neigh:
            neigh[d_idx] = []
        neigh[d_idx].append((dist, n_idx))

    if not neigh:
        return np.empty((0, 2), dtype=float)

    selected_nodriv = set()
    for d_idx, lst in neigh.items():
        lst_sorted = sorted(lst, key=lambda x: x[0])
        for dist, n_idx in lst_sorted[:k]:
            selected_nodriv.add(n_idx)

    selected_nodriv = np.array(sorted(selected_nodriv), dtype=int)
    pts_boundary_candidates = pts_nodriv_candidates[selected_nodriv]

    return pts_boundary_candidates



def order_points(points):
    """
        Order set pts for polyline visualization
    """
    points = np.array(points)
    n = len(points)
    if n <= 1:
        return points

    # 1. Find an "extreme" point to start (approximate diameter)
    # Start at index 0, find the point furthest from it
    dist_from_start = np.sum((points - points[0])**2, axis=1)
    start_idx = np.argmax(dist_from_start)
    
    # Initialize the ordered list with the first point
    ordered_indices = [start_idx]
    # Use a boolean mask to keep track of unvisited points
    mask = np.ones(n, dtype=bool)
    mask[0] = False
    for _ in range(n - 1):
        last_pt = points[ordered_indices[-1]]
        # Calculate squared Euclidean distance to all unvisited points at once
        # (Using squared distance is faster as it avoids the square root)
        unvisited_points = points[mask]
        unvisited_indices = np.where(mask)[0]
        distances = np.sum((unvisited_points - last_pt)**2, axis=1)
        # Find the index of the closest point and update the order/mask
        next_local_idx = np.argmin(distances)
        next_global_idx = unvisited_indices[next_local_idx]
        ordered_indices.append(next_global_idx)
        mask[next_global_idx] = False
        
    return points[ordered_indices]


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
    # points[:,1] = 0. # any constant => only care about X and Z
    pts_drivable, pts_nondrivable = points[indexes_drivable], points[indexes_nondrivable]
    pts_drivable = pts_drivable[:,[0,2]] # X-axis and Z-axis
    pts_nondrivable = pts_nondrivable[:,[0,2]] # X-axis and Z-axis
    # ===============================================
    # (2) Point Sparsification at dense areas for speed up
    # ===============================================
    pts_drivable = sparsify_points(pts_drivable, grid_size=cfg_bev.grid_size)
    pts_nondrivable = sparsify_points(pts_nondrivable, grid_size=cfg_bev.grid_size)

    # ===============================================
    # (3) Search candidate boundary points
    #   =>  all non-drivable pts and 
    #       associated k-nearest-drivable pts
    # ===============================================
    nn1 = NearestNeighbors(n_neighbors=1).fit(pts_drivable) # fit on drivable points
    # nn2 = NearestNeighbors(n_neighbors=1).fit(pts_nondrivable) # fit on nondrivable points
    dists_1, neighbors_1 = nn1.kneighbors(pts_nondrivable) 
    # dists_2, neighbors_2 = nn2.kneighbors(pts_drivable) 

    # Constrain max nearest neighbor distance
    mask_nondriv_boundary = np.any(dists_1 <= cfg_bev.max_dist_boundary, axis=1)
    # mask_driv_boundary = np.any(dists_2 <= cfg_bev.max_dist_boundary, axis=1)

    # candidate boundary (nondrivable pts)
    pts_nodriv_boundary = pts_nondrivable[mask_nondriv_boundary] 
    # candidate boundary (drivable pts)
    pts_driv_boundary = pts_drivable # pts_drivable[mask_driv_boundary]
    # print(pts_nodriv_boundary.shape, pts_driv_boundary.shape)

    # ===============================================
    # (4) Adaptive Local Geometric Anisotropy Filtering: 
    #     Filter "thin" regions caused by semantic inaccuracies
    # ===============================================
    pts_nodriv_boundary = adaptive_LGAF(
        pts_nodriv_boundary, 
        k_neighbors=cfg_bev.lgaf_neigbors,
        min_ratio=cfg_bev.lgaf_min_ratio, 
        max_dist_mult=cfg_bev.lgaf_max_radius,
    )


    # ===============================================
    # (5) Voronoi-based true boundary filtering
    # ===============================================
    # pts_boundary = pts_nodriv_boundary
    # pts_boundary = pts_driv_boundary
    pts_boundary = voronoi_based_road_boundary(
        pts_driv_candidates=pts_driv_boundary,
        pts_nodriv_candidates=pts_nodriv_boundary,
    )
    
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
        cluster_points = order_points(cluster_points)
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

    plt.xlim(cfg_bev.xlim) # lateral
    plt.ylim(cfg_bev.zlim) # forward 
    plt.xlabel('X [m]', fontsize=12)
    plt.ylabel('Z [m]', fontsize=12)
    ax = plt.gca()
    ax.invert_xaxis() # flip x axis: 3D points x-direction points to the left
    ax.set_aspect('equal', adjustable='box') 
    # plt.title('Vectorized BEV Map', fontsize=16)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    # plt.show()
    plt.savefig(os.path.join(output_path, 'map_polylines.png'), bbox_inches='tight')