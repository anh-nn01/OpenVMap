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
from scipy.spatial import ConvexHull

# configs
sys.path.append(f"{pwd}/../configs")
import cfg_bev

# valid class index +=1 (0 = unknown class)
road_class_idx = cfg_bev.SEMANTIC_CLASSES.index('road')+1 # 'drivable road area'
lane_marking_class_idx = cfg_bev.SEMANTIC_CLASSES.index('lane marking')+1
crosswalk_class_idx = cfg_bev.SEMANTIC_CLASSES.index('crosswalk area')+1



def vectorize_map(points, semantics):
    """ Vectorize BEV maps given semantic 3D point cloud
        Args:
            points: 3D point cloud
            semantics: corresponding semantics
        Outputs:
            polyline_dict: vectorized polyline for each type
    """
    polylines_dict = {
        'boundary': None,
        'crosswalk': None,
    }
    

    #############################
    # Vectorize crosswalk area  #
    #############################
    polylines = vectorize_convex_crosswalk(points, semantics)
    polylines_dict['crosswalk'] = polylines


    return polylines_dict


def vectorize_convex_crosswalk(points, semantics):
    """ Vectorize crosswalk areas from the point cloud
        Geometry: DBSCAN + Chan's Convex Hull Algorithm

        TODO: minimize number of points
        TODO: should inherently cover non-convexity due to occlusions?
        
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
        eps=cfg_bev.eps_crosswalk, min_samples=cfg_bev.min_points_crosswalk
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




def visualize_map(polylines_dict, output_path):
    """ Visualize the vectorized polylines for each semantic type """
    plt.figure(figsize=(12, 10))
    
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
            plt.plot(closed_poly[:, 0], closed_poly[:, 1], '-o', color=color, 
                     linewidth=2, label=label if i == 0 else "")
            
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