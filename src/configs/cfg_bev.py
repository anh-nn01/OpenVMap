OCCLUSION_CLASSES = [
	"people", "vehicle", "pole", # "pedestrian"
]
SEMANTIC_CLASSES = [
	# fundamental road elements
	# "drivable road area", "lane divider", "lane marking", "crosswalk area",
	"road", "lane divider", "lane marking", "crosswalk area",
	# union these elements to form a single class
	"curb", 
	# dynamic objects / obstacles
	# "vehicle", "pedestrian", "pole",
]
SEG_THRESHOLD = 0.2 # Semantic threshold

# valid class index +=1 (0 = unknown class)
road_class_idx = SEMANTIC_CLASSES.index('road')+1 # 'drivable road area'
lane_div_class_idx = SEMANTIC_CLASSES.index('lane divider')+1
lane_marking_class_idx = SEMANTIC_CLASSES.index('lane marking')+1
crosswalk_class_idx = SEMANTIC_CLASSES.index('crosswalk area')+1

###################################
# map BEV voxels [play with this]
###################################
BEV_HEIGHT = 1.5 # height of interest above the ground level, in meters
voxel_size = 0.3 # voxel size, in meter
xlim = (-15,15) # lateral
ylim = None # dynamically adjusted based on avg ground level / cam height
zlim = (0,30)  # depth

# vectorization
# [meters] 1 point per grid size: point sparsification for speed up
grid_size = 0.2
# [meters] maximum distance to grouped in the same polyline
eps_cluster = 2.5 
# total points to be considered a valid set of polyline
min_points_crosswalk = 50 

# vectorization: drivable
# alpha concave hull
alpha = 1.0 
# [meters] maximum distance to the nearest neighboring drivable point to be considered boundary
max_dist_nn_phase1 = 1.0 # phase 1: for efficient densification + boundary point search
max_dist_nn_phase2 = 0.2 # phase 2: for filtering out true points
# total points in valid boundary polylines
min_points_poly_boundary = 3 
k = 10 # k nearest neighbor (KDTree) for boundary identification
# [meter^2] minimum area of nondrivable boundary to consider a valid boundary => ignore false positive areas
min_area_boundary = 1

