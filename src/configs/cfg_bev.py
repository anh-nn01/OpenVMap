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
# map BEV voxels
###################################
BEV_HEIGHT = 2 # height of interest above the ground level, in meters
voxel_size = 0.3 # voxel size, in meter
xlim = (-15,15) # lateral
ylim = None # dynamically adjusted based on avg ground level / cam height
zlim = (0,30)  # depth

# for vectorization
eps_cluster = 2.5 # [meters] maximum distance to grouped in the same set of polyline
min_points = 50 # total points to be considered a valid set of polyline
alpha = 1.0 # alpha concave hull

