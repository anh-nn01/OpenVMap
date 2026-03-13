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

###################################
# map BEV voxels
###################################
BEV_HEIGHT = 2 # height of interest above the ground level, in meters
voxel_size = 0.3 # voxel size, in meter
xlim = (-15,15) # lateral
ylim = None # dynamically adjusted based on avg ground level / cam height
zlim = (0,30)  # depth

# for vectorization
eps_crosswalk = 1.0 # [meters] maximum distance to grouped in the same cluster 
min_points_crosswalk = 50 # total points to be considered a valid crosswalk

