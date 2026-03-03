SEMANTIC_CLASSES = [
	# fundamental road elements
	"drivable road area", "lane divider", "lane marking", "crosswalk area",
	# union these elements to form a single class
	"curb", 
	# dynamic objects / obstacles
	"vehicle",
]


###################################
# map BEV voxels
###################################
BEV_HEIGHT = 1.5 # height of interest above the ground level, in meters
voxel_size = 0.5 # voxel size, in meter

