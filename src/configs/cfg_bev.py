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
xlim = (-15,15) # lateral
ylim = None # dynamically adjusted based on avg ground level / cam height
zlim = (0, 30)  # depth

