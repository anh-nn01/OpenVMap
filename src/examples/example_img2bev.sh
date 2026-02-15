# !/bin/bash

##################################################################
# Author: anh-nn01; Feb 09, 2026
# Description: 
#   Generate semantic BEV voxels directly from monocular images
#   using a pre-trained monocular depth estimation model
#   1) DA3 : 3D point cloud generation from img
#   2) SAM3: 2D semantic segmentation from img
#   3) BEV generation: project 3D points to BEV grid and 
#                      assign semantic labels
##################################################################

export HOME=/fs/nexus-projects/open_vectormap/ # replace your project directory here
source $HOME/miniconda3/bin/activate
# export IMG_PATH="$HOME/src/examples/nusc/eg_2/n008-2018-05-21-11-06-59-0400__CAM_FRONT__1526915292912465.jpg"
# export IMG_PATH="$HOME/src/examples/nusc/eg_4/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151605512404.jpg"
export IMG_PATH="$HOME/src/examples/nusc/eg_5/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402942162460.jpg"

##################################################################
# Step 1: Generate 3D point cloud from monocular images using DA3
##################################################################
export GLB_PATH="$HOME/src/examples/nusc/eg_5/da3_output/"
conda activate da3
cd $HOME/src/processors/
python3 generate_3Dpc.py --img_path $IMG_PATH --export_dir $GLB_PATH

##################################################################
# Step 2: Generate 2D semantic segmentation from monocular images 
#         using SAM3
##################################################################
export SEM_PATH="$HOME/src/examples/nusc/eg_5/sam3_output/"
conda activate sam3
cd $HOME/src/processors/

##################################################################
# Step 3: Merge 3D point cloud and 2D semantic segmentation 
#         to generate BEV voxels
##################################################################
