## Install Depth Anything 3
```sh
export HOME=/fs/nexus-projects/open_vectormap/ # replace your project directory here
export TMPDIR=$HOME/tmp
mkdir -p ~/tmp

source $HOME/miniconda3/bin/activate
conda create --prefix $HOME/miniconda3/envs/da3 python=3.10 -y
conda activate da3

git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
cd Depth-Anything-3/
pip install xformers torch\>=2 torchvision
pip install -e . # Basic
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70 # for gaussian head
pip install -e ".[app]" # Gradio, python>=3.10
pip install -e ".[all]" # ALL

########################################################################
# NOTE: when executing SAM3 example, the model weight is downloaded at: 
# $HOME/.cache/huggingface/hub/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt
########################################################################
# update your SAM3 weight here
export SAM3_WEIGHT=$HOME/.cache/huggingface/hub/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt

# install utralytics for efficient SAM3 predictor
pip install -U ultralytics==8.3.237
```

Read more about the model [Here](https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE-1.1).

## Example Usage
#### Env Activation
```sh
export HOME=/fs/nexus-projects/open_vectormap/ # replace your project directory here
source $HOME/miniconda3/bin/activate
conda activate da3
cd ~/Depth-Anything-3

module load cuda/12.4.1
module load gcc/14.2.0
```
#### Simple usage
```python
import torch
from depth_anything_3.api import DepthAnything3

# Load model from Hugging Face Hub
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DepthAnything3.from_pretrained("depth-anything/da3nested-giant-large")
model = model.to(device=device)

# Run inference on images
# images = ["image1.jpg", "image2.jpg"]  # List of image paths, PIL Images, or numpy arrays
# path to your test image here
images = [
    "/fs/gamma-datasets/nuscenes/samples/CAM_FRONT/n015-2018-11-21-19-38-26+0800__CAM_FRONT__1542800955912460.jpg",
]
prediction = model.inference(
    images,
    export_dir="output_testDA3",
    export_format="glb"  # Options: glb, npz, ply, mini_npz, gs_ply, gs_video
)

# Access results
print(prediction.depth.shape)        # Depth maps: [N, H, W] float32
print(prediction.conf.shape)         # Confidence maps: [N, H, W] float32
print(prediction.extrinsics.shape)   # Camera poses (w2c): [N, 3, 4] float32
print(prediction.intrinsics.shape)   # Camera intrinsics: [N, 3, 3] float32
```