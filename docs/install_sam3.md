## Install Segment Anything 3
```sh
export HOME=/fs/nexus-projects/open_vectormap/ # replace your project directory here
export TMPDIR=$HOME/tmp
mkdir -p ~/tmp

source $HOME/miniconda3/bin/activate
conda create --prefix $HOME/miniconda3/envs/sam3 python=3.12 -y
conda activate sam3

pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
pip install -e ".[notebooks]"
pip install -e ".[train,dev]"
```

Then, request access to SAM3 on HuggingFace [here](https://github.com/facebookresearch/sam3#:~:text=access%20to%20the%20checkpoints%20on%20the%20SAM%203%20Hugging%20Face%20repo.).

## Example Usage
#### Env Activation
```sh
export HOME=/fs/nexus-projects/open_vectormap/ # replace your project directory here
source $HOME/miniconda3/bin/activate
conda activate sam3
cd ~/sam3
```

#### Simple Usage
```python
import torch
#################################### For Image ####################################
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
# Load the model
model = build_sam3_image_model()
processor = Sam3Processor(model)
# Load an image
#   Sample: "/fs/gamma-datasets/nuscenes/samples/CAM_FRONT/n015-2018-11-21-19-38-26+0800__CAM_FRONT__1542800955912460.jpg"
image = Image.open("<YOUR_IMAGE_PATH.jpg>")
inference_state = processor.set_image(image)
# Prompt the model with text
output = processor.set_text_prompt(state=inference_state, prompt="<YOUR_TEXT_PROMPT>")

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
```