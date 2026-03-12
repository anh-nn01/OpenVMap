## Setup universal environment
```sh
export HOME=/fs/nexus-projects/open_vectormap/ # replace your project directory here
export TMPDIR=$HOME/tmp
mkdir -p ~/tmp

source $HOME/miniconda3/bin/activate
conda create --prefix $HOME/miniconda3/envs/universal python=3.12 -y
conda activate universal

##########################################
# Install Depth Anything 3 dependencies
##########################################
# git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
cd $HOME/Depth-Anything-3/
pip install xformers torch\>=2 torchvision
pip install -e . # Basic
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70 # for gaussian head
pip install -e ".[app]" # Gradio, python>=3.10
pip install -e ".[all]" # ALL
pip install nuscenes-devkit==1.2.0

##########################################
# Install SAM 3 dependencies
##########################################
# git clone https://github.com/facebookresearch/sam3.git
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
cd $HOME/sam3
pip install -e .
pip install -e ".[notebooks]"
pip install -e ".[train,dev]"

##########################################
# Install ClearObject dependencies
##########################################
# git clone https://github.com/zjx0101/ObjectClear.git
cd $HOME/ObjectClear/
pip install -r requirements.txt --ignore-installed
pip install -r hugging_face/requirements.txt --ignore-installed

cd $HOME
```