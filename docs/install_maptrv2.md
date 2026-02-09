## Install MapTRv2
```sh
export HOME=/fs/nexus-projects/open_vectormap/ # replace your project directory here
export TMPDIR=$HOME/tmp
mkdir -p ~/tmp

source $HOME/miniconda3/bin/activate
conda create --prefix $HOME/miniconda3/envs/maptrv2 python=3.8 -y
conda activate maptrv2

module load cuda/11.1.1
module unload gcc/11.2.0
module load gcc/7.5.0 # verify with gcc --version
gcc --version

#####################################
#           Install torch           #
#####################################
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install timm

#####################################
# [MMCV suite] Build from source:   #
#               (more stable)       #
#####################################
# Build from source MMCV
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv; git checkout v1.4.0
MMCV_WITH_OPS=1 pip install -e .  # package mmcv-full will be installed after this step
python -c "import mmcv" # test
cd ..
# Build from source MMDET
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection; git checkout v2.14.0
pip install -r requirements/build.txt
pip install -v -e .
python -c "import mmdet" # test
cd ..
# # Build from source MMDET3D
# git clone https://github.com/open-mmlab/mmdetection3d.git
# cd mmdetection3d; git checkout v0.17.3
# pip install -r requirements/build.txt
# pip install -v -e .
# cd ..
# MMSEGMENTATION
pip install mmsegmentation==0.14.1

#####################################
#   Install MapTR dependencies      #
#####################################
git clone https://github.com/hustvl/MapTR.git
cd MapTR/
git checkout maptrv2
cd mmdetection3d
python setup.py develop
python -c "import mmdet3d" # test

cd ../projects/mmdet3d_plugin/maptr/modules/ops/geometric_kernel_attn
python setup.py build install

# additional dependencies
cd $HOME/MapTR
pip install -r requirement.txt
pip install lyft_dataset_sdk
pip install numba==0.48.0
# pip install numpy==1.19.5
pip install numpy==1.23.5
pip install scikit-image==0.19.3
pip install pandas==1.4.4
pip install nuscenes-devkit

mkdir -p ckpts
cd ckpts 
wget https://download.pytorch.org/models/resnet50-19c8e357.pth
wget https://download.pytorch.org/models/resnet18-f37072fd.pth
```

## Example Usage
#### Env Activation
```sh
export HOME=/fs/nexus-projects/open_vectormap/ # replace your project directory here
export TMPDIR=$HOME/tmp
source $HOME/miniconda3/bin/activate
conda activate maptrv2

module load cuda/11.1.1
module unload gcc/11.2.0
module load gcc/7.5.0 # verify with gcc --version
gcc --version
```

<hr style="border: 2px dashed gray;">

#### Dataset Preparation 
#### Dataset 1: NuScenes
If you have not install NuScenes, download it [here](https://www.nuscenes.org/nuscenes#download). <br>
Also, download the CAN bus expansion data [here](https://www.nuscenes.org/download).

```sh
# after env activation
cd $HOME/MapTR
export NUSC_PATH=/fs/gamma-datasets/nuscenes/ # replace path here
python tools/maptrv2/custom_nusc_map_converter.py \
    --root-path $NUSC_PATH \
    --out-dir ./data/nuscenes \
    --extra-tag nuscenes \
    --version v1.0 \
    --canbus $NUSC_PATH
```

<hr style="border: 2px dashed gray;">

#### Simple Usage
```python
# Quick test
PYTHONPATH=. \
python tools/train.py \
./projects/configs/maptrv2/maptrv2_nusc_r50_110ep_mono.py \
--deterministic
```