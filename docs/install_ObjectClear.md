## Install ObjectClear
Share the same conda env with `SAM3`. If you haven't installed SAM3, please follow [Segment Anything 3 installation guide](docs/install_sam3.md).

```sh
export HOME=/fs/nexus-projects/open_vectormap/ # replace your project directory here
export TMPDIR=$HOME/tmp
mkdir -p ~/tmp

source $HOME/miniconda3/bin/activate
# Share SAM3'venv
conda activate sam3

cd $HOME
git clone https://github.com/zjx0101/ObjectClear.git
cd ObjectClear/
pip install -r requirements.txt --ignore-installed
pip install -r hugging_face/requirements.txt --ignore-installed
```

## Example Usage
#### Env Activation
```sh
export HOME=/fs/nexus-projects/open_vectormap/ # replace your project directory here
source $HOME/miniconda3/bin/activate
conda activate sam3
```

#### Simple Usage
```sh
cd src/processors/
python generate_semantics.py \
--img_path=<PATH_TO_IMG> \
--export_path=<PATH_TO_CLEARED_IMG> \
```