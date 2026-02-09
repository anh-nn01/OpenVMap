## Todo list
* **TODO 1:** Write processor for DA3+SAM3 scene completion
* **TODO 2:** Understand train/val [split code](https://github.com/hustvl/MapTR/blob/maptrv2/tools/create_data.py) from MapTRv2
* **TODO 3:** Implement + run processor for procducing PC & semantics by splits 

## Getting started
1. [GPU interactive session](docs/interactive_bash.md) 
2. [Miniconda3 installation](docs/miniconda3.md) (install ONCE)
3. Dependencies installation
    * [Encoder] Install [Depth Anything 3](docs/install_da3.md)
    * [Encoder] Install [Segment Anything 3](docs/install_sam3.md)
    * [Decoder] Install [MapTRv2]()


## Project Structure
<pre>
project-root/
├── README.md
├── miniconda3/
├── Depth-Anything-3/
├── sam3/
├── data/
    ├── Argoverse_2/
    ├── KITTI/
    └── NuScenes/
        ├── 3D_GLB/
            └── {scene_token}_{timestamp}.glb
        ├── Semantics/
            └── {scene_token}_{timestamp}.glb
        └── BEV_semantic/
            └── {scene_token}_{timestamp}.glb
├── src
    ├── examples
        ├── nusc
        └── av2
    ├── processors
        ├── generate_3Dpc.py
        ├── generate_semantics.py
        └── generate_bev.py
    ├── models
        ├── map_decoder
        ├── alignment_modules
        └── ...
├── results
     ├── Argoverse_2
     ├── KITTI
     └── NuScenes
         └── ...
└── weights
    ├── <model1>/
        └── <model1>.pth
    └── ...
</pre>