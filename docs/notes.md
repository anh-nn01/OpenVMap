## TODO
### A. Data 
1. Generate nuscenes folder by scenes
    ```
    nusc/
        + scene_xxxxx/
            + frontview/
                00.jpg
                01.jpg
                ...
            + poses/
                00.txt
                01.txt
                ...
            + 2d_semantic/
                00.npy
                01.npy
                ...
            + depth/
                00.npy
                01.npy
                ...
            + intrinsics
                intrinsics.npy
            + 3D_semantic/
                00.glb
                01.glb
                ...
            + global_
    av2/
        ...
    ```

### B. Model 
1. Complete pipeline: 2D image to 3D semantic point cloud
2. Complete pipeline: 2D video to global 3D semantic point cloud
3. Modify vector map decoder

### Experiments
1. Compare GlobalMapNet:
    * front-view only
    * cite GAP and AP metrics
2. Demonstrate generalizability: train on `nusc` but generalize on any dataset
3. In-the-wild videos

## Some research notes on pipeline designs
### A. Local 3D PC and Voxelization
1. Define `ylim` (height range of 3D point cloud of interest) by `avg ground height` using drivable lane (semantic index = 1) <br>
    e.g. avg Y-coords of all points associated with 
    * drivable lane and 
    * in point cloud of interest <br>
    \>> more generalizable to different sensor setups (e.g. camera heights)
2. Filter out noisy 3D points using spherical filtering <br>
    \>> less noisy BEV voxels
3. `bev_voxel`: use max semantic index along height <br>
    * Compress along height dimension Y (pillar)
    * Semantic index hierarchy: designed so that class with higher index has higher priority for the same voxel
    * Nearest neighboring along BEV dimension (XZ)
4. `bev_voxel`: post-processing: use ZX kernel size = (5)


### B. Global merging
1. **Occlusion-Aware Global Point Cloud Fusion**
    * Unproject *3D point cloud* for each frame
    * Produce *2D semantic* for each frame
    * Produce *semantic 3D point cloud* for each frame
    * Occlusion-Aware Global Semantic PC Fusion:
        * Use GT pose $M_t$: provided in dataset
        * Semantic assignment using semantic hierarchy <br>
        `curb` > `lane marking` > `lane divider` > `drivable road area` > `unmatch` > `no point (occlusion)` <br> <br>
    \>> address occlusion caused by: <br> 
    (1) obstacles; <br>
    (2) limited FOV, and <br>
    (3) sparse distant point cloud

2. **Voxelization** <br>
Once Occlusion-Aware 3D PC is generated, the point cloud has (1) lower occlusion and (2) lower noises. <br>
Proceed to voxelize local map using **first-frame alignment**: 
    * At time `t=0`: initial ego pose 
    $p_{ego}(0) = 
    \begin{bmatrix} 
        1, 0, 0, 0 \\
        0, 1, 0, 0 \\
        0, 0, 1, 0 \\
        0, 0, 0, 1 \\
    \end{bmatrix}$
    * At time `t>0`: 
        * use GT pose $M_t$: 
        $$p_{ego}(t) = M_t \times p_{ego}(t-1)$$
        * Filter local point cloud set based on global point cloud $P_{global}$, local ego pose $p_{ego}(t)$, and local `xlim, ylim, zlim`:
        $$P_{local}(t) = Extract(P_{global}, p_{ego}(t-1), xlim, ylim, zlim)$$

3. **Vector map mapping**
    * Train map decoder using BEV voxel fusion
    


