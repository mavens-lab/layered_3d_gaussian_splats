<div align="center">
    <h1> L3GS: Layered 3D Gaussian Splats for Efficient 3D Scene Delivery]{L3GS: Layered 3D Gaussian Splats for Efficient 3D Scene Delivery</h1>
</div>

## Abstract

L3GS is an efficient 3D content delivery framework that allows users to view high quality 3D scenes with 3DGS as the underlying data representation. 

The key algorithms and models in L3GS that proposed in the paper are summarized as follows: (1) machine learning algorithms to train layered 3DGS scenes for efficient delivery, (2) efficient scheduling algorithms to determine the splats delivery sequence, and (3) a viewport/bandwidth predictor to estimate users' future viewport/bandwidth.

The following datasets are used for the experiments: (1) 3D scene training dataset: Mip-NeRF360, Tanks\&Temples, and 3 segmented scenes from Gaussian Grouping. 
(2) bandwidth simulation dataset: sampled and scaled outdoor user walking traces
(3) user viewport trace dataset: collected viewport traces from 6 users across 8 scenes each.

## '/Viewer'
The code for 3DGS viewer and user viewport trace collection. We modified the code from [https://github.com/clarte53/GaussianSplattingVRViewerUnity](https://github.com/clarte53/GaussianSplattingVRViewerUnity).

## '/layer'
The code for layered 3DGS model training. We modified the code from [https://github.com/clarte53/GaussianSplattingVRViewerUnity](https://github.com/VITA-Group/LightGaussian).

## '/bw_traces'
We use the outdoor 5G users’ walking traces [24] to simulate variable 5G network bandwidth

## SplatSelector.py
The scheduler determines what splats to download and in what order. It inputs the predicted viewport, bandwidth, and utility and outputs the object and layer ID.

## Utility.py
The code measures the utility of each splat. It takes the viewport and 3DGS scene as input and outputs the utility value.

