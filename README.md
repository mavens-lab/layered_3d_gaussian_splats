<div align="center">
    <h1> L3GS: Layered 3D Gaussian Splats for Efficient 3D Scene Delivery</h1>
</div>

## Abstract
Traditional 3D content representations include dense point clouds that consume large amounts of data and hence network bandwidth, while newer representations such as neural radiance fields suffer from poor frame rates due to their non-standard volumetric rendering pipeline. 3D Gaussian
splats (3DGS) can be seen as a generalization of point clouds that meet the best of both worlds, with high visual quality and efficient rendering for real-time frame rates. However, delivering 3DGS scenes from a hosting server to client devices is still challenging due to high network data consumption (e.g. 1.5 GB for a single scene). The goal of this work is to create an efficient 3D content delivery framework that allows users to view high quality 3D scenes with 3DGS as the underlying data representation. The main contributions of the paper are: (1) Creating new layered 3DGS scenes for efficient delivery, (2) Scheduling algorithms to choose what splats to download at what time, and (3) Trace-driven experiments from users wearing virtual reality headsets to evaluate the visual quality and latency. Our system for Layered 3D Gaussian Splats delivery (L3GS) demonstrates high visual quality, achieving 16.9% higher average SSIM compared to baselines, and also works with other compressed 3DGS representations. 

## Overview

L3GS is an efficient 3D content delivery framework that allows users to view high quality 3D scenes with 3DGS as the underlying data representation. 

The key algorithms and models in L3GS that proposed in the paper are summarized as follows: (1) machine learning algorithms to train layered 3DGS scenes for efficient delivery, (2) efficient scheduling algorithms to determine the splats delivery sequence, and (3) a viewport/bandwidth predictor to estimate users' future viewport/bandwidth.

The following datasets are used for the experiments: 
(1) 3D scene training dataset: Mip-NeRF360, Tanks&Temples, and 3 segmented scenes from Gaussian Grouping. 
(2) bandwidth simulation dataset: sampled and scaled outdoor user walking traces
(3) user viewport trace dataset: collected viewport traces from 6 users across 8 scenes each.

## How to access

### Train 3DGS models with object ID 
Clone the gaussian grouping repository from GitHub: [https://github.com/lkeab/gaussian-grouping/blob/main/docs/dataset.md](https://github.com/lkeab/gaussian-grouping/blob/main/docs/dataset.md). Then, set up the environment following the README. Download the datasets and pretrained models. Run the training scripts following the README.

### Train layered 3DGS models, evaluate generated scene
Clone this repository from GitHub. Then, set up the environment following the README. Download the datasets and pretrained models. Run the training scripts following the README.

### 3DGS view and user viewport trace collection
Download and install Meta Quest Link. Set up the VR headset. Clone the repository from GitHub. Put the 3DGS scene in the '/Viewer' folder. (1) viewer: Launch the executable GaussianSplattingVRViewer.exe. (2) Replay: Launch the executable GaussianSplattingVRViewer.exe from the 'replay' folder. (3) Collect trace: Launch the executable GaussianSplattingVRViewer.exe from the 'collect' folder.

### Run simulation experiments
Clone the repository from GitHub. Each module of the experiments can be obtained in the repository.

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

