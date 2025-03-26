# gaussian_splats_streaming

## '/Viewer'
The code for 3DGS viewer and user viewport trace collection. We modified the code from [https://github.com/clarte53/GaussianSplattingVRViewerUnity].

## '/layer'
The code for layered 3DGS model training. We modified the code from [https://github.com/clarte53/GaussianSplattingVRViewerUnity](https://github.com/VITA-Group/LightGaussian).

## '/bw_traces'
We use the outdoor 5G users’ walking traces [24] to simulate variable 5G network bandwidth

## SplatSelector.py
The scheduler determines what splats to download and in what order. It inputs the predicted viewport, bandwidth, and utility and outputs the object and layer ID.

## Utility.py
The code measures the utility of each splat. It takes the viewport and 3DGS scene as input and outputs the utility value.

