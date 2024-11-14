# GPU_Image_Filtering

Title: graphicalRenderSim

With our current understanding of CUDA, we aim to compare multiple simulations of photo rendering. This will allow us to gain a better understanding of the communication between CPU and GPU as the program processes photo data line by line to eventually come out with a fully rendered image. We will experiment with file types (for example, jpg vs png) and image dimensions and see how the program’s efficiency changes and where CUDA’s strengths in image processing may lie.

As we develop our program, part of our exploration will be determining if CUDA can be used for video processing in addition to photo processing. We’re currently unsure as to whether this would work, as we’re still developing an understanding of the limitations of CUDA in relation to how it uses its GPU and how it generally handles graphics. If we discover that video processing is not supported in CUDA, we will document this, as well as how we arrived at that conclusion in our work.

Further References: 
- [NVIDIA CUDA documentation](https://docs.nvidia.com/cuda/)