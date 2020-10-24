Dependencies
============
Dependencies assume compilation on Linux (Ubuntu). Tested on Ubuntu 20.04 and CUDA 11.0.

- CMake 3.16
- OpenGL
    > sudo apt-get install mesa-utils
- freeGLUT
    > sudo apt-get install freeglut3-dev
- CUDA

CLion
=====
CLion requires to set the enviroment variable for CUDA compiler:
> CUDACXX=/usr/local/cuda-11.0/bin/nvcc 
