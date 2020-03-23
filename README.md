# Projective_Spatial_Transformers_and_2D3D_Registrations
[Project](webpagelink) | [Paper](arxivlink) | [Video](youtubelink)

Pytorch implementation of **Pro**jective **S**patial **T**ransformers (**ProST**) and training convex-shape image similarity metrics

![](imgs/mov.gif)

Generalizing Spatial Transformers to Projective  Geometry with Applications to 2D/3D Registration
[Cong Gao](http://www.cs.jhu.edu/~gaoc/), [Xingtong Liu](http://www.cs.jhu.edu/~xingtongl/), Wenhao Gu, [Mehran Armand](https://ep.jhu.edu/about-us/faculty-directory/861-mehran-armand), [Russell Taylor](https://www.cs.jhu.edu/~rht/) and [Mathias Unberath](https://mathiasunberath.github.io/)

We propose a novel Projective Spatial Transformer module that generalizes spatial transformers to projective geometry, thus enabling differentiable volume rendering. We demonstrate the usefulness of this architecture on the example of 2D/3D registration between radiographs and CT scans. Specifically, we show that our trans- former enables end-to-end learning of an image processing and projection model that approximates an image similarity function that is convex with respect to the pose parameters, and can thus be optimized effectively using conventional gradient descent.

<img src="imgs/Fig_ProST.png" width="900px"/>

## Setup

### Prerequisites
- Linux or OSX (OSX has CPU support only)
- NVIDIA GPU + CUDA

### Getting Started
- Install torch, torchvision from https://pytorch.org/. We recommend torch >= 1.3.0, torchvision >= 0.4.0.
- Check requirements.txt for dependencies. You can use pip install:
```bash
pip install requirements.txt
```
### Install ProST grid generator
We implemented our ProST grid generator function using [PyTorch C++ and CUDA extension](https://pytorch.org/tutorials/advanced/cpp_extension.html). The implementation is inspired by the [Spatial Transformer Network PyTorch C++ source code](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/AffineGridGenerator.cpp), but we created our own geometries and kernel functions, which is illustrated in Fig.1(a). We take the camera intrinsic parameters (usually defined by the intrinsic matrix ![K\in \mathbf{R}^{3\times 3}](https://render.githubusercontent.com/render/math?math=K%5Cin%20%5Cmathbf%7BR%7D%5E%7B3%5Ctimes%203%7D)) as input, and generate a grid variable with shape ![B\times (M\cdot N\cdot K)\times 4](https://render.githubusercontent.com/render/math?math=B%5Ctimes%20(M%5Ccdot%20N%5Ccdot%20K)%5Ctimes%204), where B is batch size. 

The input parameters include:
- *theta*: [torch tensor] pose parameter, which is used for cloning basic properties of the grid tensor
- *size*: [torch tensor size] size of the projection image. e.g.: proj_img.size()
- *dist_min*, *dist_max*: [float] the min/max distance from source to the 8 corner points of the transformed volume, which is used to define the inner and outer radius of the green fan in Fig.1(a). This defines the grid ROI that covers the volume.
- *src*, *det*: [float] normalized source and detector z coordinates in ![F^r](https://render.githubusercontent.com/render/math?math=F%5Er).
- *pix_spacing*, *step_size*: [float] normalized 2D pixel spacing and sampling step size.

*src*, *det* and *pix_spacing* can be decomposed from the intrinsic matrix ![K](https://render.githubusercontent.com/render/math?math=K). We provide function *** that takes K as input

ProST grid generator is installed using python setuptools. 
```bash
cd ./ProSTGrid
python setup.py install
```