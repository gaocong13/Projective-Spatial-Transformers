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
We implemented our ProST grid generator function using [PyTorch C++ and CUDA extension](https://pytorch.org/tutorials/advanced/cpp_extension.html). We take the camera intrinsic parameters (usually defined by the intrinsic matrix $K\in R^{3\times 3}$) as input,
