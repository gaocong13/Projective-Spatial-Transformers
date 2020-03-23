from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='ProSTGrid_cuda',
    ext_modules=[
        CUDAExtension('ProSTGrid_cuda', [
            'ProSTGrid_cuda.cpp',
            'ProSTGrid_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
