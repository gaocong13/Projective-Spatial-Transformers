from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='ProSTGrid_cpu',
    ext_modules=[
        CppExtension('ProSTGrid_cpu', [
            'ProSTGrid_cpu.cpp',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
