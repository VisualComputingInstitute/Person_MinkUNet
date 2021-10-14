from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# From https://github.com/sshaoshuai/PointRCNN/tree/master/lib/utils/iou3d

setup(
    name='iou3d',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension('iou3d_cuda', [
            'iou3d/src/iou3d.cpp',
            'iou3d/src/iou3d_kernel.cu',
        ],
        extra_compile_args={'cxx': ['-g'],
                            'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension})
