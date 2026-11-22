#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

source_dir = os.path.dirname(os.path.abspath(__file__))
include_dir = os.path.abspath(os.path.join(source_dir, "../../cuda_rasterizer"))


setup(
    name="diff_gaussian_rasterization",
    packages=["diff_gaussian_rasterization"],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
                os.path.join(include_dir, "rasterizer_impl.cu"),
                os.path.join(include_dir, "forward.cu"),
                os.path.join(include_dir, "backward.cu"),
                os.path.join(include_dir, "rasterize_points.cu"),
                os.path.join(include_dir, "adam.cu"),
                os.path.join(include_dir, "conv.cu"),
                os.path.join(source_dir, "ext.cpp"),
            ],
            extra_compile_args=["-I" + include_dir],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
