"""Build the optional SAMVG CUDA extension for release wheels.

Normal source installs intentionally remain pure Python.  Release builders
set ``VECTRIFY_BUILD_SAMVG_CUDA=1`` after installing the matching CUDA Torch
wheel; the resulting wheel bundles ``vectrify._samvg_cuda``.
"""

from __future__ import annotations

import os

from setuptools import setup


def cuda_extension():
    if os.environ.get("VECTRIFY_BUILD_SAMVG_CUDA") != "1":
        return [], {}
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    extension = CUDAExtension(
        "vectrify._samvg_cuda",
        ["src/vectrify/refine/_samvg_cuda.cu"],
        extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
    )
    return [extension], {"build_ext": BuildExtension}


ext_modules, cmdclass = cuda_extension()
setup(ext_modules=ext_modules, cmdclass=cmdclass)
