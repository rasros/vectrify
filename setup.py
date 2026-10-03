"""Build the optional CUDA renderer extension for release wheels.

Normal source installs intentionally remain pure Python.  Release builders
set ``VECTRIFY_BUILD_CUDA_RENDERER=1`` after installing the matching CUDA Torch
wheel; the resulting wheel bundles ``vectrify._cuda_renderer``.
"""

from __future__ import annotations

import os

from setuptools import setup


def cuda_extension():
    if os.environ.get("VECTRIFY_BUILD_CUDA_RENDERER") != "1":
        return [], {}
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    extension = CUDAExtension(
        "vectrify._cuda_renderer",
        ["src/vectrify/refine/_cuda_renderer.cu"],
        extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
    )
    return [extension], {"build_ext": BuildExtension}


ext_modules, cmdclass = cuda_extension()
setup(ext_modules=ext_modules, cmdclass=cmdclass)
