# SPDX-FileCopyrightText: Copyright (c) 2026 torch-cuda sandbox contributors
# SPDX-License-Identifier: Apache-2.0
"""Build-time packaging shim. pyproject.toml remains the metadata source of truth.

The wheel contains a nanobind extension under ``torch_cuda.libs`` with embedded,
architecture-specific CUBINs. ``build_ext`` delegates that extension to CMake.
The optional extension is skipped unless ``TORCH_CUDA_BUILD_KERNELS=1``, so a
plain ``pip install -e .`` works on a machine without a CUDA toolkit — which is
what CI on the public side has.
"""

import os
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

ROOT_DIR = Path(__file__).parent.resolve()
KERNEL_LIBRARY_STEM = "_kernels"
BUILD_KERNELS_ENV = "TORCH_CUDA_BUILD_KERNELS"
TRUE_ENV_VALUES = frozenset({"1", "true", "yes", "on"})

VERSION = "0.1.0"


def build_kernels_requested() -> bool:
    return os.environ.get(BUILD_KERNELS_ENV, "").strip().lower() in TRUE_ENV_VALUES


def local_version() -> str:
    """Record the CUDA toolkit the wheel was built against, e.g. ``0.1.0+cu131``.

    Absent nvcc the wheel is pure Python, so it carries no local segment.
    """
    nvcc = os.environ.get("CUDACXX", "nvcc")
    try:
        out = subprocess.check_output([nvcc, "--version"], text=True)
    except (OSError, subprocess.CalledProcessError):
        return VERSION
    for token in out.split():
        if token.startswith("V") and token[1:2].isdigit():
            major, _, rest = token[1:].partition(".")
            minor = rest.partition(".")[0]
            return f"{VERSION}+cu{major}{minor}"
    return VERSION


class CMakeBuild(build_ext):
    """Delegate the kernel extension to CMake; leave pure-Python builds alone."""

    def build_extension(self, ext: Extension) -> None:
        if not ext.name.endswith(KERNEL_LIBRARY_STEM):
            super().build_extension(ext)
            return

        build_dir = ROOT_DIR / "cpp" / "build"
        out_dir = Path(self.get_ext_fullpath(ext.name)).parent.resolve()
        subprocess.check_call(
            [
                "cmake",
                "-S",
                str(ROOT_DIR / "cpp"),
                "-B",
                str(build_dir),
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={out_dir}",
                f"-DPython_EXECUTABLE={sys.executable}",
            ]
        )
        subprocess.check_call(["cmake", "--build", str(build_dir), "--parallel"])


setup(
    version=local_version(),
    ext_modules=[Extension(f"torch_cuda.libs.{KERNEL_LIBRARY_STEM}", sources=[])] if build_kernels_requested() else [],
    cmdclass={"build_ext": CMakeBuild} if build_kernels_requested() else {},
)
