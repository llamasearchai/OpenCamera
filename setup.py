#!/usr/bin/env python3
"""Root setup script to build the OpenCam C++ extension via pybind11.
This makes ``pip install -e .`` compile the high-performance bindings so that
Python tests use the real implementation instead of the stubs.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from setuptools import find_packages, setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _find_pkg_config(name: str) -> tuple[list[str], list[str], list[str]]:
    """Return ``(include_dirs, library_dirs, libs)`` for a *pkg-config* package.

    The helper extracts *all* relevant compiler/linker search paths from
    ``pkg-config`` so that downstream compilation succeeds even when the
    libraries live in non-standard locations (e.g. Homebrew on macOS).
    """
    try:
        cflags = subprocess.check_output(["pkg-config", "--cflags", name], text=True).split()
        ldflags = subprocess.check_output(["pkg-config", "--libs", name], text=True).split()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return [], [], []

    include_dirs: list[str] = [flag[2:] for flag in cflags if flag.startswith("-I")]
    library_dirs: list[str] = [flag[2:] for flag in ldflags if flag.startswith("-L")]
    libs: list[str] = [flag[2:] for flag in ldflags if flag.startswith("-l")]
    return include_dirs, library_dirs, libs


# Optional dependencies (do not abort if missing; users may build without them)
opencv_inc, opencv_ldirs, opencv_libs = _find_pkg_config("opencv4")
spdlog_inc, spdlog_ldirs, spdlog_libs = _find_pkg_config("spdlog")


# ---------------------------------------------------------------------------
# Extension module definition
# ---------------------------------------------------------------------------

VERSION = "0.5.0"

ext_modules = [
    Pybind11Extension(
        "opencam_auto_exposure",
        sources=[
            "python/bindings/pybind_auto_exposure.cpp",
            "algorithms/3a/auto_exposure.cpp",
            "core/src/camera.cpp",
            "algorithms/isp/debayer.cpp",
        ],
        include_dirs=[
            pybind11.get_include(),
            ".",
            "algorithms",
            "core/include",
            "algorithms/3a",
            "algorithms/isp",
        ]
        + opencv_inc
        + spdlog_inc,
        library_dirs=opencv_ldirs + spdlog_ldirs,
        libraries=opencv_libs + spdlog_libs + ["pthread"],
        cxx_std=17,
        define_macros=[("VERSION_INFO", f'"{VERSION}"')],
        extra_compile_args=["-O3", "-Wall", "-Wextra", "-fPIC"],
    )
]


# ---------------------------------------------------------------------------
# Call setup()
# ---------------------------------------------------------------------------

setup(
    name="opencam-auto-exposure",
    version=VERSION,
    author="Nik Jois",
    author_email="nikjois@llamasearch.ai",
    description="OpenCam Auto Exposure with C++ acceleration",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
) 