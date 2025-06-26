#!/usr/bin/env python3
"""
Setup script for OpenCam Auto Exposure Python bindings.
Author: Nik Jois <nikjois@llamasearch.ai>
"""

import os
import sys
import subprocess
from pathlib import Path
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
from setuptools import setup, Extension, find_packages

# Read version from file
def get_version():
    version_file = Path(__file__).parent / "version.txt"
    if version_file.exists():
        return version_file.read_text().strip()
    return "1.0.0"

# Read README
def get_long_description():
    readme_file = Path(__file__).parent.parent.parent / "README.md"
    if readme_file.exists():
        return readme_file.read_text()
    return "OpenCam Auto Exposure Algorithm Python Bindings"

# Find OpenCV
def find_opencv():
    """Locate OpenCV include directories and libraries.

    The helper first tries *pkg-config* which is the most reliable on systems
    that have the full developer package installed (e.g. `libopencv-dev` on
    Debian/Ubuntu).  If that fails we attempt a best-effort fallback by
    inspecting the wheel layout of the `opencv-python` PyPI package – while
    the wheels do not ship headers, that path can still be useful when the
    user has a custom build with headers inside the site-packages tree.
    """

    # Strategy 1: pkg-config (preferred)
    try:
        opencv_cflags = subprocess.check_output(
            ["pkg-config", "--cflags", "opencv4"], text=True
        ).split()
        opencv_ldflags = subprocess.check_output(
            ["pkg-config", "--libs", "opencv4"], text=True
        ).split()

        include_dirs = [flag[2:] for flag in opencv_cflags if flag.startswith("-I")]
        libs = [flag[2:] for flag in opencv_ldflags if flag.startswith("-l")]
        return include_dirs, libs
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass  # Fallback to wheel inspection

    # Strategy 2: Inspect opencv-python wheel (headers usually absent)
    try:
        import cv2  # Import locally to avoid hard runtime dependency

        cv2_root = Path(cv2.__file__).resolve().parent
        potential_include = cv2_root / "include"
        include_dirs = [str(potential_include)] if potential_include.exists() else []

        # Wheels link OpenCV statically; at build-time we still need the lib
        # names so that the linker picks up system libs when present.
        libs = [
            "opencv_core",
            "opencv_imgproc",
            "opencv_highgui",
            "opencv_imgcodecs",
        ]

        return include_dirs, libs
    except ImportError:
        print("Warning: OpenCV development files not found – building without OpenCV support.")
            return [], []

# Find spdlog
def find_spdlog():
    try:
        spdlog_cflags = subprocess.check_output(['pkg-config', '--cflags', 'spdlog']).decode().strip().split()
        spdlog_libs = subprocess.check_output(['pkg-config', '--libs', 'spdlog']).decode().strip().split()
        spdlog_include = [flag[2:] for flag in spdlog_cflags if flag.startswith('-I')]
        spdlog_libs = [flag[2:] for flag in spdlog_libs if flag.startswith('-l')]
        return spdlog_include, spdlog_libs
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: spdlog not found. Using header-only mode.")
        return [], []

# Get include directories and libraries
opencv_include, opencv_libs = find_opencv()
spdlog_include, spdlog_libs = find_spdlog()

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "opencam_auto_exposure",
        sources=[
            "pybind_auto_exposure.cpp",
            "../../algorithms/3a/auto_exposure.cpp",
            "../../core/src/camera.cpp",
            "../../algorithms/isp/debayer.cpp",
        ],
        include_dirs=[
            # Path to pybind11 headers
            pybind11.get_include(),
            # Project includes
            "../../",
            "../../include",
            "../../algorithms",
            "../../core/include",
        ] + opencv_include + spdlog_include,
        libraries=opencv_libs + spdlog_libs + ["pthread"],
        cxx_std="17",
        define_macros=[
            ("VERSION_INFO", f'"{get_version()}"'),
            ("PYBIND11_DETAILED_ERROR_MESSAGES", None),
        ],
        language="c++",
    ),
]

class CustomBuildExt(build_ext):
    """Custom build extension to handle compilation flags"""
    
    def build_extensions(self):
        # Add compiler-specific flags
        if self.compiler.compiler_type == 'unix':
            for ext in self.extensions:
                ext.extra_compile_args.extend([
                    '-O3',
                    '-Wall',
                    '-Wextra',
                    '-fPIC',
                    '-fopenmp',
                ])
                ext.extra_link_args.extend([
                    '-fopenmp',
                ])
        elif self.compiler.compiler_type == 'msvc':
            for ext in self.extensions:
                ext.extra_compile_args.extend([
                    '/O2',
                    '/openmp',
                ])
        
        super().build_extensions()

setup(
    name="opencam-auto-exposure",
    version=get_version(),
    author="Nik Jois",
    author_email="nikjois@llamasearch.ai",
    description="OpenCam Auto Exposure Algorithm Python Bindings",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/nikjois/opencam-auto-exposure",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": CustomBuildExt},
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "opencv-python>=4.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "isort>=5.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
        ],
        "ml": [
            "torch>=1.9.0",
            "torchvision>=0.10.0",
            "onnx>=1.9.0",
            "onnxruntime>=1.8.0",
        ],
        "api": [
            "fastapi>=0.68.0",
            "uvicorn>=0.15.0",
            "pydantic>=1.8.0",
            "openai>=1.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="computer-vision camera auto-exposure image-processing opencv",
    project_urls={
        "Bug Reports": "https://github.com/nikjois/opencam-auto-exposure/issues",
        "Source": "https://github.com/nikjois/opencam-auto-exposure",
        "Documentation": "https://opencam-auto-exposure.readthedocs.io/",
    },
) 