"""
OpenCam Auto Exposure Algorithm Python Package
Author: Nik Jois <nikjois@llamasearch.ai>

This package provides Python bindings for the OpenCam auto exposure algorithm,
including high-level APIs for easy integration.
"""

__version__ = "1.0.0"
__author__ = "Nik Jois"
__email__ = "nikjois@llamasearch.ai"

try:
    import opencam_auto_exposure as _cpp
    from .auto_exposure import AutoExposure, MeteringMode, Parameters
    from .utils import create_test_frame, load_image, save_image
    from .benchmark import AutoExposureBenchmark
    from .baseline_exposure import BaselineAutoExposure
    
    __all__ = [
        "AutoExposure",
        "MeteringMode", 
        "Parameters",
        "create_test_frame",
        "load_image",
        "save_image",
        "BaselineAutoExposure",
        "AutoExposureBenchmark",
        "opencam_auto_exposure",
    ]
    
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import OpenCam C++ extension: {e}. "
                  "Python fall-back stubs will be used. Build the extension via 'pip install -e .' to enable full performance.")
    
    # Provide stub implementations for development
    class AutoExposure:
        def __init__(self):
            raise ImportError("OpenCam C++ extension not available")
    
    from .baseline_exposure import BaselineAutoExposure  # type: ignore
    from .utils import create_test_frame, load_image, save_image  # type: ignore

    __all__ = [
        "AutoExposure",
        "create_test_frame",
        "load_image",
        "save_image",
        "BaselineAutoExposure",
    ] 