'''
Baseline (reference) implementation of a naïve auto-exposure algorithm.
This algorithm is intentionally simple and purely Python so that it can act as
an easy-to-beat reference for performance and quality evaluations of the real
OpenCam implementation.

Author: Nik Jois <nikjois@llamasearch.ai>
'''

from __future__ import annotations

import numpy as np
from typing import Any

__all__ = ["BaselineAutoExposure"]


class BaselineAutoExposure:  # pylint: disable=too-few-public-methods
    """A trivial auto-exposure controller.

    The controller computes image brightness as the mean of all pixels and then
    derives an exposure value that tries to bring the brightness close to 0.5
    in the [0,1] range.

    The formula is intentionally simple so that the implementation is easy to
    reason about and fast to execute in pure Python.  This makes it a useful
    baseline for *both* performance and quality comparisons.
    """

    def __init__(self, target_brightness: float = 0.5) -> None:
        if not 0.0 < target_brightness <= 1.0:
            raise ValueError("target_brightness must be in (0, 1]")
        self.target_brightness = float(target_brightness)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def compute_exposure(self, image: np.ndarray | Any) -> float:  # noqa: ANN401
        """Return an exposure multiplier for *image*.

        The exposure is computed as::

            exposure = target_brightness / max(current_brightness, eps)

        where *current_brightness* is the mean pixel value normalised to
        ``[0,1]``.  The result is clamped to ``[0.1, 1.0]`` so that exposure
        never drops to zero or blows up too much.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy.ndarray")

        if image.size == 0:
            raise ValueError("image cannot be empty")

        # Normalise to [0,1] regardless of dtype.
        if image.dtype == np.uint8:
            brightness = image.mean() / 255.0
        elif image.dtype == np.uint16:
            brightness = image.mean() / 65535.0
        else:
            # Assume float in [0,1] or wider.
            brightness = float(image.mean())
            if brightness > 1.0:
                brightness /= 255.0

        brightness = max(brightness, 1e-4)  # prevent divide-by-zero.
        exposure = self.target_brightness / brightness
        return float(np.clip(exposure, 0.1, 1.0)) 