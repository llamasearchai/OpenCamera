#!/usr/bin/env python3
"""Performance comparison tests between OpenCam implementation and a baseline.

These tests compare the throughput of the compiled C++ \ ``AutoExposure``
controller (exposed to Python via *pybind11*) against a very small, pure-Python
reference implementation ``BaselineAutoExposure``.  The goal is to ensure that
we never introduce changes that make the production implementation slower than
this trivial baseline.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

from __future__ import annotations

import time
from statistics import mean
from typing import Callable

import numpy as np
import pytest

try:
    from opencam import AutoExposure, BaselineAutoExposure  # type: ignore
except ImportError:  # pragma: no-cover – will be skipped below
    AutoExposure = None  # type: ignore
    BaselineAutoExposure = None  # type: ignore


ITERATIONS = 25  # Number of frames per controller during the test.
IMAGE_RESOLUTION = (1920, 1080)  # Full-HD: a realistic workload.


def _measure(func: Callable[[np.ndarray], float], img: np.ndarray) -> float:
    """Return *average* execution time (seconds) over *ITERATIONS* runs."""
    timings = []
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        func(img)
        timings.append(time.perf_counter() - start)
    return mean(timings)


@pytest.fixture(scope="module")
def _test_image() -> np.ndarray:
    """Reusable random image with a medium brightness."""
    rng = np.random.default_rng(seed=1234)
    return rng.integers(0, 255, (*IMAGE_RESOLUTION[::-1], 3), dtype=np.uint8)


@pytest.mark.benchmark
def test_against_baseline_speed(_test_image: np.ndarray):
    """Ensure the production implementation is at least as fast as the baseline."""
    if AutoExposure is None or BaselineAutoExposure is None:  # pragma: no-cover
        pytest.skip("OpenCam C++ extension not available; skipping perf test.")

    try:
        prod_ctrl = AutoExposure()  # type: ignore [call-arg]
    except ImportError:
        pytest.skip("OpenCam C++ bindings missing; skipping perf test.")

    base_ctrl = BaselineAutoExposure()

    prod_time = _measure(prod_ctrl.compute_exposure, _test_image)
    base_time = _measure(base_ctrl.compute_exposure, _test_image)

    # The production implementation should be *strictly* faster, but to account
    # for CI noise we allow a 10 % margin.
    assert prod_time <= base_time * 1.10, (
        f"Production AutoExposure slower than baseline: {prod_time*1e6:.1f} µs vs "
        f"{base_time*1e6:.1f} µs"
    )


@pytest.mark.benchmark
def test_against_baseline_quality(_test_image: np.ndarray):
    """Verify that the exposure values from both implementations are similar.

    The production algorithm is expected to be *better* than the baseline, but
    at the very least its output should be *consistent* (same order of
    magnitude) for identical inputs.  This guards against regressions that
    would produce wildly incorrect exposure multipliers.
    """
    if AutoExposure is None or BaselineAutoExposure is None:  # pragma: no-cover
        pytest.skip("OpenCam C++ extension not available; skipping quality test.")

    try:
        prod_ctrl = AutoExposure()  # type: ignore [call-arg]
    except ImportError:
        pytest.skip("OpenCam C++ bindings missing; skipping quality test.")

    base_ctrl = BaselineAutoExposure()

    prod_exposure = prod_ctrl.compute_exposure(_test_image)
    base_exposure = base_ctrl.compute_exposure(_test_image)

    # The expected exposure should not differ by more than 2×.  A tighter bound
    # may be used once the ML-based algorithm is fully validated.
    ratio = max(prod_exposure, base_exposure) / min(prod_exposure, base_exposure)
    assert ratio <= 2.0, (
        "Exposure value deviates too much from baseline "
        f"(prod={prod_exposure:.3f}, base={base_exposure:.3f})"
    ) 