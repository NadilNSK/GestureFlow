"""
gestureflow.utils.helpers
~~~~~~~~~~~~~~~~~~~~~~~~~~
Miscellaneous utility functions used across the project.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Callable, TypeVar

from gestureflow.utils.filters import LowPassFilter, OneEuroFilter, OneEuroFilter1D

T = TypeVar("T")


class FpsCounter:
    """
    Rolling-window frames-per-second counter.

    Parameters
    ----------
    window:
        Number of frames to average over (default 30).
    """

    def __init__(self, window: int = 30) -> None:
        self._timestamps: deque[float] = deque(maxlen=window)

    def tick(self) -> float:
        """Record a new frame and return the current FPS."""
        now = time.monotonic()
        self._timestamps.append(now)
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed

    @property
    def fps(self) -> float:
        """Current FPS without recording a new frame."""
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed


def clamp(value: float, lo: float, hi: float) -> float:
    """Return *value* clamped to the range [lo, hi]."""
    return max(lo, min(hi, value))


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between *a* and *b* by factor *t* ∈ [0, 1]."""
    return a + (b - a) * t


def retry(
    fn: Callable[[], T],
    attempts: int = 3,
    delay: float = 0.5,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> T:
    """
    Call *fn* up to *attempts* times, sleeping *delay* seconds between tries.

    Raises the last exception if all attempts fail.
    """
    last_exc: Exception | None = None
    for _ in range(attempts):
        try:
            return fn()
        except exceptions as exc:
            last_exc = exc
            time.sleep(delay)
    raise RuntimeError(f"All {attempts} attempts failed") from last_exc
