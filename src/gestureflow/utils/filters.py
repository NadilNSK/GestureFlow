"""
gestureflow.utils.filters
~~~~~~~~~~~~~~~~~~~~~~~~~
Signal processing and smoothing filters for virtual mouse control.

Implements:
- LowPassFilter: Standard 1st-order low-pass filter with time-dependent alpha.
- OneEuroFilter1D: 1D One Euro Filter (Casiez et al., CHI 2012).
- OneEuroFilter: 2D One Euro Filter for (x, y) coordinates with speed-adaptive
  jitter reduction at rest and minimal lag at high velocities.
"""

from __future__ import annotations

import math
import time
from typing import Optional


class LowPassFilter:
    """
    First-order low-pass filter with time-dependent smoothing.

    Parameters
    ----------
    alpha:
        Default smoothing factor in [0, 1].
    """

    def __init__(self, alpha: float = 0.5) -> None:
        self._alpha = max(0.0, min(1.0, float(alpha)))
        self._hat_x_prev: Optional[float] = None

    def reset(self) -> None:
        """Reset filter state."""
        self._hat_x_prev = None

    def filter(self, x: float, alpha: Optional[float] = None) -> float:
        """
        Filter a scalar value.

        Parameters
        ----------
        x:
            Raw input value.
        alpha:
            Optional override for the smoothing factor.

        Returns
        -------
        float
            Smoothed value.
        """
        a = self._alpha if alpha is None else max(0.0, min(1.0, float(alpha)))
        if self._hat_x_prev is None:
            self._hat_x_prev = float(x)
            return float(x)

        hat_x = a * float(x) + (1.0 - a) * self._hat_x_prev
        self._hat_x_prev = hat_x
        return hat_x

    @property
    def last_value(self) -> Optional[float]:
        """Most recent filtered value, or None if no samples filtered yet."""
        return self._hat_x_prev


class OneEuroFilter1D:
    """
    1-dimensional One Euro Filter for noise reduction and low-latency tracking.

    Reference:
    Casiez, G., Roussel, N., & Vogel, D. (2012).
    1 € filter: a simple speed-based low-pass filter for noisy input in interactive systems.
    Proceedings of the SIGCHI Conference on Human Factors in Computing Systems (CHI '12), 2527-2530.

    Parameters
    ----------
    min_cutoff:
        Minimum cutoff frequency (Hz). Lower values decrease jitter at low speeds (default 1.0).
    beta:
        Speed coefficient. Higher values reduce lag during fast movements (default 0.007).
    d_cutoff:
        Cutoff frequency for the derivative (Hz, default 1.0).
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ) -> None:
        self.min_cutoff = max(0.0001, float(min_cutoff))
        self.beta = max(0.0, float(beta))
        self.d_cutoff = max(0.0001, float(d_cutoff))

        self._x_filter = LowPassFilter()
        self._dx_filter = LowPassFilter()
        self._t_prev: Optional[float] = None
        self._x_prev: Optional[float] = None

    def reset(self) -> None:
        """Reset internal filter state."""
        self._x_filter.reset()
        self._dx_filter.reset()
        self._t_prev = None
        self._x_prev = None

    @staticmethod
    def _compute_alpha(cutoff: float, dt: float) -> float:
        """Calculate smoothing factor alpha = 1 / (1 + tau / dt) = 2*pi*fc*dt / (1 + 2*pi*fc*dt)."""
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: float, timestamp: Optional[float] = None) -> float:
        """
        Filter a single 1D coordinate with a timestamp.

        Parameters
        ----------
        x:
            Raw coordinate.
        timestamp:
            Monotonic timestamp in seconds. If None, time.monotonic() is used.

        Returns
        -------
        float
            Filtered coordinate.
        """
        t = time.monotonic() if timestamp is None else float(timestamp)

        if self._t_prev is None or self._x_prev is None:
            self._t_prev = t
            self._x_prev = float(x)
            return self._x_filter.filter(x, alpha=1.0)

        dt = t - self._t_prev
        # Guard against zero or negative or huge dt
        if dt <= 0.0:
            dt = 0.001
        elif dt > 1.0:
            # Huge pause: reset filter
            self.reset()
            self._t_prev = t
            self._x_prev = float(x)
            return self._x_filter.filter(x, alpha=1.0)

        # Estimate derivative (velocity)
        dx = (float(x) - self._x_prev) / dt
        alpha_d = self._compute_alpha(self.d_cutoff, dt)
        hat_dx = self._dx_filter.filter(dx, alpha=alpha_d)

        # Adaptive cutoff frequency based on velocity magnitude
        cutoff = self.min_cutoff + self.beta * abs(hat_dx)
        alpha = self._compute_alpha(cutoff, dt)

        # Filter the position
        hat_x = self._x_filter.filter(x, alpha=alpha)

        self._t_prev = t
        self._x_prev = float(x)
        return hat_x


class OneEuroFilter:
    """
    2-Dimensional One Euro Filter for (x, y) cursor coordinates.

    Parameters
    ----------
    min_cutoff:
        Minimum cutoff frequency (Hz, default 1.0).
    beta:
        Speed coefficient (default 0.007).
    d_cutoff:
        Derivative cutoff frequency (Hz, default 1.0).
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ) -> None:
        self._filter_x = OneEuroFilter1D(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        self._filter_y = OneEuroFilter1D(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)

    @property
    def min_cutoff(self) -> float:
        return self._filter_x.min_cutoff

    @min_cutoff.setter
    def min_cutoff(self, val: float) -> None:
        self._filter_x.min_cutoff = max(0.0001, float(val))
        self._filter_y.min_cutoff = max(0.0001, float(val))

    @property
    def beta(self) -> float:
        return self._filter_x.beta

    @beta.setter
    def beta(self, val: float) -> None:
        self._filter_x.beta = max(0.0, float(val))
        self._filter_y.beta = max(0.0, float(val))

    @property
    def d_cutoff(self) -> float:
        return self._filter_x.d_cutoff

    @d_cutoff.setter
    def d_cutoff(self, val: float) -> None:
        self._filter_x.d_cutoff = max(0.0001, float(val))
        self._filter_y.d_cutoff = max(0.0001, float(val))

    def reset(self) -> None:
        """Reset internal filter state on tracking interruption or jump."""
        self._filter_x.reset()
        self._filter_y.reset()

    def filter(
        self,
        x: float,
        y: float,
        timestamp: Optional[float] = None,
    ) -> tuple[float, float]:
        """
        Filter 2D coordinates.

        Parameters
        ----------
        x, y:
            Raw coordinates (screen pixels or normalized space).
        timestamp:
            Monotonic timestamp in seconds (optional).

        Returns
        -------
        tuple[float, float]
            Smoothed (x, y) coordinates.
        """
        fx = self._filter_x.filter(x, timestamp=timestamp)
        fy = self._filter_y.filter(y, timestamp=timestamp)
        return (fx, fy)
