"""
gestureflow.calibration.calibrator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Calibration pipeline stub — not active in Phase 1.
Phase 3 will let users calibrate the hand-to-screen mapping zone.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class CalibrationData:
    """Boundaries of the user's hand movement zone (normalised 0–1)."""

    x_min: float = 0.1
    x_max: float = 0.9
    y_min: float = 0.1
    y_max: float = 0.9


class Calibrator:
    """Collects calibration samples and computes the movement zone."""

    def __init__(self) -> None:
        self._data = CalibrationData()

    @property
    def data(self) -> CalibrationData:
        return self._data

    def reset(self) -> None:
        self._data = CalibrationData()
        log.info("Calibration reset to defaults.")
