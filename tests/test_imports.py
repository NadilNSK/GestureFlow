"""
Smoke-test: verify all public gestureflow modules can be imported cleanly.
"""

from __future__ import annotations

import importlib

import pytest

MODULES = [
    "gestureflow",
    "gestureflow.config",
    "gestureflow.config.settings",
    "gestureflow.utils",
    "gestureflow.utils.logging_setup",
    "gestureflow.utils.helpers",
    "gestureflow.camera",
    "gestureflow.camera.capture",
    "gestureflow.tracking",
    "gestureflow.tracking.hand_landmarker",
    "gestureflow.tracking.pipeline",
    "gestureflow.gestures",
    "gestureflow.gestures.classifier",
    "gestureflow.controller",
    "gestureflow.controller.mouse_controller",
    "gestureflow.calibration",
    "gestureflow.calibration.calibrator",
    "gestureflow.ui",
    "gestureflow.ui.app",
    "gestureflow.ui.main_window",
    "gestureflow.ui.workers",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name: str) -> None:
    """Each module must import without error."""
    mod = importlib.import_module(module_name)
    assert mod is not None
