"""
Tests for stubs: mouse_controller, classifier, calibrator, camera capture
"""

from __future__ import annotations

import pytest
import numpy as np

from gestureflow.controller.mouse_controller import MouseController
from gestureflow.gestures.classifier import GestureClassifier, Gesture, GestureResult
from gestureflow.calibration.calibrator import Calibrator, CalibrationData
from gestureflow.camera.capture import CameraFrame, CameraCapture
from gestureflow.config.settings import CameraSettings


class TestMouseController:
    def test_failsafe_invariant(self):
        controller = MouseController()
        if controller._pag is not None:
            assert controller._pag.FAILSAFE is True

    def test_stubs_execute_without_error(self):
        controller = MouseController()
        controller.move(100, 200)
        controller.click()
        controller.right_click()
        controller.scroll(3)


class TestGestureClassifier:
    def test_stub_returns_none_gesture(self):
        classifier = GestureClassifier()
        # Mock hand
        class DummyHand:
            pass

        result = classifier.classify(DummyHand())
        assert isinstance(result, GestureResult)
        assert result.gesture == Gesture.NONE


class TestCalibrator:
    def test_defaults(self):
        cal = Calibrator()
        assert cal.data.x_min == 0.1
        assert cal.data.x_max == 0.9
        assert cal.data.y_min == 0.1
        assert cal.data.y_max == 0.9

    def test_reset(self):
        cal = Calibrator()
        cal._data.x_min = 0.3
        cal.reset()
        assert cal.data.x_min == 0.1


class TestCameraFrame:
    def test_camera_frame_fields(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = CameraFrame(
            image=img,
            timestamp_ms=150,
            frame_index=5,
            fps=30.0,
        )
        assert frame.image.shape == (480, 640, 3)
        assert frame.timestamp_ms == 150
        assert frame.frame_index == 5
        assert frame.fps == 30.0
