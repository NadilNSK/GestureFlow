"""
tests.test_phase4_2_diagnosis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Automated verification tests for Phase 4.2:
Evidence-Based FPS Bottleneck Diagnosis & Camera Backend Selection.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, call, patch
import cv2
import numpy as np
import pytest

from gestureflow.camera.capture import (
    CameraCapture,
    CameraFrame,
    discover_available_cameras,
)
from gestureflow.config.settings import CameraSettings, Settings
from gestureflow.tracking.pipeline import PipelineFrame, TrackingPipeline


class TestCameraBackendAndFourcc:
    @patch("cv2.VideoCapture")
    def test_discover_available_cameras_msmf(self, mock_vc_cls):
        """discover_available_cameras with backend='MSMF' uses cv2.CAP_MSMF."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_vc_cls.return_value = mock_cap

        discover_available_cameras(max_indices=2, backend="MSMF")
        mock_vc_cls.assert_any_call(0, cv2.CAP_MSMF)
        mock_vc_cls.assert_any_call(1, cv2.CAP_MSMF)

    @patch("cv2.VideoCapture")
    def test_discover_available_cameras_dshow(self, mock_vc_cls):
        """discover_available_cameras with backend='DSHOW' uses cv2.CAP_DSHOW."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_vc_cls.return_value = mock_cap

        discover_available_cameras(max_indices=2, backend="DSHOW")
        mock_vc_cls.assert_any_call(0, cv2.CAP_DSHOW)
        mock_vc_cls.assert_any_call(1, cv2.CAP_DSHOW)

    @patch("cv2.VideoCapture")
    def test_open_camera_fourcc_ordering(self, mock_vc_cls):
        """_open_camera sets width, height, fps before setting CAP_PROP_FOURCC."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 640
        mock_vc_cls.return_value = mock_cap

        settings = CameraSettings(
            device_index=0,
            backend="DSHOW",
            frame_width=640,
            frame_height=480,
            fps=30,
        )
        capture = CameraCapture(settings)
        capture._open_camera()

        # Check call order on mock_cap.set
        set_calls = [c[0] for c in mock_cap.set.call_args_list]
        props_called = [args[0] for args in set_calls]

        assert cv2.CAP_PROP_FRAME_WIDTH in props_called
        assert cv2.CAP_PROP_FRAME_HEIGHT in props_called
        assert cv2.CAP_PROP_FPS in props_called
        assert cv2.CAP_PROP_FOURCC in props_called

        idx_width = props_called.index(cv2.CAP_PROP_FRAME_WIDTH)
        idx_height = props_called.index(cv2.CAP_PROP_FRAME_HEIGHT)
        idx_fps = props_called.index(cv2.CAP_PROP_FPS)
        idx_fourcc = props_called.index(cv2.CAP_PROP_FOURCC)

        # fourcc must come AFTER width, height, fps
        assert idx_fourcc > idx_width
        assert idx_fourcc > idx_height
        assert idx_fourcc > idx_fps


class TestFrameAgeAccuracy:
    def test_post_capture_frame_age_validity(self, default_settings):
        """frame_age_ms in PipelineFrame reflects post-read timestamp accuracy."""
        pipeline = TrackingPipeline(default_settings)
        pipeline._landmarker = MagicMock()
        pipeline._landmarker.detect.return_value = MagicMock(hands=[])

        # Simulate frame captured right now
        t_capture = time.perf_counter()
        cam_frame = CameraFrame(
            image=np.zeros((480, 640, 3), dtype=np.uint8),
            timestamp_ms=int(t_capture * 1000),
            frame_index=1,
            fps=30.0,
            capture_perf_time=t_capture,
        )

        p_frame = pipeline.process(cam_frame)
        # Pipeline frame age should be minimal (typically < 50ms, usually < 5ms in test)
        assert p_frame.frame_age_ms >= 0.0
        assert p_frame.frame_age_ms < 100.0  # Must not reflect artificial pre-read blocking
