"""
Tests for gestureflow.tracking (HandLandmarkerWrapper and TrackingPipeline)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from gestureflow.config.settings import Settings, TrackingSettings
from gestureflow.camera.capture import CameraFrame
from gestureflow.tracking.hand_landmarker import (
    HandLandmarkerWrapper,
    TrackingResult,
    HandLandmarks,
)
from gestureflow.tracking.pipeline import TrackingPipeline, PipelineFrame


class TestHandLandmarkerWrapper:
    def test_missing_model_raises_file_not_found(self, tmp_path):
        bad_path = tmp_path / "nonexistent.task"
        wrapper = HandLandmarkerWrapper(TrackingSettings(), bad_path)
        with pytest.raises(FileNotFoundError, match="Hand Landmarker model not found"):
            wrapper.open()

    def test_context_manager_lifecycle(self, default_settings):
        if not default_settings.model_path.exists():
            pytest.skip("Model file not downloaded yet")

        wrapper = HandLandmarkerWrapper(default_settings.tracking, default_settings.model_path)
        with wrapper as w:
            assert w._landmarker is not None
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            result = w.detect(frame, timestamp_ms=0)
            assert isinstance(result, TrackingResult)
            assert result.has_hands is False
            assert result.primary_hand is None
        assert wrapper._landmarker is None

    def test_detect_without_open_raises_runtime_error(self, default_settings):
        wrapper = HandLandmarkerWrapper(default_settings.tracking, default_settings.model_path)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="open.*must be called first"):
            wrapper.detect(frame, 0)


class TestTrackingPipeline:
    def test_pipeline_process_frame(self, default_settings):
        if not default_settings.model_path.exists():
            pytest.skip("Model file not downloaded yet")

        pipeline = TrackingPipeline(default_settings)
        pipeline.open()
        try:
            bgr = np.zeros((480, 640, 3), dtype=np.uint8)
            cam_frame = CameraFrame(
                image=bgr,
                timestamp_ms=100,
                frame_index=1,
                fps=30.0,
            )
            processed = pipeline.process(cam_frame)
            assert isinstance(processed, PipelineFrame)
            assert processed.display_image.shape == (480, 640, 3)
            assert processed.fps == 30.0
            assert processed.frame_index == 1
            assert processed.tracking.has_hands is False
        finally:
            pipeline.close()
