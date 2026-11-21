"""
tests.test_phase4_1_stability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Automated tests for Phase 4.1:
Webcam Tracking Stability, FPS Optimization & Dashboard Refinement.

Covers all 15 required verification scenarios:
1.  Temporary landmark loss (1-2 frames grace period)
2.  Genuine hand disappearance (transition to HAND LOST)
3.  Hand reacquisition after temporary loss
4.  Primary-hand spatial continuity
5.  Gesture stability after reacquisition
6.  Tracking quality rolling calculation
7.  Frame-age measurement
8.  Camera FPS and processing FPS calculations
9.  Bounded frame processing (no backlog accumulation)
10. UI diagnostics and telemetry updates
11. Dashboard activity panel event formatting
12. Mouse automation safety during tracking loss
13. Drag release on tracking loss
14. Emergency Stop aborts automation
15. Camera 0 selection persistence
"""

from __future__ import annotations

import collections
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from gestureflow.camera.capture import CameraCapture, CameraFrame
from gestureflow.config.settings import Settings, CameraSettings, TrackingSettings
from gestureflow.controller.mouse_controller import MouseController, MouseState
from gestureflow.gestures.classifier import (
    FingerStates,
    Gesture,
    GestureClassifier,
    GestureResult,
)
from gestureflow.tracking.hand_landmarker import (
    HandLandmarks,
    HandLandmarkerWrapper,
    TrackingResult,
)
from gestureflow.tracking.pipeline import TrackingPipeline, PipelineFrame
from gestureflow.ui.main_window import MainWindow
from gestureflow.ui.pages.dashboard_page import DashboardPage


# Helper dummy landmark generator
def make_mock_hand(
    wrist_x: float = 0.5,
    wrist_y: float = 0.5,
    score: float = 0.90,
    handedness: str = "Right",
) -> HandLandmarks:
    """Create a mock HandLandmarks object with 21 normalized landmarks."""
    lms = []
    for i in range(21):
        # Create Landmark-like object with x, y, z
        mock_lm = MagicMock()
        mock_lm.x = wrist_x + (i * 0.005)
        mock_lm.y = wrist_y + (i * 0.005)
        mock_lm.z = 0.0
        lms.append(mock_lm)

    return HandLandmarks(
        landmarks=lms,
        world_landmarks=lms,
        handedness=handedness,
        score=score,
    )


@pytest.fixture
def clean_settings(project_root, tmp_path):
    """Provide isolated Settings instance with user_settings in tmp_path."""
    user_file = tmp_path / "user_settings.json"
    settings = Settings.load(
        defaults_path=project_root / "default_settings.json",
        user_path=user_file,
    )
    settings.ui.onboarding_completed = True
    return settings


class TestTrackingStabilityAndContinuity:
    """Tests 1 through 6: Tracking state machine, grace periods, and spatial continuity."""

    def test_temporary_landmark_loss_grace_period(self, clean_settings):
        """Scenario 1: 1-2 frames of missing hand do not transition to HAND LOST."""
        pipeline = TrackingPipeline(clean_settings)
        # Mock landmarker
        pipeline._landmarker = MagicMock()

        # Frame 1: Hand present -> HAND DETECTED
        hand = make_mock_hand(0.5, 0.5)
        pipeline._landmarker.detect.return_value = TrackingResult(hands=[hand], timestamp_ms=33)
        f1 = pipeline.process(CameraFrame(np.zeros((480, 640, 3), dtype=np.uint8), 33, 1, 30.0))
        assert pipeline.tracking_state == "HAND DETECTED"
        assert f1.tracking_state == "HAND DETECTED"
        assert f1.active_hand is not None

        # Frame 2: Missing (1 frame) -> TRACKING UNSTABLE (grace period)
        pipeline._landmarker.detect.return_value = TrackingResult(hands=[], timestamp_ms=66)
        f2 = pipeline.process(CameraFrame(np.zeros((480, 640, 3), dtype=np.uint8), 66, 2, 30.0))
        assert pipeline.tracking_state == "TRACKING UNSTABLE"
        assert f2.tracking_state == "TRACKING UNSTABLE"
        assert f2.tracking_quality == "UNSTABLE"
        assert f2.active_hand is None
        assert pipeline.consecutive_lost_frames == 1

        # Frame 3: Missing (2 frames) -> REACQUIRING (grace period)
        pipeline._landmarker.detect.return_value = TrackingResult(hands=[], timestamp_ms=99)
        f3 = pipeline.process(CameraFrame(np.zeros((480, 640, 3), dtype=np.uint8), 99, 3, 30.0))
        assert pipeline.tracking_state == "REACQUIRING"
        assert f3.tracking_state == "REACQUIRING"
        assert f3.tracking_quality == "REACQUIRING"
        assert f3.active_hand is None
        assert pipeline.consecutive_lost_frames == 2

    def test_genuine_hand_disappearance_hand_lost(self, clean_settings):
        """Scenario 2: >= 3 consecutive frames without hands transition state to HAND LOST."""
        pipeline = TrackingPipeline(clean_settings)
        pipeline._landmarker = MagicMock()

        # Start with hand present
        hand = make_mock_hand(0.5, 0.5)
        pipeline._landmarker.detect.return_value = TrackingResult(hands=[hand], timestamp_ms=33)
        pipeline.process(CameraFrame(np.zeros((480, 640, 3), dtype=np.uint8), 33, 1, 30.0))
        assert pipeline.tracking_state == "HAND DETECTED"

        # 3 consecutive frames with no hand
        pipeline._landmarker.detect.return_value = TrackingResult(hands=[], timestamp_ms=66)
        pipeline.process(CameraFrame(np.zeros((480, 640, 3), dtype=np.uint8), 66, 2, 30.0))  # 1: UNSTABLE
        pipeline.process(CameraFrame(np.zeros((480, 640, 3), dtype=np.uint8), 99, 3, 30.0))  # 2: REACQUIRING
        f4 = pipeline.process(CameraFrame(np.zeros((480, 640, 3), dtype=np.uint8), 132, 4, 30.0))  # 3: HAND LOST

        assert pipeline.tracking_state == "HAND LOST"
        assert f4.tracking_state == "HAND LOST"
        assert f4.tracking_quality == "LOST"
        assert pipeline.consecutive_lost_frames == 3
        assert pipeline.tracking_lost_count == 1
        assert pipeline._prev_active_wrist is None

    def test_hand_reacquisition(self, clean_settings):
        """Scenario 3: Hand returns after temporary loss -> state returns to HAND DETECTED, reacquired_count increments."""
        pipeline = TrackingPipeline(clean_settings)
        pipeline._landmarker = MagicMock()

        # Frame 1: Present
        hand = make_mock_hand(0.5, 0.5)
        pipeline._landmarker.detect.return_value = TrackingResult(hands=[hand], timestamp_ms=33)
        pipeline.process(CameraFrame(np.zeros((480, 640, 3), dtype=np.uint8), 33, 1, 30.0))

        # Frame 2: 1-frame dropout
        pipeline._landmarker.detect.return_value = TrackingResult(hands=[], timestamp_ms=66)
        pipeline.process(CameraFrame(np.zeros((480, 640, 3), dtype=np.uint8), 66, 2, 30.0))
        assert pipeline.tracking_state == "TRACKING UNSTABLE"

        # Frame 3: Reacquired!
        pipeline._landmarker.detect.return_value = TrackingResult(hands=[hand], timestamp_ms=99)
        f3 = pipeline.process(CameraFrame(np.zeros((480, 640, 3), dtype=np.uint8), 99, 3, 30.0))

        assert pipeline.tracking_state == "HAND DETECTED"
        assert f3.tracking_state == "HAND DETECTED"
        assert pipeline.reacquired_count == 1
        assert pipeline.consecutive_lost_frames == 0
        assert f3.active_hand is not None

    def test_primary_hand_continuity(self, clean_settings):
        """Scenario 4: When two hands appear, previously tracked hand is maintained based on spatial distance."""
        pipeline = TrackingPipeline(clean_settings)
        pipeline._landmarker = MagicMock()

        # Track hand A at (0.3, 0.4)
        hand_a = make_mock_hand(wrist_x=0.3, wrist_y=0.4, handedness="Right", score=0.88)
        pipeline._landmarker.detect.return_value = TrackingResult(hands=[hand_a], timestamp_ms=33)
        f1 = pipeline.process(CameraFrame(np.zeros((480, 640, 3), dtype=np.uint8), 33, 1, 30.0))
        assert f1.active_hand.landmarks[0].x == pytest.approx(0.3)

        # In next frame, two hands appear: Hand A slightly moved to (0.31, 0.41), and Hand B at (0.75, 0.50) with higher score
        hand_a_moved = make_mock_hand(wrist_x=0.31, wrist_y=0.41, handedness="Right", score=0.80)
        hand_b = make_mock_hand(wrist_x=0.75, wrist_y=0.50, handedness="Left", score=0.99)

        pipeline._landmarker.detect.return_value = TrackingResult(hands=[hand_b, hand_a_moved], timestamp_ms=66)
        f2 = pipeline.process(CameraFrame(np.zeros((480, 640, 3), dtype=np.uint8), 66, 2, 30.0))

        # Despite hand_b having a higher score (0.99 vs 0.80), spatial continuity selects hand_a_moved!
        assert f2.active_hand.landmarks[0].x == pytest.approx(0.31)
        assert pipeline.continuity_rejected_count == 0

    def test_gesture_stability_after_reacquisition(self, clean_settings):
        """Scenario 5: Gesture classifier maintains stable recognition across 1-frame dropout."""
        classifier = GestureClassifier(clean_settings.gestures)

        hand = make_mock_hand()
        # Mock finger states for pointing
        f_pointing = FingerStates(thumb=False, index=True, middle=False, ring=False, pinky=False)
        with patch("gestureflow.gestures.classifier.detect_finger_states", return_value=f_pointing):
            # Prime pointing over 3 frames
            for _ in range(3):
                r1 = classifier.classify(hand)
            assert r1.gesture == Gesture.POINTING
            assert r1.is_stable is True

            # 1 frame absence (None hand) -> Holds gesture during loss tolerance
            r_none = classifier.classify(None)
            assert r_none.gesture == Gesture.POINTING
            assert "Hold" in r_none.label

            # Hand returns next frame
            r_return = classifier.classify(hand)
            assert r_return.gesture == Gesture.POINTING
            assert r_return.confidence >= 0.70

    def test_tracking_quality_rolling_calculation(self, clean_settings):
        """Scenario 6: Rolling detection ratio correctly tracks detection percentage over 30-frame window."""
        pipeline = TrackingPipeline(clean_settings)
        pipeline._landmarker = MagicMock()

        hand = make_mock_hand()
        pipeline._landmarker.detect.return_value = TrackingResult(hands=[hand], timestamp_ms=33)

        # 24 frames with hands detected
        for i in range(24):
            pipeline.process(CameraFrame(np.zeros((480, 640, 3), dtype=np.uint8), (i + 1) * 33, i + 1, 30.0))

        # 6 frames without hands
        pipeline._landmarker.detect.return_value = TrackingResult(hands=[], timestamp_ms=33)
        for i in range(24, 30):
            f = pipeline.process(CameraFrame(np.zeros((480, 640, 3), dtype=np.uint8), (i + 1) * 33, i + 1, 30.0))

        # In rolling 30-frame window: 24 True, 6 False -> 24 / 30 = 0.80 (80%)
        assert pipeline.detection_ratio == pytest.approx(0.80)
        assert f.detection_ratio == pytest.approx(0.80)
        assert pipeline.mediapipe_no_hands_count == 6


class TestPerformanceAndQueueDecoupling:
    """Tests 7 through 9: Frame age, FPS metrics, and decoupled bounded processing."""

    def test_frame_age_measurement(self, clean_settings):
        """Scenario 7: Frame age is measured in milliseconds from capture to pipeline processing."""
        pipeline = TrackingPipeline(clean_settings)
        pipeline._landmarker = MagicMock()
        pipeline._landmarker.detect.return_value = TrackingResult(hands=[], timestamp_ms=100)

        # Simulated capture occurred 15ms in the past
        t_capture = time.perf_counter() - 0.015
        cam_frame = CameraFrame(
            image=np.zeros((480, 640, 3), dtype=np.uint8),
            timestamp_ms=100,
            frame_index=1,
            fps=30.0,
            capture_perf_time=t_capture,
        )

        f = pipeline.process(cam_frame)
        assert f.frame_age_ms >= 12.0  # Approx 15ms with tolerance
        assert f.frame_age_ms < 500.0

    def test_camera_fps_and_processing_fps(self, clean_settings):
        """Scenario 8: Pipeline calculates both camera FPS and processing FPS accurately."""
        pipeline = TrackingPipeline(clean_settings)
        pipeline._landmarker = MagicMock()
        pipeline._landmarker.detect.return_value = TrackingResult(hands=[], timestamp_ms=100)

        cam_frame = CameraFrame(
            image=np.zeros((480, 640, 3), dtype=np.uint8),
            timestamp_ms=100,
            frame_index=1,
            fps=29.8,
            capture_perf_time=time.perf_counter(),
        )

        f = pipeline.process(cam_frame)
        assert f.camera_fps == pytest.approx(29.8)
        assert f.processing_fps >= 0.0

    def test_bounded_frame_processing(self):
        """Scenario 9: Decoupled 1-slot buffer discards stale frames and prevents backlog."""
        from gestureflow.ui.workers import PipelineWorker

        settings = Settings()
        worker = PipelineWorker(settings)
        worker._running = True

        # Simulate camera thread pushing 10 frames rapidly
        for i in range(10):
            frame = CameraFrame(
                image=np.zeros((480, 640, 3), dtype=np.uint8),
                timestamp_ms=i * 33,
                frame_index=i,
                fps=30.0,
                capture_perf_time=time.perf_counter(),
            )
            worker._on_frame(frame)

        # Only the latest frame (index 9) must be retained
        with worker._frame_lock:
            latest = worker._latest_camera_frame
            assert latest is not None
            assert latest.frame_index == 9


class TestUiDashboardAndSafetyIntegration:
    """Tests 10 through 15: UI telemetry, activity feed, mouse safety, and camera persistence."""

    def test_ui_diagnostics_updates(self, qapp, clean_settings):
        """Scenario 10: MainWindow updates telemetry cards and tracking status pills upon receiving PipelineFrame."""
        window = MainWindow(settings=clean_settings)
        dummy_bgr = np.zeros((480, 640, 3), dtype=np.uint8)

        frame = PipelineFrame(
            display_image=dummy_bgr,
            tracking=TrackingResult(hands=[], timestamp_ms=100),
            fps=30.0,
            frame_index=1,
            tracking_quality="GOOD",
            tracking_state="HAND DETECTED",
            detection_ratio=0.95,
            processing_fps=29.5,
            camera_fps=30.0,
        )

        window._on_frame_ready(frame)
        assert "ACTIVE (95%)" in window._page_dashboard._pill_tracking.text()
        assert window._page_dashboard._card_proc_fps._lbl_value.text() == "29.5"
        assert window._page_dashboard._card_cam_fps._lbl_value.text() == "30.0"
        window.close()

    def test_activity_panel_event_formatting(self, qapp, clean_settings):
        """Scenario 11: Dashboard activity panel formats events with timestamps, badges, and clean HTML."""
        page = DashboardPage(settings=clean_settings)
        page.log_activity("Webcam DirectShow stream started", "cyan")
        page.log_activity("Left Click executed", "success")
        page.log_activity("Tracking degraded due to occlusion", "warning")
        page.log_activity("Emergency stop triggered", "danger")

        assert len(page._activity_log) == 4
        # Verify badges are present
        assert "[SYSTEM]" in page._activity_log[0]
        assert "[ACTION]" in page._activity_log[1]
        assert "[WARN]" in page._activity_log[2]
        assert "[SAFETY]" in page._activity_log[3]

    def test_mouse_safety_during_tracking_loss(self, clean_settings):
        """Scenario 12: Virtual mouse halts cursor movement immediately on any frame with no hand."""
        controller = MouseController(settings=clean_settings.controller, mock_mode=False)
        controller.enable()
        controller._pag = MagicMock()
        controller._pag.FAILSAFE = True
        controller._pag.position.return_value = (500, 500)

        # When active_hand is None, process_hand returns None and does NOT move cursor
        res = controller.process_hand(None, None)
        assert res is None
        controller._pag.moveTo.assert_not_called()

    def test_drag_release_on_tracking_loss(self, clean_settings):
        """Scenario 13: If dragging when hand disappears for >= 3 frames, mouse held buttons are unconditionally released."""
        pipeline = TrackingPipeline(clean_settings)
        pipeline._landmarker = MagicMock()
        pipeline._mouse_controller._pag = MagicMock()
        pipeline._mouse_controller._pag.FAILSAFE = True
        pipeline._mouse_controller._is_dragging = True
        pipeline._mouse_controller._drag_button = "left"

        # Missing for 3 frames
        pipeline._landmarker.detect.return_value = TrackingResult(hands=[], timestamp_ms=33)
        pipeline.process(CameraFrame(np.zeros((480, 640, 3), dtype=np.uint8), 33, 1, 30.0))
        pipeline.process(CameraFrame(np.zeros((480, 640, 3), dtype=np.uint8), 66, 2, 30.0))
        pipeline.process(CameraFrame(np.zeros((480, 640, 3), dtype=np.uint8), 99, 3, 30.0))

        # Verify mouseButtonUp / release_held_buttons was executed
        assert pipeline._mouse_controller.is_dragging is False
        pipeline._mouse_controller._pag.mouseUp.assert_called_with(button="left")

    def test_emergency_stop_aborts_automation(self, qapp, clean_settings):
        """Scenario 14: Emergency stop immediately disables virtual mouse and halts automation."""
        window = MainWindow(settings=clean_settings)
        window._worker = MagicMock()

        window._on_emergency_stop()
        assert window._btn_reset_emergency.isEnabled() is True
        assert window._btn_mouse_enable.isEnabled() is False
        assert "EMERGENCY STOP" in window._lbl_mouse_status.text()
        window._worker.emergency_stop.assert_called_once()
        window.close()

    def test_camera_0_selection_persistence(self, clean_settings, tmp_path):
        """Scenario 15: Settings default to Camera 0 (DSHOW) and retain selection across restarts."""
        assert clean_settings.camera.device_index == 0
        assert clean_settings.camera.backend == "DSHOW"

        # Persist and reload
        user_file = tmp_path / "user_settings_test.json"
        clean_settings.save_user_settings(user_file)

        reloaded = Settings.load(user_path=user_file)
        assert reloaded.camera.device_index == 0
        assert reloaded.camera.backend == "DSHOW"
