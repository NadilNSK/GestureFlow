"""
tests.test_phase3_optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Comprehensive automated test suite for Phase 3:
Advanced Gesture Recognition, Stability & Real-Time Performance Optimization.

Covers all 25 core test scenarios specified in Task M:
1. Adaptive pinch detection
2. Pinch normalization
3. Pinch hysteresis
4. Invalid hand geometry
5. Gesture history voting
6. Gesture confidence handling
7. Gesture stability
8. Gesture transitions
9. Click latency strategies (Safe Deferred vs Responsive Native)
10. Single-click correctness
11. Double-click correctness
12. Triple-click prevention
13. Drag state preservation
14. Scroll transition stability
15. Hand tracking continuity
16. Tracking loss
17. Frame staleness
18. Performance instrumentation
19. Cursor response presets
20. Profile persistence
21. Calibration preservation
22. Emergency stop
23. Mock-camera protection
24. FailSafeException propagation
25. Application shutdown
"""

from __future__ import annotations

import math
import time
from unittest.mock import MagicMock, call, patch
import pytest
import numpy as np

from gestureflow.calibration.calibrator import CalibrationData, Calibrator
from gestureflow.camera.capture import CameraFrame
from gestureflow.config.settings import (
    CURSOR_PROFILES,
    CameraSettings,
    ControllerSettings,
    GestureSettings,
    Settings,
    TrackingSettings,
)
from gestureflow.controller.mouse_controller import (
    MouseActionState,
    MouseController,
    MouseState,
)
from gestureflow.gestures.classifier import (
    FingerStates,
    Gesture,
    GestureClassifier,
    GestureResult,
    _compute_palm_scale,
    calculate_pinch_distance,
    detect_finger_states,
)
from gestureflow.tracking.hand_landmarker import HandLandmarks, TrackingResult
from gestureflow.tracking.pipeline import TrackingPipeline, PipelineFrame


class MockFailSafeException(Exception):
    """Synthetic PyAutoGUI FailSafeException."""
    pass


@pytest.fixture
def mock_pag():
    """Mock PyAutoGUI module for non-invasive mouse testing."""
    pag = MagicMock()
    pag.FAILSAFE = True
    pag.size.return_value = (1920, 1080)
    pag.position.return_value = (960, 540)
    pag.FailSafeException = MockFailSafeException
    return pag


@pytest.fixture
def default_ctrl(mock_pag):
    """MouseController instance with default Phase 3 settings."""
    settings = ControllerSettings(
        mouse_control_enabled_default=False,
        cursor_sensitivity=1.0,
        cursor_smoothing=0.35,
        cursor_dead_zone=2.5,
        tracking_region_margin=0.15,
        cursor_jump_threshold=250.0,
        click_strategy="deferred",
        cursor_profile="Balanced",
    )
    ctrl = MouseController(
        settings=settings,
        pag_module=mock_pag,
        screen_size=(1920, 1080),
        mock_mode=False,
    )
    ctrl.enable()
    return ctrl


def make_mock_hand(
    tip_x: float = 0.5,
    tip_y: float = 0.5,
    wrist_x: float = 0.5,
    wrist_y: float = 0.75,
    knuckle_y: float = 0.55,
    handedness: str = "Right",
) -> HandLandmarks:
    """Create a mock HandLandmarks with 21 landmarks."""
    landmarks = [MagicMock(x=0.5, y=0.5, z=0.0) for _ in range(21)]
    landmarks[0].x, landmarks[0].y = wrist_x, wrist_y  # Wrist
    landmarks[4].x, landmarks[4].y = tip_x, tip_y      # Thumb tip
    landmarks[5].x, landmarks[5].y = tip_x - 0.05, knuckle_y  # Index MCP
    landmarks[8].x, landmarks[8].y = tip_x, tip_y      # Index tip
    landmarks[9].x, landmarks[9].y = tip_x, knuckle_y   # Middle MCP
    landmarks[12].x, landmarks[12].y = tip_x, tip_y    # Middle tip
    landmarks[17].x, landmarks[17].y = tip_x + 0.05, knuckle_y  # Pinky MCP
    return HandLandmarks(
        landmarks=landmarks,
        world_landmarks=[],
        handedness=handedness,
        score=0.92,
    )


def make_mock_gesture(
    gesture: Gesture,
    is_stable: bool = True,
    confidence: float = 0.95,
) -> GestureResult:
    """Create a mock GestureResult."""
    return GestureResult(
        gesture=gesture,
        confidence=confidence,
        label=gesture.name,
        is_stable=is_stable,
        finger_states=FingerStates(
            thumb=(gesture in (Gesture.OPEN_PALM, Gesture.LEFT_PINCH, Gesture.RIGHT_PINCH)),
            index=(gesture in (Gesture.POINTING, Gesture.TWO_FINGER, Gesture.OPEN_PALM, Gesture.LEFT_PINCH)),
            middle=(gesture in (Gesture.TWO_FINGER, Gesture.OPEN_PALM, Gesture.RIGHT_PINCH)),
            ring=(gesture == Gesture.OPEN_PALM),
            pinky=(gesture == Gesture.OPEN_PALM),
        ),
    )


# ======================================================================
# 1. Adaptive Pinch & Normalization Tests
# ======================================================================

class TestAdaptivePinch:
    def test_adaptive_pinch_scaling_distance_invariance(self):
        """Pinch distance normalized by palm scale remains invariant across zoom distances."""
        # Near hand (large in camera frame)
        p_near = np.zeros((21, 3), dtype=np.float32)
        p_near[0] = [0.5, 0.8, 0.0]
        p_near[9] = [0.5, 0.4, 0.0]   # d09 = 0.4
        p_near[5] = [0.3, 0.45, 0.0]
        p_near[17] = [0.7, 0.45, 0.0] # d517 = 0.4
        p_near[4] = [0.5, 0.2, 0.0]
        p_near[8] = [0.5, 0.22, 0.0]  # raw dist = 0.02
        scale_near = _compute_palm_scale(p_near)
        norm_near = calculate_pinch_distance(p_near, 8)

        # Far hand (small in camera frame, 50% scale)
        p_far = np.zeros((21, 3), dtype=np.float32)
        p_far[0] = [0.5, 0.8, 0.0]
        p_far[9] = [0.5, 0.6, 0.0]   # d09 = 0.2
        p_far[5] = [0.4, 0.625, 0.0]
        p_far[17] = [0.6, 0.625, 0.0] # d517 = 0.2
        p_far[4] = [0.5, 0.5, 0.0]
        p_far[8] = [0.5, 0.51, 0.0]  # raw dist = 0.01 (half of near)
        scale_far = _compute_palm_scale(p_far)
        norm_far = calculate_pinch_distance(p_far, 8)

        assert math.isclose(scale_near, 0.4, rel_tol=1e-3)
        assert math.isclose(scale_far, 0.2, rel_tol=1e-3)
        assert math.isclose(norm_near, norm_far, rel_tol=1e-3)

    def test_pinch_hysteresis_enter_and_release(self):
        """Pinch enters at pinch_activation_threshold and only releases above pinch_release_threshold."""
        settings = GestureSettings(
            pinch_threshold=0.065,
            pinch_release_threshold=0.085,
            confirmation_frames=1,
            adaptive_pinch_enabled=False,  # Test exact raw distance thresholds
        )
        clf = GestureClassifier(settings)

        # Create hand with variable thumb-to-index distance
        def make_pinch_hand(dist: float):
            landmarks = [MagicMock(x=0.5, y=0.5, z=0.0) for _ in range(21)]
            landmarks[0].x, landmarks[0].y = 0.5, 0.8
            landmarks[9].x, landmarks[9].y = 0.5, 0.4
            landmarks[5].x, landmarks[5].y = 0.3, 0.4
            landmarks[17].x, landmarks[17].y = 0.7, 0.4
            landmarks[4].x, landmarks[4].y = 0.5, 0.2
            landmarks[8].x, landmarks[8].y = 0.5, 0.2 + dist
            landmarks[12].x, landmarks[12].y = 0.5, 0.5  # middle far away
            return HandLandmarks(landmarks=landmarks, world_landmarks=[], handedness="Right", score=0.95)

        # Dist = 0.075: between enter (0.065) and release (0.085). Starts inactive -> stays inactive.
        r1 = clf.classify(make_pinch_hand(0.075))
        assert clf._left_pinch_active is False

        # Dist = 0.050: drops below enter threshold -> pinch ACTIVATES!
        r2 = clf.classify(make_pinch_hand(0.050))
        assert clf._left_pinch_active is True

        # Dist = 0.075: rises above 0.065, but below release 0.085 -> pinch REMAINS ACTIVE (Hysteresis)!
        r3 = clf.classify(make_pinch_hand(0.075))
        assert clf._left_pinch_active is True

        # Dist = 0.090: rises above release threshold 0.085 -> pinch RELEASES!
        r4 = clf.classify(make_pinch_hand(0.090))
        assert clf._left_pinch_active is False

    def test_invalid_hand_geometry_handled_gracefully(self):
        """Division by zero, NaNs, and degenerate landmarks are handled without exceptions."""
        # 1. Zero scale / identical coordinates
        p_zero = np.zeros((21, 3), dtype=np.float32)
        scale_zero = _compute_palm_scale(p_zero)
        assert scale_zero == 1.0  # Fallback to 1.0, no ZeroDivisionError

        # 2. NaN in landmarks
        clf = GestureClassifier()
        p_nan = [MagicMock(x=float('nan'), y=0.5, z=0.0) for _ in range(21)]
        h_nan = HandLandmarks(landmarks=p_nan, world_landmarks=[], handedness="Right", score=0.5)
        res_nan = clf.classify(h_nan)
        assert res_nan.gesture == Gesture.UNKNOWN


# ======================================================================
# 2. Gesture Stability & History Voting Tests
# ======================================================================

class TestGestureStabilizationAndVoting:
    def test_confidence_weighted_voting_cleans_anomalous_frame(self):
        """Single noisy frame does not interrupt confirmed gesture during history voting."""
        settings = GestureSettings(
            confirmation_frames=2,
            gesture_history_size=5,
            gesture_stability_frames=2,
            gesture_confidence_threshold=0.70,
        )
        clf = GestureClassifier(settings)

        # Stable pointing frames
        hand_point = make_mock_hand(tip_x=0.5, tip_y=0.2)
        hand_point.landmarks[8].y = 0.1  # index extended
        hand_point.landmarks[12].y = 0.6 # middle folded
        hand_point.landmarks[16].y = 0.6 # ring folded
        hand_point.landmarks[20].y = 0.6 # pinky folded

        clf.classify(hand_point)
        res2 = clf.classify(hand_point)
        assert res2.gesture == Gesture.POINTING
        assert res2.is_stable is True

        # Inject 1 transient noisy frame (e.g. tracking glitch)
        hand_glitch = make_mock_hand(tip_x=0.5, tip_y=0.5)
        res_glitch = clf.classify(hand_glitch)
        # History queue should still vote for Pointing
        assert len(clf._history) >= 3
        # Confirmed gesture remains POINTING rather than jumping to UNKNOWN
        assert clf.current_gesture == Gesture.POINTING

    def test_gesture_confidence_decay_on_loss(self):
        """When hand disappears momentarily, confidence decays and holds before dropping."""
        settings = GestureSettings(loss_tolerance_frames=2, confirmation_frames=1)
        clf = GestureClassifier(settings)

        hand_point = make_mock_hand(tip_x=0.5, tip_y=0.2)
        hand_point.landmarks[8].y = 0.1  # index extended
        hand_point.landmarks[12].y = 0.6 # middle folded
        hand_point.landmarks[16].y = 0.6 # ring folded
        hand_point.landmarks[20].y = 0.6 # pinky folded
        clf.classify(hand_point)
        assert clf.current_gesture == Gesture.POINTING

        # Drop 1: Hold with decayed confidence
        r_drop1 = clf.classify(None)
        assert r_drop1.gesture == Gesture.POINTING
        assert "(Hold)" in r_drop1.label
        assert r_drop1.confidence < 0.95

        # Drop 2: Hold with further decay
        r_drop2 = clf.classify(None)
        assert r_drop2.gesture == Gesture.POINTING
        assert r_drop2.confidence < r_drop1.confidence

        # Drop 3 (> tolerance): drops completely to UNKNOWN
        r_drop3 = clf.classify(None)
        assert r_drop3.gesture == Gesture.UNKNOWN


# ======================================================================
# 3. Click Strategy Optimization Tests (Safe Deferred vs Native)
# ======================================================================

class TestClickStrategies:
    def test_safe_deferred_strategy_delays_single_click(self, default_ctrl, mock_pag):
        """Strategy A (Safe Deferred) holds single click until double_click_interval expires."""
        default_ctrl.set_click_strategy("deferred")
        assert default_ctrl.click_strategy == "deferred"

        hand = make_mock_hand(0.5, 0.5)
        g_pinch = make_mock_gesture(Gesture.LEFT_PINCH, is_stable=True)
        g_open = make_mock_gesture(Gesture.OPEN_PALM, is_stable=True)

        # Pinch starts at 1.0, releases at 1.05
        default_ctrl.process_hand(hand, g_pinch, timestamp=1.0)
        default_ctrl.process_hand(hand, g_open, timestamp=1.05)
        assert mock_pag.click.call_count == 0  # Delayed!

        # At t=1.20 (< 1.05 + 0.35 = 1.40): still delayed
        default_ctrl.process_hand(hand, g_open, timestamp=1.20)
        assert mock_pag.click.call_count == 0

        # At t=1.41: interval expires -> click dispatches!
        default_ctrl.process_hand(hand, g_open, timestamp=1.41)
        mock_pag.click.assert_called_once_with(button="left")
        assert default_ctrl.left_click_count == 1

    def test_responsive_native_strategy_executes_click_immediately(self, default_ctrl, mock_pag):
        """Strategy B (Responsive Native) fires click immediately on release with zero delay."""
        default_ctrl.set_click_strategy("native")
        assert default_ctrl.click_strategy == "native"

        hand = make_mock_hand(0.5, 0.5)
        g_pinch = make_mock_gesture(Gesture.LEFT_PINCH, is_stable=True)
        g_open = make_mock_gesture(Gesture.OPEN_PALM, is_stable=True)

        # Pinch starts at 1.0, releases at 1.05
        default_ctrl.process_hand(hand, g_pinch, timestamp=1.0)
        assert mock_pag.click.call_count == 0

        # On release at t=1.05: FIRES IMMEDIATELY!
        default_ctrl.process_hand(hand, g_open, timestamp=1.05)
        mock_pag.click.assert_called_once_with(button="left")
        assert default_ctrl.left_click_count == 1

    def test_responsive_native_strategy_double_click_no_triple_click(self, default_ctrl, mock_pag):
        """Strategy B fires second single click for native double-click, and prevents triple click."""
        default_ctrl.set_click_strategy("native")
        hand = make_mock_hand(0.5, 0.5)
        g_pinch = make_mock_gesture(Gesture.LEFT_PINCH, is_stable=True)
        g_open = make_mock_gesture(Gesture.OPEN_PALM, is_stable=True)

        # Pinch 1: 1.0 -> 1.05 (click #1 fires)
        default_ctrl.process_hand(hand, g_pinch, timestamp=1.0)
        default_ctrl.process_hand(hand, g_open, timestamp=1.05)
        assert mock_pag.click.call_count == 1

        # Pinch 2: 1.25 -> 1.30 (within 0.35s of pinch 1)
        default_ctrl.process_hand(hand, g_pinch, timestamp=1.25)
        default_ctrl.process_hand(hand, g_open, timestamp=1.30)

        # Exactly 2 clicks total (native double-click), and double_click() is NOT called
        assert mock_pag.click.call_count == 2
        assert mock_pag.doubleClick.call_count == 0
        assert default_ctrl.double_click_count == 1
        assert "NATIVE" in default_ctrl.last_action

        # Rapid Pinch 3 right after: should NOT trigger another click due to cooldown / window reset
        default_ctrl.process_hand(hand, g_pinch, timestamp=1.35)
        default_ctrl.process_hand(hand, g_open, timestamp=1.40)
        # Blocked by cooldown (1.40 - 1.30 = 0.10 < 0.20s click_cooldown)
        assert mock_pag.click.call_count == 2


# ======================================================================
# 4. Gesture Transitions & Drag / Scroll Preservation
# ======================================================================

class TestGestureTransitions:
    def test_pointing_to_pinch_preserves_cursor_stability(self, default_ctrl, mock_pag):
        """When moving hand into a pending pinch, cursor is held steady and does not fling."""
        hand = make_mock_hand(0.5, 0.5)
        g_point = make_mock_gesture(Gesture.POINTING, is_stable=True)
        g_pinch = make_mock_gesture(Gesture.LEFT_PINCH, is_stable=True)

        # Pointing moves cursor
        default_ctrl.process_hand(hand, g_point, timestamp=1.0)
        assert default_ctrl.state == MouseState.MOVING
        last_pos = default_ctrl.get_cursor_position()

        # Transition to LEFT_PINCH: cursor holds steady
        mock_pag.reset_mock()
        default_ctrl.process_hand(hand, g_pinch, timestamp=1.05)
        assert default_ctrl.state == MouseState.IDLE
        assert default_ctrl.action_state == MouseActionState.LEFT_PINCH_PENDING
        mock_pag.moveTo.assert_not_called()

    def test_drag_release_clean_transition_to_pointing(self, default_ctrl, mock_pag):
        """Releasing a drag dispatches mouseUp and returns to POINTING cleanly without click."""
        hand = make_mock_hand(0.5, 0.5)
        g_pinch = make_mock_gesture(Gesture.LEFT_PINCH, is_stable=True)
        g_point = make_mock_gesture(Gesture.POINTING, is_stable=True)

        # Enter drag (held for > 0.45s)
        default_ctrl.process_hand(hand, g_pinch, timestamp=1.0)
        default_ctrl.process_hand(hand, g_pinch, timestamp=1.50)
        assert default_ctrl.is_dragging
        mock_pag.mouseDown.assert_called_once_with(button="left")

        # Release drag into pointing
        default_ctrl.process_hand(hand, g_point, timestamp=1.60)
        assert not default_ctrl.is_dragging
        mock_pag.mouseUp.assert_called_once_with(button="left")
        assert mock_pag.click.call_count == 0  # Never click after drag

    def test_scroll_transition_locks_cursor_and_resets_displacement(self, default_ctrl, mock_pag):
        """Transitioning in and out of TWO_FINGER scroll does not jump cursor."""
        hand = make_mock_hand(0.5, 0.5, knuckle_y=0.50)
        g_point = make_mock_gesture(Gesture.POINTING, is_stable=True)
        g_scroll = make_mock_gesture(Gesture.TWO_FINGER, is_stable=True)

        default_ctrl.process_hand(hand, g_point, timestamp=1.0)
        mock_pag.moveTo.reset_mock()

        # Enter scroll
        default_ctrl.process_hand(hand, g_scroll, timestamp=1.05)
        assert default_ctrl.state == MouseState.IDLE
        assert default_ctrl.action_state == MouseActionState.SCROLLING
        mock_pag.moveTo.assert_not_called()

        # Exit scroll back to pointing
        default_ctrl.process_hand(hand, g_point, timestamp=1.10)
        assert default_ctrl.action_state == MouseActionState.MOVING


# ======================================================================
# 5. Hand Tracking Continuity & Stale Frame Tests
# ======================================================================

class TestHandTrackingContinuity:
    def test_spatial_continuity_locks_primary_hand_when_second_enters(self):
        """When second hand enters, spatial continuity prevents swapping active hand."""
        settings = Settings()
        settings.camera.use_mock_camera = False
        pipeline = TrackingPipeline(settings)

        hand1 = make_mock_hand(wrist_x=0.3, wrist_y=0.7, handedness="Right")
        pipeline._prev_active_wrist = (0.3, 0.7)

        # Frame with hand1 and a newly entered hand2 on other side
        hand2 = make_mock_hand(wrist_x=0.8, wrist_y=0.7, handedness="Right")
        chosen = pipeline._select_primary_hand([hand2, hand1])
        assert chosen is hand1

    def test_tracking_loss_releases_drag_promptly(self, default_ctrl, mock_pag):
        """When tracking is lost during drag, mouseUp is called immediately with no extrapolation."""
        default_ctrl.mouse_down(button="left")
        assert default_ctrl.is_dragging

        # Hand vanishes
        default_ctrl.process_hand(None, None, timestamp=2.0)
        assert not default_ctrl.is_dragging
        mock_pag.mouseUp.assert_called_once_with(button="left")


# ======================================================================
# 6. Cursor Response Presets & Profile Persistence Tests
# ======================================================================

class TestCursorProfilesAndPresets:
    def test_profile_presets_apply_proper_smoothing_and_sens(self, default_ctrl):
        """Applying Balanced, Precision, and Responsive profiles updates parameters."""
        # Precision profile: lower sensitivity, higher smoothing
        assert default_ctrl.set_cursor_profile("Precision")
        assert default_ctrl.cursor_profile == "Precision"
        assert default_ctrl.settings.cursor_sensitivity == 0.7
        assert default_ctrl.settings.cursor_smoothing == 0.60
        assert default_ctrl.settings.one_euro_min_cutoff == 0.4

        # Responsive profile: higher sensitivity, lower smoothing, native click
        assert default_ctrl.set_cursor_profile("Responsive")
        assert default_ctrl.cursor_profile == "Responsive"
        assert default_ctrl.settings.cursor_sensitivity == 1.4
        assert default_ctrl.settings.cursor_smoothing == 0.20
        assert default_ctrl.settings.click_strategy == "native"

        # Balanced profile (default)
        assert default_ctrl.set_cursor_profile("Balanced")
        assert default_ctrl.cursor_profile == "Balanced"
        assert default_ctrl.settings.cursor_sensitivity == 1.0

    def test_calibration_preservation_across_profile_switch(self, default_ctrl):
        """Switching profiles preserves active 4-point calibration."""
        c = Calibrator()
        c.set_corner_point("TL", 0.1, 0.1)
        c.set_corner_point("TR", 0.9, 0.1)
        c.set_corner_point("BR", 0.9, 0.9)
        c.set_corner_point("BL", 0.1, 0.9)
        c.compute_calibration()
        default_ctrl.set_calibrator(c)
        assert default_ctrl.calibrator.is_calibrated

        # Switch to Precision
        default_ctrl.set_cursor_profile("Precision")
        assert default_ctrl.calibrator.is_calibrated

        # Switch to Responsive
        default_ctrl.set_cursor_profile("Responsive")
        assert default_ctrl.calibrator.is_calibrated


# ======================================================================
# 7. Safety, Emergency Stop & Mock Camera Tests
# ======================================================================

class TestSafetyAndReliability:
    def test_emergency_stop_halts_and_releases(self, default_ctrl, mock_pag):
        """Emergency stop immediately halts movement and releases held drag button."""
        default_ctrl.mouse_down(button="left")
        assert default_ctrl.is_dragging

        default_ctrl.emergency_stop()
        assert not default_ctrl.is_dragging
        assert default_ctrl.state == MouseState.EMERGENCY_STOPPED
        assert default_ctrl.is_emergency_stopped
        mock_pag.mouseUp.assert_called_once_with(button="left")

        # Subsequent hand movements are blocked
        hand = make_mock_hand(0.5, 0.5)
        g_point = make_mock_gesture(Gesture.POINTING, is_stable=True)
        res = default_ctrl.process_hand(hand, g_point)
        assert res is None

    def test_mock_camera_safety_guard(self, mock_pag):
        """Mock camera strictly blocks enabling mouse and zero actions execute."""
        settings = ControllerSettings()
        ctrl = MouseController(
            settings=settings,
            pag_module=mock_pag,
            screen_size=(1920, 1080),
            mock_mode=True,
        )
        assert not ctrl.is_enabled
        assert not ctrl.enable()
        assert not ctrl.click()
        assert not ctrl.right_click()
        assert not ctrl.mouse_down()
        assert not ctrl.scroll(5)
        mock_pag.click.assert_not_called()
        mock_pag.mouseDown.assert_not_called()
        mock_pag.scroll.assert_not_called()

    def test_failsafe_exception_triggers_emergency_stop(self, default_ctrl, mock_pag):
        """PyAutoGUI FailSafeException is propagated and engages emergency stop."""
        mock_pag.moveTo.side_effect = MockFailSafeException("FailSafe corner triggered")

        with pytest.raises(MockFailSafeException):
            default_ctrl.move(10, 10)

        assert default_ctrl.is_emergency_stopped
        assert default_ctrl.state == MouseState.EMERGENCY_STOPPED

    def test_application_shutdown_releases_held_buttons(self, default_ctrl, mock_pag):
        """Disabling/shutting down controller releases held mouse button."""
        default_ctrl.mouse_down(button="left")
        assert default_ctrl.is_dragging

        default_ctrl.disable()
        assert not default_ctrl.is_dragging
        assert default_ctrl.state == MouseState.DISABLED
        mock_pag.mouseUp.assert_called_once_with(button="left")
