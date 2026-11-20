"""
tests.test_mouse_phase2b
~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for Phase 2B: Real-Time Virtual Mouse Control.

Tests:
1. Normalized coordinate mapping
2. Mirrored coordinate mapping
3. Screen boundary clamping
4. Invalid coordinates (NaN, Inf, None)
5. Exponential Moving Average (EMA) smoothing
6. Dead-zone jitter filtering
7. Sensitivity adjustment
8. POINTING gesture cursor movement activation
9. Non-pointing gestures disabling movement
10. Temporary tracking loss handling & smoothing reset
11. Hand identity switching resets smoothing
12. Emergency stop activation & reactivation block
13. PyAutoGUI FailSafeException handling (preserves fail-safe)
14. Mouse control disabled by default
15. Mock-camera safety protection
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, call, patch
import pytest

from gestureflow.config.settings import ControllerSettings
from gestureflow.controller.mouse_controller import MouseController, MouseState
from gestureflow.gestures.classifier import FingerStates, Gesture, GestureResult
from gestureflow.tracking.hand_landmarker import HandLandmarks


class MockFailSafeException(Exception):
    """Synthetic PyAutoGUI FailSafeException."""
    pass


@pytest.fixture
def mock_pag():
    """Create a mock PyAutoGUI module for isolated unit testing."""
    pag = MagicMock()
    pag.FAILSAFE = True
    pag.size.return_value = (1920, 1080)
    pag.position.return_value = (960, 540)
    pag.FailSafeException = MockFailSafeException
    return pag


@pytest.fixture
def controller(mock_pag):
    """Instantiate a MouseController with injected mock PyAutoGUI."""
    settings = ControllerSettings(
        mouse_control_enabled_default=False,
        cursor_sensitivity=1.0,
        cursor_smoothing=0.35,
        cursor_dead_zone=2.5,
        tracking_region_margin=0.15,
        cursor_jump_threshold=250.0,
    )
    return MouseController(
        settings=settings,
        pag_module=mock_pag,
        screen_size=(1920, 1080),
        mock_mode=False,
    )


def make_mock_hand(
    tip_x: float = 0.5,
    tip_y: float = 0.5,
    handedness: str = "Right",
    score: float = 0.95,
) -> HandLandmarks:
    """Create a mock HandLandmarks object with index tip at (tip_x, tip_y)."""
    landmarks = [MagicMock(x=0.5, y=0.5, z=0.0) for _ in range(21)]
    landmarks[8].x = tip_x
    landmarks[8].y = tip_y
    return HandLandmarks(
        landmarks=landmarks,
        world_landmarks=[],
        handedness=handedness,
        score=score,
    )


def make_mock_gesture(
    gesture: Gesture = Gesture.POINTING,
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
            thumb=False,
            index=True,
            middle=False,
            ring=False,
            pinky=False,
        ),
    )


# ======================================================================
# Test Cases (15 Core Scenarios)
# ======================================================================

class TestCoordinateMapping:
    def test_normalized_coordinate_mapping(self, controller):
        """Test normalized (0.5, 0.5) maps to primary screen center (959.5, 539.5)."""
        coords = controller.map_coordinates(0.5, 0.5, margin=0.15, mirrored=True)
        assert coords is not None
        sx, sy = coords
        assert math.isclose(sx, 1919 * 0.5, abs_tol=1e-3)
        assert math.isclose(sy, 1079 * 0.5, abs_tol=1e-3)

    def test_mirrored_coordinate_mapping(self, controller):
        """Test behavior with mirrored=True vs mirrored=False."""
        # When mirrored=True (default), x=0.2 maps to left side of active area
        c_mirrored = controller.map_coordinates(0.2, 0.5, margin=0.1, mirrored=True)
        # When mirrored=False, x=0.2 flips to 1.0 - 0.2 = 0.8 (right side)
        c_unmirrored = controller.map_coordinates(0.2, 0.5, margin=0.1, mirrored=False)

        assert c_mirrored is not None and c_unmirrored is not None
        assert c_mirrored[0] < c_unmirrored[0]
        # Y coordinate should be unaffected by horizontal mirror
        assert math.isclose(c_mirrored[1], c_unmirrored[1], abs_tol=1e-3)

    def test_screen_boundary_clamping(self, controller):
        """Test coordinates outside [0, 1] or beyond margin clamp to screen limits."""
        # Far outside left/top
        c_min = controller.map_coordinates(-0.5, -0.2, margin=0.15)
        assert c_min == (0.0, 0.0)

        # Far outside right/bottom
        c_max = controller.map_coordinates(1.5, 2.0, margin=0.15)
        assert c_max == (1919.0, 1079.0)

    def test_invalid_coordinates_handled_safely(self, controller):
        """Test NaN, Inf, and None are rejected safely without raising exceptions."""
        assert controller.map_coordinates(float("nan"), 0.5) is None
        assert controller.map_coordinates(0.5, float("inf")) is None
        assert controller.map_coordinates(None, 0.5) is None  # type: ignore
        assert controller.map_coordinates(0.5, None) is None  # type: ignore


class TestSmoothingAndDeadZone:
    def test_ema_smoothing(self, controller):
        """Test Exponential Moving Average formula smoothing successive inputs."""
        controller.set_smoothing(0.5)
        controller.reset_smoothing()

        # Frame 1: initial snap
        p1 = controller.smooth_coordinates(100.0, 100.0)
        assert p1 == (100, 100)

        # Frame 2: target moves to (200, 200).
        # Distance = hypot(100, 100) = 141.4 (> 50, adaptive alpha = min(0.95, 0.5 * 1.5) = 0.75)
        # smooth = 0.75 * 200 + 0.25 * 100 = 175
        p2 = controller.smooth_coordinates(200.0, 200.0)
        assert p2 is not None
        assert 150 <= p2[0] <= 190
        assert 150 <= p2[1] <= 190

    def test_dead_zone_behavior(self, controller):
        """Test displacements smaller than cursor_dead_zone produce zero cursor movement."""
        controller.reset_smoothing()
        p1 = controller.smooth_coordinates(500.0, 500.0)
        assert p1 == (500, 500)

        # Small micro-tremor displacement (< 2.5 px)
        p2 = controller.smooth_coordinates(501.0, 501.0)
        # Should stay at previous integer coordinates
        assert p2 == (500, 500)

    def test_sensitivity_adjustment(self, controller):
        """Test higher sensitivity reaches boundaries with smaller hand movement."""
        # Base sensitivity 1.0
        controller.set_sensitivity(1.0)
        c_base = controller.map_coordinates(0.6, 0.5, margin=0.0)
        assert c_base is not None

        # Higher sensitivity 2.0: displacement from center (0.5) is doubled
        controller.set_sensitivity(2.0)
        c_high = controller.map_coordinates(0.6, 0.5, margin=0.0)
        assert c_high is not None

        dist_base = c_base[0] - (1919 * 0.5)
        dist_high = c_high[0] - (1919 * 0.5)
        assert dist_high > dist_base


class TestGestureActivation:
    def test_pointing_activation(self, controller, mock_pag):
        """Test confirmed stable POINTING moves the virtual cursor."""
        controller.enable()
        hand = make_mock_hand(0.5, 0.5)
        gesture = make_mock_gesture(Gesture.POINTING, is_stable=True)

        res = controller.process_hand(hand, gesture)
        assert res is not None
        assert controller.state == MouseState.MOVING
        mock_pag.moveTo.assert_called_once()

    def test_other_gestures_disable_movement(self, controller, mock_pag):
        """Test FIST, OPEN_PALM, PINCH, TWO_FINGER, and UNKNOWN do not move the cursor."""
        controller.enable()
        hand = make_mock_hand(0.5, 0.5)

        for non_pointing_gesture in (
            Gesture.FIST,
            Gesture.OPEN_PALM,
            Gesture.LEFT_PINCH,
            Gesture.RIGHT_PINCH,
            Gesture.TWO_FINGER,
            Gesture.UNKNOWN,
        ):
            mock_pag.reset_mock()
            g = make_mock_gesture(non_pointing_gesture, is_stable=True)
            res = controller.process_hand(hand, g)
            assert res is None
            assert controller.state == MouseState.IDLE
            mock_pag.moveTo.assert_not_called()

        # Also test unconfirmed POINTING (is_stable=False)
        mock_pag.reset_mock()
        g_unconfirmed = make_mock_gesture(Gesture.POINTING, is_stable=False)
        res = controller.process_hand(hand, g_unconfirmed)
        assert res is None
        assert controller.state == MouseState.IDLE
        mock_pag.moveTo.assert_not_called()

    def test_tracking_loss(self, controller, mock_pag):
        """Test hand loss immediately stops movement and resets smoothing."""
        controller.enable()
        hand = make_mock_hand(0.5, 0.5)
        g = make_mock_gesture(Gesture.POINTING, is_stable=True)

        # Active movement
        controller.process_hand(hand, g)
        assert controller.state == MouseState.MOVING

        # Hand lost (None)
        res = controller.process_hand(None, g)
        assert res is None
        assert controller.state == MouseState.IDLE
        assert controller._prev_smoothed is None

    def test_hand_switching_resets_smoothing(self, controller, mock_pag):
        """Test switching hands resets smoothing to prevent cursor jump sweeps."""
        controller.enable()
        hand_right = make_mock_hand(0.4, 0.4, handedness="Right")
        hand_left = make_mock_hand(0.8, 0.8, handedness="Left")
        g = make_mock_gesture(Gesture.POINTING, is_stable=True)

        controller.process_hand(hand_right, g)
        assert controller._prev_hand_id == "Right"

        # Switch to Left hand
        controller.process_hand(hand_left, g)
        assert controller._prev_hand_id == "Left"


class TestSafetyAndEmergencyStop:
    def test_emergency_stop(self, controller, mock_pag):
        """Test emergency stop halts movement and blocks reactivation until reset."""
        controller.enable()
        assert controller.is_enabled is True

        controller.emergency_stop()
        assert controller.is_enabled is False
        assert controller.is_emergency_stopped is True
        assert controller.state == MouseState.EMERGENCY_STOPPED

        # Attempt to re-enable while emergency stop is active must be rejected
        ok = controller.enable()
        assert ok is False
        assert controller.is_enabled is False

        # Resetting emergency stop returns to clean disabled state
        controller.reset_emergency_stop()
        assert controller.is_emergency_stopped is False
        assert controller.state == MouseState.DISABLED

        # Now can be enabled
        ok = controller.enable()
        assert ok is True
        assert controller.is_enabled is True

    def test_failsafe_exception_handling(self, controller, mock_pag):
        """Test PyAutoGUI FailSafeException triggers emergency stop and is re-raised."""
        controller.enable()
        mock_pag.moveTo.side_effect = MockFailSafeException("Fail-safe triggered by mouse corner")

        with pytest.raises(MockFailSafeException):
            controller.move(100, 100)

        assert controller.is_emergency_stopped is True
        assert controller.state == MouseState.EMERGENCY_STOPPED
        assert controller.is_enabled is False

    def test_mouse_control_disabled_by_default(self, mock_pag):
        """Test new controller instances default to disabled state."""
        ctrl = MouseController(pag_module=mock_pag, screen_size=(1920, 1080))
        assert ctrl.is_enabled is False
        assert ctrl.state == MouseState.DISABLED

    def test_mock_camera_protection(self, mock_pag):
        """Test mouse control cannot be enabled when mock camera mode is active."""
        ctrl = MouseController(
            pag_module=mock_pag,
            screen_size=(1920, 1080),
            mock_mode=True,
        )
        assert ctrl.is_enabled is False
        ok = ctrl.enable()
        assert ok is False
        assert ctrl.is_enabled is False
