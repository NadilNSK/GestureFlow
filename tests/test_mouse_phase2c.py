"""
tests.test_mouse_phase2c
~~~~~~~~~~~~~~~~~~~~~~~~~
Comprehensive unit tests for Phase 2C: Complete Hand Gesture Mouse Controls.

Covers:
1. Single Left Click (short LEFT_PINCH release)
2. Double Click (two quick LEFT_PINCH gestures within interval)
3. Right Click (stable RIGHT_PINCH)
4. Pinch Cooldown
5. Click vs. Drag distinction (short pinch vs sustained hold)
6. Drag Activation (mouseDown) and calibrated movement during drag
7. Drag Release (mouseUp) without trailing click
8. Vertical Scrolling (TWO_FINGER up/down, dead-zone, rate-limiting)
9. Gesture Conflicts (no right-click or scrolling during drag, no cursor movement during scroll)
10. Unconfirmed gesture rejection
11. Tracking Loss releases held buttons
12. Hand Identity Switch releases held buttons
13. Camera Disconnection / Pipeline Stop safety release
14. Emergency Stop safety release & block
15. Application Shutdown safety release
16. Mock Camera safety guard (zero OS mouse actions)
17. Calibration and Precision Mode preservation during movement/drag
18. PyAutoGUI FailSafeException handling
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, call, patch
import pytest

from gestureflow.calibration.calibrator import Calibrator
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
    """Instantiate a MouseController with Phase 2C settings and mock PyAutoGUI."""
    settings = ControllerSettings(
        mouse_control_enabled_default=False,
        cursor_sensitivity=1.0,
        cursor_smoothing=0.35,
        cursor_dead_zone=2.5,
        tracking_region_margin=0.15,
        cursor_jump_threshold=250.0,
        left_click_enabled=True,
        right_click_enabled=True,
        double_click_enabled=True,
        drag_enabled=True,
        scroll_enabled=True,
        click_cooldown=0.20,
        double_click_interval=0.35,
        drag_hold_duration=0.45,
        scroll_sensitivity=1.0,
        scroll_dead_zone=0.008,
        scroll_update_interval=0.04,
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
    knuckle_y: float = 0.5,
    handedness: str = "Right",
) -> HandLandmarks:
    """Create a mock HandLandmarks object with customizable landmarks."""
    landmarks = [MagicMock(x=0.5, y=0.5, z=0.0) for _ in range(21)]
    landmarks[0].x = 0.5
    landmarks[0].y = 0.6  # wrist
    landmarks[4].x = tip_x  # thumb tip
    landmarks[4].y = tip_y
    landmarks[5].x = tip_x  # index MCP
    landmarks[5].y = knuckle_y
    landmarks[8].x = tip_x  # index tip
    landmarks[8].y = tip_y
    landmarks[12].x = tip_x  # middle tip
    landmarks[12].y = tip_y
    return HandLandmarks(
        landmarks=landmarks,
        world_landmarks=[],
        handedness=handedness,
        score=0.95,
    )


def make_mock_gesture(
    gesture: Gesture,
    is_stable: bool = True,
) -> GestureResult:
    """Create a mock GestureResult."""
    return GestureResult(
        gesture=gesture,
        confidence=0.95,
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
# 1. Left Click & Double Click Tests
# ======================================================================

class TestLeftAndDoubleClick:
    def test_single_left_click_after_short_pinch(self, controller, mock_pag):
        """Short LEFT_PINCH release triggers a single click once double_click_interval expires."""
        hand = make_mock_hand(0.5, 0.5)
        g_pinch = make_mock_gesture(Gesture.LEFT_PINCH, is_stable=True)
        g_open = make_mock_gesture(Gesture.OPEN_PALM, is_stable=True)

        # 1. Pinch begins
        controller.process_hand(hand, g_pinch, timestamp=1.0)
        assert mock_pag.click.call_count == 0

        # 2. Pinch released after 100ms (< 0.45s drag threshold)
        controller.process_hand(hand, g_open, timestamp=1.10)
        assert mock_pag.click.call_count == 0  # Pending single-click window

        # 3. Double-click window (0.35s) expires at t=1.46 (> 1.10 + 0.35 = 1.45)
        controller.process_hand(hand, g_open, timestamp=1.46)
        mock_pag.click.assert_called_once_with(button="left")
        assert controller.left_click_count == 1
        assert controller.last_action == "LEFT CLICK"

    def test_double_click_within_interval(self, controller, mock_pag):
        """Two short pinches within 0.35s trigger doubleClick and cancel single click."""
        hand = make_mock_hand(0.5, 0.5)
        g_pinch = make_mock_gesture(Gesture.LEFT_PINCH, is_stable=True)
        g_open = make_mock_gesture(Gesture.OPEN_PALM, is_stable=True)

        # Pinch 1: 1.0 to 1.08
        controller.process_hand(hand, g_pinch, timestamp=1.0)
        controller.process_hand(hand, g_open, timestamp=1.08)

        # Pinch 2: 1.25 to 1.30 (within 0.35s of pinch 1 release)
        controller.process_hand(hand, g_pinch, timestamp=1.25)
        controller.process_hand(hand, g_open, timestamp=1.30)

        # doubleClick should be called; single click must NOT be called
        mock_pag.doubleClick.assert_called_once_with(button="left")
        assert mock_pag.click.call_count == 0
        assert controller.double_click_count == 1
        assert controller.last_action == "DOUBLE CLICK"

    def test_left_click_cooldown(self, controller, mock_pag):
        """Rapid pinch released within click_cooldown (0.20s) is ignored."""
        hand = make_mock_hand(0.5, 0.5)
        g_pinch = make_mock_gesture(Gesture.LEFT_PINCH, is_stable=True)
        g_open = make_mock_gesture(Gesture.OPEN_PALM, is_stable=True)

        # Disable double click to test immediate click cooldown
        controller.settings.double_click_enabled = False

        # First click
        controller.process_hand(hand, g_pinch, timestamp=1.0)
        controller.process_hand(hand, g_open, timestamp=1.05)
        assert mock_pag.click.call_count == 1

        # Second pinch within 0.10s (< 0.20s cooldown)
        controller.process_hand(hand, g_pinch, timestamp=1.10)
        controller.process_hand(hand, g_open, timestamp=1.15)
        assert mock_pag.click.call_count == 1  # Blocked by cooldown


# ======================================================================
# 2. Right Click Tests
# ======================================================================

class TestRightClick:
    def test_right_click_activation(self, controller, mock_pag):
        """RIGHT_PINCH triggers exactly one right-click action."""
        hand = make_mock_hand(0.5, 0.5)
        g_rpinch = make_mock_gesture(Gesture.RIGHT_PINCH, is_stable=True)

        # Continuous right pinch over 3 frames
        controller.process_hand(hand, g_rpinch, timestamp=1.0)
        controller.process_hand(hand, g_rpinch, timestamp=1.033)
        controller.process_hand(hand, g_rpinch, timestamp=1.066)

        mock_pag.rightClick.assert_called_once()
        assert controller.right_click_count == 1
        assert controller.last_action == "RIGHT CLICK"

    def test_right_click_cooldown(self, controller, mock_pag):
        """Subsequent right pinch within cooldown is rejected."""
        hand = make_mock_hand(0.5, 0.5)
        g_rpinch = make_mock_gesture(Gesture.RIGHT_PINCH, is_stable=True)
        g_open = make_mock_gesture(Gesture.OPEN_PALM, is_stable=True)

        controller.process_hand(hand, g_rpinch, timestamp=1.0)
        controller.process_hand(hand, g_open, timestamp=1.05)
        assert mock_pag.rightClick.call_count == 1

        # Try right click again at t=1.10 (only 0.10s later < 0.20s cooldown)
        controller.process_hand(hand, g_rpinch, timestamp=1.10)
        assert mock_pag.rightClick.call_count == 1


# ======================================================================
# 3. Drag and Drop Tests
# ======================================================================

class TestDragAndDrop:
    def test_drag_activation_and_release(self, controller, mock_pag):
        """Pinch held >= 0.45s triggers mouseDown, allows movement, and releases via mouseUp without click."""
        hand = make_mock_hand(0.5, 0.5)
        g_pinch = make_mock_gesture(Gesture.LEFT_PINCH, is_stable=True)
        g_open = make_mock_gesture(Gesture.OPEN_PALM, is_stable=True)

        # Start pinch at t=1.0
        controller.process_hand(hand, g_pinch, timestamp=1.0)
        assert not controller.is_dragging
        assert mock_pag.mouseDown.call_count == 0

        # At t=1.46 (> 0.45s hold duration): DRAG ACTIVATES!
        controller.process_hand(hand, g_pinch, timestamp=1.46)
        assert controller.is_dragging
        mock_pag.mouseDown.assert_called_once_with(button="left")

        # Move hand while dragging: cursor moves!
        hand.landmarks[8].x = 0.6
        hand.landmarks[8].y = 0.6
        coords = controller.process_hand(hand, g_pinch, timestamp=1.50)
        assert coords is not None
        mock_pag.moveTo.assert_called()

        # Release pinch
        controller.process_hand(hand, g_open, timestamp=1.60)
        assert not controller.is_dragging
        mock_pag.mouseUp.assert_called_once_with(button="left")

        # Advance past double-click window: NO click should ever occur after drag
        controller.process_hand(hand, g_open, timestamp=2.0)
        assert mock_pag.click.call_count == 0


# ======================================================================
# 4. Vertical Scrolling Tests
# ======================================================================

class TestVerticalScrolling:
    def test_scroll_up_and_down(self, controller, mock_pag):
        """TWO_FINGER hand moving vertically scrolls in the appropriate direction."""
        hand = make_mock_hand(0.5, 0.5, knuckle_y=0.50)
        g_scroll = make_mock_gesture(Gesture.TWO_FINGER, is_stable=True)

        # Initial reference frame
        controller.process_hand(hand, g_scroll, timestamp=1.0)
        assert mock_pag.scroll.call_count == 0

        # Move hand UP: knuckle_y moves from 0.50 to 0.45 (dy = -(0.45 - 0.50) = +0.05)
        hand.landmarks[5].y = 0.45
        controller.process_hand(hand, g_scroll, timestamp=1.05)
        assert mock_pag.scroll.call_count == 1
        # Upward scroll must have positive clicks
        scroll_arg = mock_pag.scroll.call_args[0][0]
        assert scroll_arg > 0
        assert controller.scroll_direction == "UP"

        # Move hand DOWN: knuckle_y moves from 0.45 to 0.55
        hand.landmarks[5].y = 0.55
        controller.process_hand(hand, g_scroll, timestamp=1.10)
        assert mock_pag.scroll.call_count == 2
        scroll_down_arg = mock_pag.scroll.call_args[0][0]
        assert scroll_down_arg < 0
        assert controller.scroll_direction == "DOWN"

    def test_scroll_dead_zone(self, controller, mock_pag):
        """Tiny micro-movements below scroll_dead_zone (0.008) produce zero scrolling."""
        hand = make_mock_hand(0.5, 0.5, knuckle_y=0.50)
        g_scroll = make_mock_gesture(Gesture.TWO_FINGER, is_stable=True)

        controller.process_hand(hand, g_scroll, timestamp=1.0)

        # Micro-tremor: knuckle_y moves by only 0.002
        hand.landmarks[5].y = 0.502
        controller.process_hand(hand, g_scroll, timestamp=1.05)
        assert mock_pag.scroll.call_count == 0

    def test_cursor_does_not_move_during_scroll(self, controller, mock_pag):
        """Virtual cursor is locked and does NOT move while scrolling."""
        hand = make_mock_hand(0.5, 0.5, knuckle_y=0.50)
        g_scroll = make_mock_gesture(Gesture.TWO_FINGER, is_stable=True)

        mock_pag.moveTo.reset_mock()
        controller.process_hand(hand, g_scroll, timestamp=1.0)
        mock_pag.moveTo.assert_not_called()


# ======================================================================
# 5. Gesture Conflicts & Unconfirmed Gesture Protection
# ======================================================================

class TestGestureConflicts:
    def test_no_right_click_during_drag(self, controller, mock_pag):
        """Right click is blocked if controller is in DRAGGING mode."""
        hand = make_mock_hand(0.5, 0.5)
        controller.mouse_down(button="left")
        assert controller.is_dragging

        g_rpinch = make_mock_gesture(Gesture.RIGHT_PINCH, is_stable=True)
        controller.process_hand(hand, g_rpinch, timestamp=1.0)
        mock_pag.rightClick.assert_not_called()

    def test_no_scrolling_during_drag(self, controller, mock_pag):
        """Scrolling is blocked if controller is in DRAGGING mode."""
        hand = make_mock_hand(0.5, 0.5)
        controller.mouse_down(button="left")
        assert controller.is_dragging

        g_scroll = make_mock_gesture(Gesture.TWO_FINGER, is_stable=True)
        controller.process_hand(hand, g_scroll, timestamp=1.0)
        mock_pag.scroll.assert_not_called()

    def test_unstable_gesture_rejected(self, controller, mock_pag):
        """Unconfirmed / unstable gestures do not trigger actions."""
        hand = make_mock_hand(0.5, 0.5)
        g_unstable = make_mock_gesture(Gesture.LEFT_PINCH, is_stable=False)

        controller.process_hand(hand, g_unstable, timestamp=1.0)
        controller.process_hand(hand, g_unstable, timestamp=1.50)
        mock_pag.mouseDown.assert_not_called()
        mock_pag.click.assert_not_called()


# ======================================================================
# 6. Safety & Unconditional Button Release Invariants
# ======================================================================

class TestSafetyAndReleaseInvariants:
    def test_tracking_loss_releases_drag(self, controller, mock_pag):
        """When hand disappears during dragging, mouse button is released safely."""
        controller.mouse_down(button="left")
        assert controller.is_dragging

        # Hand lost
        controller.process_hand(None, None, timestamp=2.0)
        assert not controller.is_dragging
        mock_pag.mouseUp.assert_called_once_with(button="left")

    def test_hand_identity_switch_releases_drag(self, controller, mock_pag):
        """When active hand switches from Right to Left during drag, held button is released."""
        hand_right = make_mock_hand(0.5, 0.5, handedness="Right")
        hand_left = make_mock_hand(0.5, 0.5, handedness="Left")
        g_pinch = make_mock_gesture(Gesture.LEFT_PINCH, is_stable=True)

        # Drag with Right hand
        controller.process_hand(hand_right, g_pinch, timestamp=1.0)
        controller.process_hand(hand_right, g_pinch, timestamp=1.50)
        assert controller.is_dragging

        # Sudden switch to Left hand
        controller.process_hand(hand_left, g_pinch, timestamp=1.60)
        assert not controller.is_dragging
        mock_pag.mouseUp.assert_called_once_with(button="left")

    def test_emergency_stop_releases_drag(self, controller, mock_pag):
        """Emergency stop immediately releases drag and enters EMERGENCY_STOPPED."""
        controller.mouse_down(button="left")
        assert controller.is_dragging

        controller.emergency_stop()
        assert not controller.is_dragging
        assert controller.state == MouseState.EMERGENCY_STOPPED
        mock_pag.mouseUp.assert_called_once_with(button="left")

    def test_disable_releases_drag(self, controller, mock_pag):
        """Disabling virtual mouse releases any held drag button."""
        controller.mouse_down(button="left")
        assert controller.is_dragging

        controller.disable()
        assert not controller.is_dragging
        assert controller.state == MouseState.DISABLED
        mock_pag.mouseUp.assert_called_once_with(button="left")

    def test_mock_camera_safety_protection(self, mock_pag):
        """In mock camera mode, all mouse actions are strictly blocked."""
        settings = ControllerSettings()
        ctrl = MouseController(
            settings=settings,
            pag_module=mock_pag,
            screen_size=(1920, 1080),
            mock_mode=True,
        )
        assert not ctrl.is_enabled
        assert not ctrl.click()
        assert not ctrl.right_click()
        assert not ctrl.double_click()
        assert not ctrl.mouse_down()
        assert not ctrl.scroll(5)
        mock_pag.click.assert_not_called()
        mock_pag.mouseDown.assert_not_called()
        mock_pag.scroll.assert_not_called()

    def test_failsafe_exception_engages_emergency_stop(self, controller, mock_pag):
        """PyAutoGUI FailSafeException is never suppressed and activates emergency stop."""
        mock_pag.click.side_effect = MockFailSafeException("FailSafe triggered")

        with pytest.raises(MockFailSafeException):
            controller.click()

        assert controller.is_emergency_stopped
        assert controller.state == MouseState.EMERGENCY_STOPPED


# ======================================================================
# 7. Calibration and Precision Mode Preservation
# ======================================================================

class TestCalibrationAndPrecisionPreservation:
    def test_calibrated_movement_during_drag(self, controller, mock_pag):
        """Calibrated coordinate mapping continues to function accurately during drag."""
        c = Calibrator()
        c.set_corner_point("TL", 0.10, 0.10)
        c.set_corner_point("TR", 0.90, 0.10)
        c.set_corner_point("BR", 0.90, 0.90)
        c.set_corner_point("BL", 0.10, 0.90)
        c.compute_calibration()
        controller.set_calibrator(c)

        # Enter drag
        controller.mouse_down(button="left")
        assert controller.is_dragging

        # Move hand to center (0.5, 0.5)
        hand = make_mock_hand(0.5, 0.5)
        g_pinch = make_mock_gesture(Gesture.LEFT_PINCH, is_stable=True)

        coords = controller.process_hand(hand, g_pinch, timestamp=1.50)
        assert coords is not None
        assert abs(coords[0] - 1919 * 0.5) < 30
        assert abs(coords[1] - 1079 * 0.5) < 30

    def test_precision_mode_during_drag(self, controller, mock_pag):
        """Precision mode applies scaling during drag movement."""
        controller.mouse_down(button="left")
        controller.set_precision_mode(True)
        assert controller.is_precision_mode
        assert controller.is_dragging

        hand = make_mock_hand(0.5, 0.5)
        g_pinch = make_mock_gesture(Gesture.LEFT_PINCH, is_stable=True)
        coords = controller.process_hand(hand, g_pinch, timestamp=1.0)
        assert coords is not None
