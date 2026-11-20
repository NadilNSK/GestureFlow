"""
gestureflow.controller.mouse_controller
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Production-quality virtual mouse controller using PyAutoGUI.
Dispatches smooth, fail-safe virtual cursor movement derived from
MediaPipe Hand Landmarker tracking.

Multi-Monitor Support Note:
---------------------------
Windows desktop automation via PyAutoGUI operates on the primary display
coordinate space [0, screen_width - 1] x [0, screen_height - 1], where
(0, 0) is the top-left corner of the primary screen. Virtual cursor
movement is strictly clamped to the primary monitor boundaries to ensure
deterministic tracking behavior. Secondary/virtual monitors extending
outside this primary coordinate rectangle are reserved for future phases.
"""

from __future__ import annotations

import enum
import logging
import math
import time
from typing import Any, Optional

import numpy as np

from gestureflow.config.settings import ControllerSettings
from gestureflow.utils.helpers import clamp

log = logging.getLogger(__name__)


class MouseState(str, enum.Enum):
    """Current operational state of the virtual mouse controller."""

    DISABLED = "DISABLED"
    IDLE = "IDLE"                         # Enabled, but hand missing or gesture not moving
    MOVING = "MOVING"                     # Enabled, active hand pointing, cursor moving
    EMERGENCY_STOPPED = "EMERGENCY_STOPPED"


class MouseController:
    """
    Virtual mouse controller that maps hand landmark positions to Windows
    cursor coordinates with speed-adaptive EMA smoothing, dead-zone filtering,
    active tracking region normalization, and non-negotiable PyAutoGUI fail-safe
    enforcement.

    Parameters
    ----------
    settings:
        ControllerSettings containing sensitivity, smoothing, dead zone, etc.
    pag_module:
        Optional PyAutoGUI-compatible module (injected for testing/mocking).
    screen_size:
        Optional (width, height) screen dimensions (injected for testing).
    mock_mode:
        True if camera is running in mock/synthetic mode. When True, mouse
        control cannot be enabled as a safety guard.
    """

    def __init__(
        self,
        settings: ControllerSettings | None = None,
        pag_module: Any = None,
        screen_size: tuple[int, int] | None = None,
        mock_mode: bool = False,
    ) -> None:
        self._settings = settings or ControllerSettings()
        self._mock_mode = mock_mode

        # PyAutoGUI initialization and fail-safe verification
        if pag_module is not None:
            self._pag = pag_module
        else:
            try:
                import pyautogui

                # PyAutoGUI fail-safe must ALWAYS be enabled.
                assert pyautogui.FAILSAFE, (
                    "PyAutoGUI fail-safe is OFF — refusing to start controller."
                )
                self._pag = pyautogui
            except (ImportError, AssertionError) as exc:
                self._pag = None
                log.warning("PyAutoGUI unavailable or invalid: %s", exc)

        # Screen dimensions resolution
        if screen_size is not None:
            self._screen_width, self._screen_height = int(screen_size[0]), int(screen_size[1])
        elif self._pag is not None and hasattr(self._pag, "size"):
            try:
                sz = self._pag.size()
                self._screen_width, self._screen_height = int(sz[0]), int(sz[1])
            except Exception as exc:
                log.warning("Could not query screen size via PyAutoGUI: %s. Using 1920x1080.", exc)
                self._screen_width, self._screen_height = (1920, 1080)
        else:
            self._screen_width, self._screen_height = (1920, 1080)

        # State tracking
        self._enabled = False
        self._emergency_stopped = False
        self._state = MouseState.DISABLED

        # Smoothing & tracking history
        self._prev_smoothed: Optional[tuple[float, float]] = None
        self._prev_hand_id: Optional[str] = None
        self._last_moved_pos: Optional[tuple[int, int]] = None
        self._last_move_time: float = 0.0

        if self._settings.mouse_control_enabled_default and not self._mock_mode:
            self.enable()

    # ------------------------------------------------------------------
    # Properties & Status
    # ------------------------------------------------------------------

    @property
    def is_enabled(self) -> bool:
        """True if virtual mouse control is currently enabled."""
        return self._enabled and not self._emergency_stopped

    @property
    def is_emergency_stopped(self) -> bool:
        """True if emergency stop has been activated."""
        return self._emergency_stopped

    @property
    def state(self) -> MouseState:
        """Current operational state."""
        return self._state

    @property
    def screen_size(self) -> tuple[int, int]:
        """Detected primary screen dimensions (width, height)."""
        return (self._screen_width, self._screen_height)

    @property
    def settings(self) -> ControllerSettings:
        """Active controller configuration."""
        return self._settings

    # ------------------------------------------------------------------
    # Lifecycle & Control
    # ------------------------------------------------------------------

    def enable(self) -> bool:
        """
        Enable gesture-based mouse control.

        Returns
        -------
        bool
            True if enabled successfully, False if blocked (e.g. mock mode
            or active emergency stop).
        """
        if self._mock_mode:
            log.warning("Safety protection: Cannot enable mouse control in mock camera mode.")
            return False

        if self._emergency_stopped:
            log.warning("Safety protection: Emergency stop is active. Reset emergency stop first.")
            return False

        self._enabled = True
        self._state = MouseState.IDLE
        self.reset_smoothing()
        log.info("MouseController: Virtual mouse control ENABLED.")
        return True

    def disable(self) -> None:
        """Disable virtual mouse control cleanly."""
        self._enabled = False
        if not self._emergency_stopped:
            self._state = MouseState.DISABLED
        self.reset_smoothing()
        log.info("MouseController: Virtual mouse control DISABLED.")

    def emergency_stop(self) -> None:
        """
        Immediately stop virtual mouse automation and block reactivation
        until reset_emergency_stop() is explicitly called.
        """
        self._emergency_stopped = True
        self._enabled = False
        self._state = MouseState.EMERGENCY_STOPPED
        self.reset_smoothing()
        log.warning("MouseController: EMERGENCY STOP ACTIVATED. All automation halted.")

    def reset_emergency_stop(self) -> None:
        """Clear emergency stop condition back to clean disabled state."""
        self._emergency_stopped = False
        self._state = MouseState.DISABLED
        self.reset_smoothing()
        log.info("MouseController: Emergency stop cleared. Controller is now DISABLED (ready to enable).")

    def reset_smoothing(self) -> None:
        """Reset historical EMA coordinates to prevent cursor jumps."""
        self._prev_smoothed = None
        self._last_moved_pos = None

    def set_mock_mode(self, enabled: bool) -> None:
        """Update mock camera status. Disables mouse if mock mode is turned on."""
        self._mock_mode = enabled
        if enabled and self._enabled:
            self.disable()

    def set_sensitivity(self, val: float) -> None:
        """Update cursor sensitivity multiplier (clamped to [0.1, 5.0])."""
        self._settings.cursor_sensitivity = float(clamp(val, 0.1, 5.0))

    def set_smoothing(self, val: float) -> None:
        """Update EMA base smoothing factor (clamped to [0.01, 1.0])."""
        self._settings.cursor_smoothing = float(clamp(val, 0.01, 1.0))

    def get_cursor_position(self) -> tuple[int, int]:
        """Read the current cursor position via PyAutoGUI or last known pos."""
        if self._pag is not None and hasattr(self._pag, "position"):
            try:
                pos = self._pag.position()
                return (int(pos[0]), int(pos[1]))
            except Exception:
                pass
        if self._prev_smoothed is not None:
            return (int(round(self._prev_smoothed[0])), int(round(self._prev_smoothed[1])))
        return (0, 0)

    # ------------------------------------------------------------------
    # Coordinate Mapping & Active Tracking Region
    # ------------------------------------------------------------------

    def map_coordinates(
        self,
        norm_x: float,
        norm_y: float,
        margin: float | None = None,
        mirrored: bool = True,
    ) -> tuple[float, float] | None:
        """
        Map normalized camera coordinates [0.0, 1.0] to target screen coordinates
        [0, screen_width - 1] and [0, screen_height - 1] using active region margins.

        Parameters
        ----------
        norm_x, norm_y:
            Normalized coordinates from MediaPipe (e.g. index fingertip landmark 8).
        margin:
            Active tracking region margin ratio (e.g. 0.15). If None, uses settings.
        mirrored:
            True if camera frame was already horizontally flipped (GestureFlow default).
            If False, norm_x is mirrored to provide natural interaction.

        Returns
        -------
        tuple[float, float] | None
            (screen_x, screen_y) floating-point target coordinates, or None if invalid.
        """
        if norm_x is None or norm_y is None:
            return None

        try:
            x_val = float(norm_x)
            y_val = float(norm_y)
        except (TypeError, ValueError):
            return None

        # Check for NaN / Inf
        if math.isnan(x_val) or math.isinf(x_val) or math.isnan(y_val) or math.isinf(y_val):
            log.warning("Invalid landmark coordinates: NaN/Inf detected.")
            return None

        # Clamp normalized inputs to [0.0, 1.0]
        x_val = clamp(x_val, 0.0, 1.0)
        y_val = clamp(y_val, 0.0, 1.0)

        # If camera stream was not mirrored, flip X to match user hand orientation
        if not mirrored:
            x_val = 1.0 - x_val

        # Active tracking region mapping
        m = margin if margin is not None else self._settings.tracking_region_margin
        m = clamp(float(m), 0.0, 0.45)
        span = 1.0 - 2.0 * m
        if span <= 0.0:
            span = 1.0

        u = clamp((x_val - m) / span, 0.0, 1.0)
        v = clamp((y_val - m) / span, 0.0, 1.0)

        # Sensitivity centered scaling around 0.5
        sens = self._settings.cursor_sensitivity
        if sens != 1.0:
            u = clamp((u - 0.5) * sens + 0.5, 0.0, 1.0)
            v = clamp((v - 0.5) * sens + 0.5, 0.0, 1.0)

        screen_x = u * float(self._screen_width - 1)
        screen_y = v * float(self._screen_height - 1)
        return (screen_x, screen_y)

    # ------------------------------------------------------------------
    # Smoothing & Dead-Zone Filtering
    # ------------------------------------------------------------------

    def smooth_coordinates(
        self,
        target_x: float,
        target_y: float,
    ) -> tuple[int, int] | None:
        """
        Apply Exponential Moving Average (EMA) with speed-adaptive alpha
        and dead-zone filtering to target screen coordinates.

        Parameters
        ----------
        target_x, target_y:
            Target screen coordinates in pixels.

        Returns
        -------
        tuple[int, int] | None
            (smoothed_x, smoothed_y) integer coordinates clamped to screen.
        """
        if self._prev_smoothed is None:
            # First frame after activation or tracking reset: snap directly
            self._prev_smoothed = (target_x, target_y)
            sx = int(round(clamp(target_x, 0.0, float(self._screen_width - 1))))
            sy = int(round(clamp(target_y, 0.0, float(self._screen_height - 1))))
            return (sx, sy)

        prev_x, prev_y = self._prev_smoothed
        dx = target_x - prev_x
        dy = target_y - prev_y
        dist = math.hypot(dx, dy)

        # 1. Large jump detection: if tracking skipped across the screen,
        # snap to target directly rather than slowly sweeping intermediate pixels
        if dist > self._settings.cursor_jump_threshold:
            self._prev_smoothed = (target_x, target_y)
            sx = int(round(clamp(target_x, 0.0, float(self._screen_width - 1))))
            sy = int(round(clamp(target_y, 0.0, float(self._screen_height - 1))))
            return (sx, sy)

        # 2. Dead-zone filtering: eliminate micro-tremor when holding hand still
        if dist < self._settings.cursor_dead_zone:
            return (int(round(prev_x)), int(round(prev_y)))

        # 3. Speed-adaptive EMA smoothing
        base_alpha = clamp(self._settings.cursor_smoothing, 0.01, 1.0)
        if dist < 15.0:
            # Slow precision aiming: apply heavier filtering to suppress jitter
            alpha = max(0.05, base_alpha * 0.75)
        elif dist > 50.0:
            # Fast glide: increase responsiveness to prevent lag
            alpha = min(0.95, base_alpha * 1.5)
        else:
            alpha = base_alpha

        smooth_x = alpha * target_x + (1.0 - alpha) * prev_x
        smooth_y = alpha * target_y + (1.0 - alpha) * prev_y

        smooth_x = clamp(smooth_x, 0.0, float(self._screen_width - 1))
        smooth_y = clamp(smooth_y, 0.0, float(self._screen_height - 1))

        self._prev_smoothed = (smooth_x, smooth_y)
        return (int(round(smooth_x)), int(round(smooth_y)))

    # ------------------------------------------------------------------
    # Movement Dispatch & PyAutoGUI Safety
    # ------------------------------------------------------------------

    def move(self, x: float, y: float) -> bool:
        """
        Move the OS cursor to absolute screen coordinates.

        Preserves PyAutoGUI FAILSAFE = True.
        Never suppresses or bypasses FailSafeException.

        Parameters
        ----------
        x, y:
            Screen pixel coordinates.

        Returns
        -------
        bool
            True if cursor was successfully positioned.
        """
        if self._emergency_stopped or not self._enabled:
            return False

        if self._pag is None:
            log.debug("PyAutoGUI not available — move skipped.")
            return False

        if math.isnan(x) or math.isinf(x) or math.isnan(y) or math.isinf(y):
            log.warning("Invalid cursor coordinates rejected: (%s, %s)", x, y)
            return False

        target_x = int(round(clamp(x, 0.0, float(self._screen_width - 1))))
        target_y = int(round(clamp(y, 0.0, float(self._screen_height - 1))))

        # Avoid redundant OS calls if cursor position is unchanged
        if self._last_moved_pos == (target_x, target_y):
            return True

        # Throttle by update interval if configured
        now = time.monotonic()
        if self._settings.cursor_update_interval > 0.0:
            if (now - self._last_move_time) < self._settings.cursor_update_interval:
                return True

        try:
            self._pag.moveTo(target_x, target_y)
            self._last_moved_pos = (target_x, target_y)
            self._last_move_time = now
            return True
        except Exception as exc:
            # Check for PyAutoGUI FailSafeException
            is_failsafe = False
            failsafe_cls = getattr(self._pag, "FailSafeException", None)
            if failsafe_cls is not None and isinstance(exc, failsafe_cls):
                is_failsafe = True

            # Immediately engage emergency stop
            self.emergency_stop()

            if is_failsafe:
                log.critical(
                    "FAILSAFE TRIGGERED: Mouse moved to screen corner! Automation stopped immediately."
                )
                raise
            else:
                log.error("MouseController movement error: %s", exc)
                raise

    # ------------------------------------------------------------------
    # High-Level Hand & Gesture Processing
    # ------------------------------------------------------------------

    def process_hand(
        self,
        active_hand: Any,
        gesture_result: Any,
    ) -> tuple[int, int] | None:
        """
        Process detected hand landmarks and gesture classification to drive
        virtual mouse cursor movement.

        Rules:
        - Only confirmed, stable POINTING moves the cursor.
        - FIST, OPEN_PALM, PINCH, TWO_FINGER, UNKNOWN, or unconfirmed gestures
          immediately halt movement and reset smoothing.
        - Hand tracking loss or hand-identity switching resets smoothing to
          prevent cursor flings.

        Parameters
        ----------
        active_hand:
            Primary HandLandmarks object (or None if hand lost).
        gesture_result:
            GestureResult object containing classification and stability.

        Returns
        -------
        tuple[int, int] | None
            (screen_x, screen_y) if moved, None otherwise.
        """
        if self._emergency_stopped:
            self._state = MouseState.EMERGENCY_STOPPED
            return None

        if not self._enabled:
            self._state = MouseState.DISABLED
            return None

        # Hand lost: stop movement and reset smoothing
        if active_hand is None or gesture_result is None:
            self.reset_smoothing()
            self._state = MouseState.IDLE
            return None

        # Check for hand switching (e.g. Left -> Right hand)
        hand_id = getattr(active_hand, "handedness", None)
        if self._prev_hand_id is not None and hand_id is not None and self._prev_hand_id != hand_id:
            self.reset_smoothing()
        self._prev_hand_id = hand_id

        # Check gesture activation: ONLY stable POINTING moves cursor
        from gestureflow.gestures.classifier import Gesture

        gesture = getattr(gesture_result, "gesture", None)
        is_stable = getattr(gesture_result, "is_stable", False)

        if gesture != Gesture.POINTING or not is_stable:
            self._state = MouseState.IDLE
            self.reset_smoothing()
            return None

        # Extract index fingertip landmark (ID 8)
        landmarks = getattr(active_hand, "landmarks", None)
        if not landmarks or len(landmarks) <= 8:
            self.reset_smoothing()
            self._state = MouseState.IDLE
            return None

        lm8 = landmarks[8]
        if hasattr(lm8, "x") and hasattr(lm8, "y"):
            norm_x, norm_y = lm8.x, lm8.y
        elif isinstance(lm8, (list, tuple, np.ndarray)) and len(lm8) >= 2:
            norm_x, norm_y = lm8[0], lm8[1]
        else:
            self.reset_smoothing()
            self._state = MouseState.IDLE
            return None

        # Coordinate mapping
        target = self.map_coordinates(norm_x, norm_y)
        if target is None:
            return None

        # Smoothing & dead-zone
        smoothed = self.smooth_coordinates(target[0], target[1])
        if smoothed is None:
            return None

        # Move cursor
        self.move(float(smoothed[0]), float(smoothed[1]))
        self._state = MouseState.MOVING
        return smoothed

    # ------------------------------------------------------------------
    # Backward Compatibility Stubs (Phase 2B does not implement clicks/scrolls)
    # ------------------------------------------------------------------

    def click(self) -> None:
        """Perform a left click (stub — disabled in Phase 2B)."""
        log.debug("MouseController.click() — disabled in Phase 2B.")

    def right_click(self) -> None:
        """Perform a right click (stub — disabled in Phase 2B)."""
        log.debug("MouseController.right_click() — disabled in Phase 2B.")

    def scroll(self, clicks: int) -> None:
        """Scroll (stub — disabled in Phase 2B)."""
        log.debug("MouseController.scroll(%d) — disabled in Phase 2B.", clicks)
