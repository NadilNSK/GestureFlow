"""
gestureflow.controller.mouse_controller
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Production-quality virtual mouse controller using PyAutoGUI.
Dispatches smooth, fail-safe virtual cursor movement and full gesture mouse actions
derived from MediaPipe Hand Landmarker tracking.

Phase 2C Features:
1. Left Click: Single short thumb-index pinch (LEFT_PINCH).
2. Double Click: Two quick short thumb-index pinches within double_click_interval.
3. Right Click: Thumb-middle pinch (RIGHT_PINCH) with debounce.
4. Drag and Drop: Sustained thumb-index pinch held >= drag_hold_duration enters
   DRAGGING mode, issuing mouseDown(left) and allowing continuous calibrated movement.
   On release, issues mouseUp(left) without triggering a single-click.
5. Vertical Scrolling: Relative vertical hand displacement in TWO_FINGER mode.
   Rate-limited, dead-zone filtered, fractional accumulator.
6. Safety First:
   - Held mouse buttons unconditionally released on tracking loss, hand identity switch,
     camera switch, emergency stop, app shutdown, and controller exceptions.
   - Non-negotiable PyAutoGUI FAILSAFE = True.
   - Zero mouse actions in mock camera mode.
"""

from __future__ import annotations

import enum
import logging
import math
import time
from typing import Any, Optional

import numpy as np

from gestureflow.calibration.calibrator import CalibrationData, Calibrator
from gestureflow.config.settings import ControllerSettings
from gestureflow.gestures.classifier import Gesture
from gestureflow.utils.filters import OneEuroFilter
from gestureflow.utils.helpers import clamp

log = logging.getLogger(__name__)


class MouseState(str, enum.Enum):
    """Operational states of virtual mouse cursor movement."""

    DISABLED = "DISABLED"
    IDLE = "IDLE"                         # Enabled, but cursor not moving
    MOVING = "MOVING"                     # Stable pointing, cursor moving
    DRAGGING = "DRAGGING"                 # Drag mode, left button DOWN, cursor moving
    EMERGENCY_STOPPED = "EMERGENCY_STOPPED"


class MouseActionState(str, enum.Enum):
    """Dedicated states of the gesture mouse action state machine."""

    DISABLED = "DISABLED"
    IDLE = "IDLE"
    MOVING = "MOVING"
    LEFT_PINCH_PENDING = "LEFT_PINCH_PENDING"  # Holding thumb-index pinch (measuring click vs drag)
    CLICK_PENDING = "CLICK_PENDING"            # Short pinch released, awaiting possible 2nd pinch
    RIGHT_PINCH_PENDING = "RIGHT_PINCH_PENDING"  # Right pinch active
    DRAGGING = "DRAGGING"                      # Sustained pinch held, left button DOWN
    SCROLLING = "SCROLLING"                    # Two-finger gesture, vertical scroll active
    EMERGENCY_STOPPED = "EMERGENCY_STOPPED"


class MouseController:
    """
    Virtual mouse controller that maps hand landmark positions to Windows
    cursor coordinates and executes left click, double click, right click,
    drag-and-drop, and vertical scrolling with safety fail-safe enforcement.

    Parameters
    ----------
    settings:
        ControllerSettings containing sensitivity, smoothing, action timing, etc.
    pag_module:
        Optional PyAutoGUI-compatible module (injected for testing/mocking).
    screen_size:
        Optional (width, height) screen dimensions (injected for testing).
    mock_mode:
        True if camera is running in mock/synthetic mode. When True, mouse
        actions cannot be executed as a safety guard.
    calibrator:
        Optional Calibrator instance for calibrated 4-point screen mapping.
    """

    def __init__(
        self,
        settings: ControllerSettings | None = None,
        pag_module: Any = None,
        screen_size: tuple[int, int] | None = None,
        mock_mode: bool = False,
        calibrator: Calibrator | None = None,
    ) -> None:
        self._settings = settings or ControllerSettings()
        self._mock_mode = mock_mode

        # Calibration engine
        if calibrator is not None:
            self._calibrator = calibrator
        elif self._settings.calibration_data is not None:
            self._calibrator = Calibrator(
                CalibrationData.from_dict(self._settings.calibration_data)
            )
        else:
            self._calibrator = Calibrator()

        # One Euro Filter
        self._one_euro = OneEuroFilter(
            min_cutoff=self._settings.one_euro_min_cutoff,
            beta=self._settings.one_euro_beta,
            d_cutoff=self._settings.one_euro_d_cutoff,
        )

        # Precision mode state
        self._precision_mode = bool(self._settings.precision_mode)

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

        # Basic lifecycle state
        self._enabled = False
        self._emergency_stopped = False
        self._state = MouseState.DISABLED
        self._action_state = MouseActionState.DISABLED

        # Smoothing & tracking history
        self._prev_smoothed: Optional[tuple[float, float]] = None
        self._last_smooth_time: Optional[float] = None
        self._prev_hand_id: Optional[str] = None
        self._last_moved_pos: Optional[tuple[int, int]] = None
        self._last_move_time: float = 0.0

        # Action State Machine variables
        self._pinch_start_time: Optional[float] = None
        self._is_dragging: bool = False
        self._drag_started_this_pinch: bool = False
        self._pending_click_time: Optional[float] = None
        self._right_pinch_triggered: bool = False

        self._last_left_click_time: float = 0.0
        self._last_right_click_time: float = 0.0
        self._last_double_click_time: float = 0.0

        # Scrolling state
        self._prev_scroll_y: Optional[float] = None
        self._scroll_accumulator: float = 0.0
        self._last_scroll_time: float = 0.0
        self._scroll_direction: str = "IDLE"

        # Action metrics & telemetry
        self._left_click_count: int = 0
        self._right_click_count: int = 0
        self._double_click_count: int = 0
        self._last_action: str = "NONE"

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
    def is_precision_mode(self) -> bool:
        """True if precision aiming mode is active."""
        return self._precision_mode

    @property
    def is_dragging(self) -> bool:
        """True if currently in a drag-and-drop operation."""
        return self._is_dragging

    @property
    def state(self) -> MouseState:
        """Current operational state of cursor movement."""
        return self._state

    @property
    def action_state(self) -> MouseActionState:
        """Dedicated state of the mouse action state machine."""
        return self._action_state

    @property
    def last_action(self) -> str:
        """Name of the most recently performed mouse action."""
        return self._last_action

    @property
    def left_click_count(self) -> int:
        return self._left_click_count

    @property
    def right_click_count(self) -> int:
        return self._right_click_count

    @property
    def double_click_count(self) -> int:
        return self._double_click_count

    @property
    def click_strategy(self) -> str:
        """Current click dispatch strategy ('deferred' or 'native')."""
        return getattr(self._settings, "click_strategy", "deferred")

    @property
    def cursor_profile(self) -> str:
        """Current cursor response profile name."""
        return getattr(self._settings, "cursor_profile", "Balanced")

    @property
    def scroll_direction(self) -> str:
        return self._scroll_direction

    @property
    def screen_size(self) -> tuple[int, int]:
        """Detected primary screen dimensions (width, height)."""
        return (self._screen_width, self._screen_height)

    @property
    def settings(self) -> ControllerSettings:
        """Active controller configuration."""
        return self._settings

    @property
    def calibrator(self) -> Calibrator:
        """Active calibration engine."""
        return self._calibrator

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
        self._reset_action_states()
        log.info("MouseController: Virtual mouse control ENABLED.")
        return True

    def disable(self) -> None:
        """Disable virtual mouse control cleanly and release any held buttons."""
        self.release_held_buttons()
        self._enabled = False
        if not self._emergency_stopped:
            self._state = MouseState.DISABLED
        self.reset_smoothing()
        self._reset_action_states()
        log.info("MouseController: Virtual mouse control DISABLED.")

    def emergency_stop(self) -> None:
        """
        Immediately stop virtual mouse automation, release any held buttons,
        and block reactivation until reset_emergency_stop() is called.
        """
        self.release_held_buttons()
        self._emergency_stopped = True
        self._enabled = False
        self._state = MouseState.EMERGENCY_STOPPED
        self.reset_smoothing()
        self._reset_action_states()
        log.warning("MouseController: EMERGENCY STOP ACTIVATED. All automation halted.")

    def reset_emergency_stop(self) -> None:
        """Clear emergency stop condition back to clean disabled state."""
        self._emergency_stopped = False
        self._state = MouseState.DISABLED
        self.reset_smoothing()
        self._reset_action_states()
        log.info("MouseController: Emergency stop cleared. Controller is now DISABLED (ready to enable).")

    def reset_smoothing(self) -> None:
        """Reset historical coordinates and filter state to prevent cursor jumps."""
        self._prev_smoothed = None
        self._last_smooth_time = None
        self._last_moved_pos = None
        self._one_euro.reset()

    def _reset_action_states(self) -> None:
        """Reset internal timers and flags for click/drag/scroll state machines."""
        self._pinch_start_time = None
        self._drag_started_this_pinch = False
        self._pending_click_time = None
        self._right_pinch_triggered = False
        self._prev_scroll_y = None
        self._scroll_accumulator = 0.0
        self._scroll_direction = "IDLE"

    def release_held_buttons(self) -> None:
        """
        Safety guarantee: Unconditionally release any held mouse button.

        Called during tracking loss, hand switching, camera changes,
        shutdown, emergency stop, and exceptions.
        """
        if self._is_dragging:
            self._is_dragging = False
            if self._pag is not None and not self._mock_mode:
                try:
                    self._pag.mouseUp(button="left")
                    log.info("MouseController: Drag released safely via mouseUp(left).")
                except Exception as exc:
                    self._check_failsafe(exc)
                    log.error("Error releasing held mouse button: %s", exc)

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

    def set_filter_type(self, filter_type: str) -> None:
        """Set smoothing filter algorithm: 'one_euro' or 'ema'."""
        ft = str(filter_type).lower().strip()
        self._settings.smoothing_filter_type = "one_euro" if ft == "one_euro" else "ema"
        self.reset_smoothing()
        log.info("MouseController filter type set to: %s", self._settings.smoothing_filter_type)

    def set_one_euro_params(
        self,
        min_cutoff: Optional[float] = None,
        beta: Optional[float] = None,
        d_cutoff: Optional[float] = None,
    ) -> None:
        """Update One Euro Filter tuning parameters."""
        if min_cutoff is not None:
            self._settings.one_euro_min_cutoff = float(clamp(min_cutoff, 0.01, 10.0))
            self._one_euro.min_cutoff = self._settings.one_euro_min_cutoff
        if beta is not None:
            self._settings.one_euro_beta = float(clamp(beta, 0.0, 1.0))
            self._one_euro.beta = self._settings.one_euro_beta
        if d_cutoff is not None:
            self._settings.one_euro_d_cutoff = float(clamp(d_cutoff, 0.01, 10.0))
            self._one_euro.d_cutoff = self._settings.one_euro_d_cutoff

    def set_precision_mode(self, enabled: bool) -> None:
        """
        Enable or disable Precision Aiming Mode.

        Provides a seamless transition: previous smoothed position is maintained
        so toggling does not cause the cursor to jump or fling across the screen.
        """
        if self._precision_mode == enabled:
            return
        self._precision_mode = bool(enabled)
        self._settings.precision_mode = self._precision_mode
        if self._prev_smoothed is not None:
            self._one_euro.reset()
            self._one_euro.filter(self._prev_smoothed[0], self._prev_smoothed[1])
        log.info("Precision Mode toggled: %s", self._precision_mode)

    def set_calibrator(self, calibrator: Calibrator) -> None:
        """Assign an updated Calibrator instance."""
        self._calibrator = calibrator
        self.reset_smoothing()

    def set_click_strategy(self, strategy: str) -> None:
        """
        Set click strategy ('deferred' for Safe Deferred, 'native' for Responsive Native).
        """
        strat = str(strategy).lower().strip()
        self._settings.click_strategy = "native" if strat == "native" else "deferred"
        self._pending_click_time = None
        log.info("MouseController: click strategy set to %s", self._settings.click_strategy)

    def set_cursor_profile(self, profile_name: str) -> bool:
        """
        Apply a preset cursor response profile ('Balanced', 'Precision', 'Responsive', 'Custom').

        Preserves 4-point calibration and active tracking region.
        """
        from gestureflow.config.settings import CURSOR_PROFILES

        p_name = str(profile_name).strip().capitalize()
        if p_name == "Custom":
            self._settings.cursor_profile = "Custom"
            return True

        if p_name not in CURSOR_PROFILES:
            log.warning("Unknown cursor profile '%s'.", profile_name)
            return False

        preset = CURSOR_PROFILES[p_name]
        for k, v in preset.items():
            if hasattr(self._settings, k):
                setattr(self._settings, k, v)

        # Update One Euro parameters on active filter
        self._one_euro.min_cutoff = self._settings.one_euro_min_cutoff
        self._one_euro.beta = self._settings.one_euro_beta
        self._one_euro.d_cutoff = self._settings.one_euro_d_cutoff
        self._settings.cursor_profile = p_name
        self.reset_smoothing()
        log.info("MouseController: applied cursor profile '%s'.", p_name)
        return True

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
        [0, screen_width - 1] and [0, screen_height - 1].
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

        # 1. Calibrated mapping if calibrator is active
        if self._calibrator.is_calibrated and margin is None:
            screen_x, screen_y = self._calibrator.map_point(
                x_val, y_val, self._screen_width, self._screen_height
            )
            sens = self._settings.cursor_sensitivity
            if sens != 1.0:
                cx = float(self._screen_width - 1) * 0.5
                cy = float(self._screen_height - 1) * 0.5
                screen_x = clamp(cx + (screen_x - cx) * sens, 0.0, float(self._screen_width - 1))
                screen_y = clamp(cy + (screen_y - cy) * sens, 0.0, float(self._screen_height - 1))
            return (screen_x, screen_y)

        # 2. Standard Active tracking region margin mapping
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
        timestamp: Optional[float] = None,
    ) -> tuple[int, int] | None:
        """
        Smooth target screen coordinates using One Euro Filter or EMA,
        with velocity limiting, precision mode scaling, and dead-zone filtering.
        """
        now = time.monotonic() if timestamp is None else float(timestamp)

        if self._prev_smoothed is None:
            sx = clamp(target_x, 0.0, float(self._screen_width - 1))
            sy = clamp(target_y, 0.0, float(self._screen_height - 1))
            self._prev_smoothed = (sx, sy)
            self._last_smooth_time = now
            self._one_euro.reset()
            self._one_euro.filter(sx, sy, timestamp=now)
            return (int(round(sx)), int(round(sy)))

        prev_x, prev_y = self._prev_smoothed

        if timestamp is not None:
            now = float(timestamp)
            dt: Optional[float] = (now - self._last_smooth_time) if self._last_smooth_time is not None else 0.033
            if dt is not None and dt <= 0.0:
                dt = 0.001
            self._last_smooth_time = now
        else:
            now = time.monotonic()
            dt = None
            self._last_smooth_time = now

        dx = target_x - prev_x
        dy = target_y - prev_y
        dist = math.hypot(dx, dy)

        # 1. Large jump detection: if tracking skipped across the screen, snap to target
        if dist > self._settings.cursor_jump_threshold:
            sx = clamp(target_x, 0.0, float(self._screen_width - 1))
            sy = clamp(target_y, 0.0, float(self._screen_height - 1))
            self._prev_smoothed = (sx, sy)
            self._one_euro.reset()
            self._one_euro.filter(sx, sy, timestamp=now)
            return (int(round(sx)), int(round(sy)))

        # 2. Tracking Stability: Velocity clamping
        if dt is not None and dt >= 0.005 and self._settings.max_cursor_velocity > 0.0 and dist > 0.0:
            max_step = self._settings.max_cursor_velocity * dt
            if dist > max_step:
                ratio = max_step / dist
                target_x = prev_x + dx * ratio
                target_y = prev_y + dy * ratio
                dx = target_x - prev_x
                dy = target_y - prev_y
                dist = max_step

        # 3. Precision Mode scaling
        if self._precision_mode:
            mult = self._settings.precision_multiplier
            target_x = prev_x + dx * mult
            target_y = prev_y + dy * mult
            dist = dist * mult

        # 4. Dead-zone filtering
        if dist < self._settings.cursor_dead_zone:
            return (int(round(prev_x)), int(round(prev_y)))

        # 5. Filter execution: One Euro Filter vs EMA
        if self._settings.smoothing_filter_type == "one_euro":
            smooth_x, smooth_y = self._one_euro.filter(target_x, target_y, timestamp=now)
        else:
            base_alpha = clamp(self._settings.cursor_smoothing, 0.01, 1.0)
            if dist < 15.0:
                alpha = max(0.05, base_alpha * 0.75)
            elif dist > 50.0:
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

        if self._last_moved_pos == (target_x, target_y):
            return True

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
            self._handle_pag_exception(exc)
            raise

    # ------------------------------------------------------------------
    # Mouse Action Execution (Left, Double, Right, Drag, Scroll)
    # ------------------------------------------------------------------

    def click(self, button: str = "left", timestamp: Optional[float] = None) -> bool:
        """
        Execute a single mouse click.
        """
        if not self._can_execute_actions():
            return False

        now = time.monotonic() if timestamp is None else float(timestamp)
        btn = button.lower()
        try:
            if btn == "left":
                self._pag.click(button="left")
                self._left_click_count += 1
                self._last_action = "LEFT CLICK"
                self._last_left_click_time = now
                log.info("Virtual Mouse: LEFT CLICK executed (total: %d)", self._left_click_count)
            elif btn == "right":
                self._pag.rightClick()
                self._right_click_count += 1
                self._last_action = "RIGHT CLICK"
                self._last_right_click_time = now
                log.info("Virtual Mouse: RIGHT CLICK executed (total: %d)", self._right_click_count)
            return True
        except Exception as exc:
            self._handle_pag_exception(exc)
            raise

    def double_click(self, timestamp: Optional[float] = None) -> bool:
        """
        Execute a double-click action.
        """
        if not self._can_execute_actions():
            return False

        now = time.monotonic() if timestamp is None else float(timestamp)
        try:
            self._pag.doubleClick(button="left")
            self._double_click_count += 1
            self._last_action = "DOUBLE CLICK"
            self._last_double_click_time = now
            self._last_left_click_time = now
            log.info("Virtual Mouse: DOUBLE CLICK executed (total: %d)", self._double_click_count)
            return True
        except Exception as exc:
            self._handle_pag_exception(exc)
            raise

    def right_click(self, timestamp: Optional[float] = None) -> bool:
        """
        Execute a right click.
        """
        return self.click(button="right", timestamp=timestamp)

    def mouse_down(self, button: str = "left") -> bool:
        """
        Press and hold the mouse button (enters dragging).
        """
        if not self._can_execute_actions():
            return False

        try:
            self._pag.mouseDown(button=button)
            self._is_dragging = True
            self._last_action = "DRAGGING"
            self._state = MouseState.DRAGGING
            log.info("Virtual Mouse: DRAG STARTED (mouseDown)")
            return True
        except Exception as exc:
            self._handle_pag_exception(exc)
            raise

    def mouse_up(self, button: str = "left") -> bool:
        """
        Release held mouse button (exits dragging).
        """
        if not self._can_execute_actions():
            self._is_dragging = False
            return False

        try:
            self._pag.mouseUp(button=button)
            self._is_dragging = False
            self._last_action = "DRAG RELEASE"
            log.info("Virtual Mouse: DRAG RELEASED (mouseUp)")
            return True
        except Exception as exc:
            self._handle_pag_exception(exc)
            raise

    def scroll(self, clicks: int) -> bool:
        """
        Scroll vertically by the specified number of clicks.
        """
        if not self._can_execute_actions() or clicks == 0:
            return False

        try:
            self._pag.scroll(clicks)
            self._scroll_direction = "UP" if clicks > 0 else "DOWN"
            self._last_action = f"SCROLL {self._scroll_direction}"
            log.debug("Virtual Mouse: SCROLL %d clicks", clicks)
            return True
        except Exception as exc:
            self._handle_pag_exception(exc)
            raise

    def _can_execute_actions(self) -> bool:
        """Verify action dispatch is permitted by safety invariants."""
        if self._emergency_stopped or not self._enabled:
            return False
        if self._mock_mode:
            log.warning("Safety protection: Mouse actions blocked in mock mode.")
            return False
        if self._pag is None:
            log.debug("PyAutoGUI not available — action skipped.")
            return False
        return True

    def _check_failsafe(self, exc: Exception) -> bool:
        """Detect and react to PyAutoGUI FailSafeException."""
        failsafe_cls = getattr(self._pag, "FailSafeException", None)
        if failsafe_cls is not None and isinstance(exc, failsafe_cls):
            self.emergency_stop()
            log.critical("FAILSAFE TRIGGERED: Mouse moved to corner! Automation stopped.")
            return True
        return False

    def _handle_pag_exception(self, exc: Exception) -> None:
        """Engage emergency stop on any PyAutoGUI failure."""
        is_failsafe = self._check_failsafe(exc)
        if not is_failsafe:
            self.emergency_stop()
            log.error("MouseController automation error: %s", exc)

    # ------------------------------------------------------------------
    # High-Level Hand & Gesture Processing
    # ------------------------------------------------------------------

    def process_hand(
        self,
        active_hand: Any,
        gesture_result: Any,
        timestamp: Optional[float] = None,
    ) -> tuple[int, int] | None:
        """
        Process detected hand landmarks and gesture classification to drive
        virtual cursor movement and gesture actions (click, double-click,
        right-click, drag-and-drop, and scroll).
        """
        now = time.monotonic() if timestamp is None else float(timestamp)

        if self._emergency_stopped:
            self._state = MouseState.EMERGENCY_STOPPED
            return None

        if not self._enabled:
            self._state = MouseState.DISABLED
            return None

        # 1. Check expiration of pending single click (Double Click State Machine)
        if self._pending_click_time is not None:
            if (now - self._pending_click_time) >= self._settings.double_click_interval:
                # Double-click interval expired:
                strategy = getattr(self._settings, "click_strategy", "deferred")
                if strategy == "deferred":
                    # Deferred strategy: fire the delayed single click now!
                    self._pending_click_time = None
                    self.click(button="left", timestamp=now)
                    if self._action_state == MouseActionState.CLICK_PENDING:
                        self._action_state = MouseActionState.IDLE
                else:
                    # Native strategy: first click was already fired immediately!
                    self._pending_click_time = None
                    if self._action_state == MouseActionState.CLICK_PENDING:
                        self._action_state = MouseActionState.IDLE

        # 2. Hand lost: release any held button safely, reset smoothing and gestures
        if active_hand is None or gesture_result is None:
            self.release_held_buttons()
            self.reset_smoothing()
            self._reset_action_states()
            self._state = MouseState.IDLE
            self._action_state = MouseActionState.IDLE
            return None

        # 3. Hand identity switching (e.g. Left -> Right hand)
        hand_id = getattr(active_hand, "handedness", None)
        if self._prev_hand_id is not None and hand_id is not None and self._prev_hand_id != hand_id:
            self.release_held_buttons()
            self.reset_smoothing()
            self._reset_action_states()
            self._action_state = MouseActionState.IDLE
        self._prev_hand_id = hand_id

        # 4. Extract landmarks and classification
        gesture = getattr(gesture_result, "gesture", Gesture.UNKNOWN)
        is_stable = getattr(gesture_result, "is_stable", False)
        landmarks = getattr(active_hand, "landmarks", None)

        # --------------------------------------------------------------
        # GESTURE: LEFT_PINCH (Left Click, Double Click, Drag and Drop)
        # --------------------------------------------------------------
        if gesture == Gesture.LEFT_PINCH and is_stable:
            # Clear scrolling state
            self._prev_scroll_y = None
            self._scroll_accumulator = 0.0

            if self._pinch_start_time is None:
                self._pinch_start_time = now
                self._drag_started_this_pinch = False
                self._action_state = MouseActionState.LEFT_PINCH_PENDING

            elapsed_pinch = now - self._pinch_start_time

            # Check drag threshold
            if self._settings.drag_enabled and not self._is_dragging and elapsed_pinch >= self._settings.drag_hold_duration:
                # Enter dragging mode!
                self.mouse_down(button="left")
                self._drag_started_this_pinch = True
                # A drag cancels any pending single click
                self._pending_click_time = None

            if self._is_dragging:
                # While dragging, cursor moves with index fingertip
                self._state = MouseState.DRAGGING
                self._action_state = MouseActionState.DRAGGING
                return self._update_cursor_movement(landmarks, timestamp=now)

            # Still in pending short pinch: hold cursor steady
            self._state = MouseState.IDLE
            self._action_state = MouseActionState.LEFT_PINCH_PENDING
            return None

        # If pinch just ended in this frame:
        if self._pinch_start_time is not None and (gesture != Gesture.LEFT_PINCH or not is_stable):
            if self._is_dragging:
                # Release drag
                self.mouse_up(button="left")
                self._state = MouseState.IDLE
                self._action_state = MouseActionState.IDLE
            elif not self._drag_started_this_pinch:
                # Completed short pinch -> process single / double click
                elapsed_pinch = now - self._pinch_start_time
                if elapsed_pinch < self._settings.drag_hold_duration:
                    self._handle_completed_short_left_pinch(now)

            self._pinch_start_time = None
            self._drag_started_this_pinch = False

        # --------------------------------------------------------------
        # GESTURE: RIGHT_PINCH (Right Click)
        # --------------------------------------------------------------
        if gesture == Gesture.RIGHT_PINCH and is_stable:
            self._prev_scroll_y = None
            self._scroll_accumulator = 0.0

            if not self._is_dragging and self._settings.right_click_enabled:
                if not self._right_pinch_triggered:
                    if (now - self._last_right_click_time) >= self._settings.click_cooldown:
                        self.right_click(timestamp=now)
                        self._right_pinch_triggered = True
                        self._action_state = MouseActionState.RIGHT_PINCH_PENDING
            self._state = MouseState.IDLE
            return None
        else:
            self._right_pinch_triggered = False

        # --------------------------------------------------------------
        # GESTURE: TWO_FINGER (Vertical Scrolling)
        # --------------------------------------------------------------
        if gesture == Gesture.TWO_FINGER and is_stable and not self._is_dragging:
            if self._settings.scroll_enabled:
                self._handle_vertical_scroll(landmarks, now)
                self._action_state = MouseActionState.SCROLLING
            self._state = MouseState.IDLE
            return None

        # Reset scrolling reference when not in two-finger gesture
        self._prev_scroll_y = None
        self._scroll_accumulator = 0.0
        self._scroll_direction = "IDLE"

        # --------------------------------------------------------------
        # GESTURE: POINTING (Virtual Mouse Cursor Movement)
        # --------------------------------------------------------------
        if gesture == Gesture.POINTING and is_stable and not self._is_dragging:
            self._action_state = MouseActionState.MOVING
            return self._update_cursor_movement(landmarks, timestamp=now)

        # --------------------------------------------------------------
        # OTHER GESTURES / UNSTABLE / IDLE
        # --------------------------------------------------------------
        self._state = MouseState.IDLE
        if self._action_state != MouseActionState.CLICK_PENDING:
            self._action_state = MouseActionState.IDLE
        self.reset_smoothing()
        return None

    def _handle_completed_short_left_pinch(self, now: float) -> None:
        """Process a completed short pinch for single or double click."""
        if not self._settings.left_click_enabled:
            return

        if (now - self._last_left_click_time) < self._settings.click_cooldown:
            log.debug("Left pinch ignored due to click cooldown.")
            return

        strategy = getattr(self._settings, "click_strategy", "deferred")

        if self._settings.double_click_enabled:
            if (
                self._pending_click_time is not None
                and (now - self._pending_click_time) <= self._settings.double_click_interval
            ):
                if strategy == "native":
                    # Native Strategy: Second click executed immediately; Windows registers native double click!
                    self._pending_click_time = None
                    self.click(button="left", timestamp=now)
                    self._double_click_count += 1
                    self._last_action = "DOUBLE CLICK (NATIVE)"
                    log.info("Virtual Mouse: NATIVE DOUBLE CLICK (2nd click, total: %d)", self._double_click_count)
                    self._state = MouseState.IDLE
                    self._action_state = MouseActionState.IDLE
                else:
                    # Safe Deferred Strategy: Fire single logical doubleClick action!
                    self._pending_click_time = None
                    self.double_click(timestamp=now)
                    self._state = MouseState.IDLE
                    self._action_state = MouseActionState.IDLE
            else:
                if strategy == "native":
                    # Native Strategy: Fire first click IMMEDIATELY without waiting!
                    self.click(button="left", timestamp=now)
                    self._pending_click_time = now
                    self._state = MouseState.IDLE
                    self._action_state = MouseActionState.CLICK_PENDING
                else:
                    # Safe Deferred Strategy: Start double-click window
                    self._pending_click_time = now
                    self._state = MouseState.IDLE
                    self._action_state = MouseActionState.CLICK_PENDING
        else:
            # Double click disabled: issue single click immediately
            self.click(button="left", timestamp=now)
            self._state = MouseState.IDLE
            self._action_state = MouseActionState.IDLE

    def _handle_vertical_scroll(self, landmarks: Any, now: float) -> None:
        """Compute relative vertical hand movement and dispatch OS scroll."""
        if not landmarks or len(landmarks) <= 5:
            self._prev_scroll_y = None
            return

        # Landmark 5 (Index MCP knuckle) provides clean vertical reference
        lm5 = landmarks[5]
        norm_y = getattr(lm5, "y", lm5[1] if isinstance(lm5, (list, tuple)) else None)
        if norm_y is None:
            return

        curr_y = float(norm_y)
        if self._prev_scroll_y is not None:
            # In normalized frame space: moving hand UP reduces y (curr_y < prev_y)
            # Moving hand UP should scroll UP (positive clicks).
            dy = -(curr_y - self._prev_scroll_y)

            if abs(dy) >= self._settings.scroll_dead_zone:
                # Accumulate displacement
                step_scale = 120.0 * self._settings.scroll_sensitivity
                self._scroll_accumulator += dy * step_scale

                if (now - self._last_scroll_time) >= self._settings.scroll_update_interval:
                    clicks = int(round(self._scroll_accumulator))
                    if clicks != 0:
                        self.scroll(clicks)
                        self._scroll_accumulator -= clicks
                        self._last_scroll_time = now

        self._prev_scroll_y = curr_y

    def _update_cursor_movement(
        self,
        landmarks: Any,
        timestamp: float,
    ) -> tuple[int, int] | None:
        """Map fingertip landmarks and move cursor."""
        if not landmarks or len(landmarks) <= 8:
            self.reset_smoothing()
            return None

        lm8 = landmarks[8]
        if hasattr(lm8, "x") and hasattr(lm8, "y"):
            norm_x, norm_y = lm8.x, lm8.y
        elif isinstance(lm8, (list, tuple, np.ndarray)) and len(lm8) >= 2:
            norm_x, norm_y = lm8[0], lm8[1]
        else:
            self.reset_smoothing()
            return None

        # Coordinate mapping
        target = self.map_coordinates(norm_x, norm_y)
        if target is None:
            return None

        # Smoothing & dead-zone
        smoothed = self.smooth_coordinates(target[0], target[1], timestamp=timestamp)
        if smoothed is None:
            return None

        # Move OS cursor
        self.move(float(smoothed[0]), float(smoothed[1]))
        if not self._is_dragging:
            self._state = MouseState.MOVING
            self._action_state = MouseActionState.MOVING
        else:
            self._state = MouseState.DRAGGING
            self._action_state = MouseActionState.DRAGGING
        return smoothed
