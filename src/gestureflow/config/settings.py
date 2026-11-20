"""
gestureflow.config.settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Loads, validates, and provides typed access to application settings.

Priority (highest → lowest):
  1. user_settings.json (user overrides, excluded from VCS)
  2. default_settings.json (shipped defaults)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Project root is three levels up from this file (src/gestureflow/config/settings.py)
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


@dataclass
class CameraSettings:
    device_index: int = 1
    backend: str = "DSHOW"
    frame_width: int = 640
    frame_height: int = 480
    fps: int = 30
    flip_horizontal: bool = True
    use_mock_camera: bool = False


@dataclass
class TrackingSettings:
    num_hands: int = 2
    min_detection_confidence: float = 0.7
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    model_path: str = "assets/models/hand_landmarker.task"


@dataclass
class GestureSettings:
    pinch_threshold: float = 0.065
    pinch_release_threshold: float = 0.085
    confirmation_frames: int = 3
    loss_tolerance_frames: int = 3
    preferred_handedness: str = "Right"


@dataclass
class ControllerSettings:
    smoothing_factor: int = 7
    movement_threshold: int = 3
    click_cooldown_ms: int = 400
    scroll_speed: int = 3

    # Phase 2B additions:
    mouse_control_enabled_default: bool = False
    cursor_sensitivity: float = 1.0
    cursor_smoothing: float = 0.35
    cursor_dead_zone: float = 2.5
    tracking_region_margin: float = 0.15
    cursor_update_interval: float = 0.010
    cursor_jump_threshold: float = 250.0

    # Phase 2B.1 additions:
    smoothing_filter_type: str = "ema"  # "one_euro" or "ema" (default_settings.json uses "one_euro")
    one_euro_min_cutoff: float = 1.0
    one_euro_beta: float = 0.007
    one_euro_d_cutoff: float = 1.0
    precision_mode: bool = False
    precision_multiplier: float = 0.35
    max_cursor_velocity: float = 4000.0  # max pixels per second
    calibration_data: dict[str, Any] | None = None

    # Phase 2C additions (Mouse Actions & Gestures):
    left_click_enabled: bool = True
    right_click_enabled: bool = True
    double_click_enabled: bool = True
    drag_enabled: bool = True
    scroll_enabled: bool = True
    click_cooldown: float = 0.20
    double_click_interval: float = 0.35
    drag_hold_duration: float = 0.45
    scroll_sensitivity: float = 1.0
    scroll_dead_zone: float = 0.008
    scroll_update_interval: float = 0.04


@dataclass
class UiSettings:
    window_width: int = 1100
    window_height: int = 720
    preview_width: int = 640
    preview_height: int = 480
    theme: str = "dark"
    show_landmarks: bool = True
    show_fps: bool = True
    show_gesture_label: bool = True


@dataclass
class LoggingSettings:
    level: str = "INFO"
    file: str = "logs/gestureflow.log"
    max_bytes: int = 5_242_880  # 5 MB
    backup_count: int = 3


@dataclass
class PyAutoGuiSettings:
    failsafe: bool = True  # NEVER set to False
    pause: float = 0.0


@dataclass
class Settings:
    camera: CameraSettings = field(default_factory=CameraSettings)
    tracking: TrackingSettings = field(default_factory=TrackingSettings)
    gestures: GestureSettings = field(default_factory=GestureSettings)
    controller: ControllerSettings = field(default_factory=ControllerSettings)
    ui: UiSettings = field(default_factory=UiSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    pyautogui: PyAutoGuiSettings = field(default_factory=PyAutoGuiSettings)

    # Resolved absolute path to the MediaPipe model.
    model_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.model_path = _PROJECT_ROOT / self.tracking.model_path
        # Safety invariant: fail-safe must remain enabled.
        if not self.pyautogui.failsafe:
            log.warning(
                "PyAutoGUI fail-safe was disabled in settings — forcing it back on."
            )
            self.pyautogui.failsafe = True

        # Validate controller settings
        self.controller.cursor_sensitivity = max(0.1, min(5.0, float(self.controller.cursor_sensitivity)))
        self.controller.cursor_smoothing = max(0.01, min(1.0, float(self.controller.cursor_smoothing)))
        self.controller.cursor_dead_zone = max(0.0, float(self.controller.cursor_dead_zone))
        self.controller.tracking_region_margin = max(0.0, min(0.45, float(self.controller.tracking_region_margin)))
        self.controller.cursor_jump_threshold = max(10.0, float(self.controller.cursor_jump_threshold))
        self.controller.smoothing_filter_type = (
            "one_euro" if str(self.controller.smoothing_filter_type).lower() == "one_euro" else "ema"
        )
        self.controller.one_euro_min_cutoff = max(0.01, min(10.0, float(self.controller.one_euro_min_cutoff)))
        self.controller.one_euro_beta = max(0.0, min(1.0, float(self.controller.one_euro_beta)))
        self.controller.one_euro_d_cutoff = max(0.01, min(10.0, float(self.controller.one_euro_d_cutoff)))
        self.controller.precision_multiplier = max(0.05, min(1.0, float(self.controller.precision_multiplier)))
        self.controller.max_cursor_velocity = max(100.0, float(self.controller.max_cursor_velocity))

        # Validate Phase 2C controller settings
        self.controller.click_cooldown = max(0.05, min(2.0, float(self.controller.click_cooldown)))
        self.controller.double_click_interval = max(0.10, min(1.5, float(self.controller.double_click_interval)))
        self.controller.drag_hold_duration = max(0.15, min(2.0, float(self.controller.drag_hold_duration)))
        self.controller.scroll_sensitivity = max(0.1, min(10.0, float(self.controller.scroll_sensitivity)))
        self.controller.scroll_dead_zone = max(0.001, min(0.1, float(self.controller.scroll_dead_zone)))
        self.controller.scroll_update_interval = max(0.01, min(0.5, float(self.controller.scroll_update_interval)))

    @classmethod
    def load(
        cls,
        defaults_path: Path | None = None,
        user_path: Path | None = None,
    ) -> "Settings":
        """
        Load settings by deep-merging user overrides on top of defaults.

        Parameters
        ----------
        defaults_path:
            Path to the shipped ``default_settings.json``.  Defaults to
            ``<project_root>/default_settings.json``.
        user_path:
            Path to the optional ``user_settings.json``.  Defaults to
            ``<project_root>/user_settings.json``.
        """
        defaults_path = defaults_path or (_PROJECT_ROOT / "default_settings.json")
        user_path = user_path or (_PROJECT_ROOT / "user_settings.json")

        raw: dict[str, Any] = {}

        if defaults_path.exists():
            try:
                with defaults_path.open(encoding="utf-8") as fh:
                    raw = json.load(fh)
                log.debug("Loaded defaults from %s", defaults_path)
            except json.JSONDecodeError as exc:
                log.error("Failed to parse %s: %s", defaults_path, exc)
        else:
            log.warning("Default settings file not found: %s", defaults_path)

        if user_path.exists():
            try:
                with user_path.open(encoding="utf-8") as fh:
                    user_raw: dict[str, Any] = json.load(fh)
                _deep_merge(raw, user_raw)
                log.debug("Merged user overrides from %s", user_path)
            except json.JSONDecodeError as exc:
                log.error("Failed to parse user settings %s: %s", user_path, exc)

        return cls(
            camera=CameraSettings(**raw.get("camera", {})),
            tracking=TrackingSettings(**raw.get("tracking", {})),
            gestures=GestureSettings(**raw.get("gestures", {})),
            controller=ControllerSettings(**raw.get("controller", {})),
            ui=UiSettings(**raw.get("ui", {})),
            logging=LoggingSettings(**raw.get("logging", {})),
            pyautogui=PyAutoGuiSettings(**raw.get("pyautogui", {})),
        )

    def save_user(self, path: Path | None = None) -> None:
        """Persist current settings as user overrides."""
        path = path or (_PROJECT_ROOT / "user_settings.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "camera": self.camera.__dict__,
            "tracking": self.tracking.__dict__,
            "gestures": self.gestures.__dict__,
            "controller": self.controller.__dict__,
            "ui": self.ui.__dict__,
            "logging": self.logging.__dict__,
            "pyautogui": self.pyautogui.__dict__,
        }
        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        log.info("User settings saved to %s", path)

    def save_user_settings(self, path: Path | None = None) -> None:
        """Alias for save_user() to persist user overrides."""
        self.save_user(path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge *override* into *base* in-place."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
