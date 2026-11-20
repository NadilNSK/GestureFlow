"""
Tests for gestureflow.config.settings
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from gestureflow.config.settings import Settings, _deep_merge


class TestDeepMerge:
    def test_flat(self):
        base = {"a": 1, "b": 2}
        _deep_merge(base, {"b": 99, "c": 3})
        assert base == {"a": 1, "b": 99, "c": 3}

    def test_nested(self):
        base = {"camera": {"device_index": 0, "fps": 30}}
        _deep_merge(base, {"camera": {"fps": 60}})
        assert base["camera"]["fps"] == 60
        assert base["camera"]["device_index"] == 0

    def test_non_dict_override(self):
        base = {"key": {"nested": 1}}
        _deep_merge(base, {"key": "flat"})
        assert base["key"] == "flat"


class TestSettingsLoad:
    def test_defaults_loaded(self, project_root):
        s = Settings.load(defaults_path=project_root / "default_settings.json")
        assert s.camera.frame_width == 640
        assert s.camera.frame_height == 480
        assert s.tracking.num_hands == 2
        assert s.gestures.pinch_threshold == 0.065
        assert s.gestures.preferred_handedness == "Right"

    def test_failsafe_always_true(self, project_root, tmp_path):
        """Even if user sets failsafe=false it must be forced back to True."""
        user = tmp_path / "user_settings.json"
        user.write_text(json.dumps({"pyautogui": {"failsafe": False}}))
        s = Settings.load(
            defaults_path=project_root / "default_settings.json",
            user_path=user,
        )
        assert s.pyautogui.failsafe is True

    def test_user_override_applied(self, project_root, tmp_path):
        user = tmp_path / "user_settings.json"
        user.write_text(json.dumps({"camera": {"device_index": 2}}))
        s = Settings.load(
            defaults_path=project_root / "default_settings.json",
            user_path=user,
        )
        assert s.camera.device_index == 2
        # Other camera fields are still from defaults.
        assert s.camera.frame_width == 640

    def test_model_path_is_absolute(self, project_root):
        s = Settings.load(defaults_path=project_root / "default_settings.json")
        assert s.model_path.is_absolute()

    def test_missing_defaults_does_not_raise(self, tmp_path):
        s = Settings.load(
            defaults_path=tmp_path / "nonexistent.json",
        )
        # Falls back to dataclass defaults.
        assert s.camera.device_index == 1
        assert s.camera.backend == "DSHOW"
