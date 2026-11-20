"""
tests.test_camera_selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for automatic camera discovery, cv2.CAP_DSHOW backend selection,
camera switching in UI, and fail-safe mouse disabling during camera switch.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
import cv2
import pytest

from gestureflow.config.settings import Settings, CameraSettings
from gestureflow.camera.capture import (
    CameraCapture,
    DiscoveredCamera,
    discover_available_cameras,
)
from gestureflow.ui.main_window import MainWindow


class TestCameraDiscovery:
    def test_camera_settings_default_to_index_1_and_dshow(self):
        """Settings must default to camera device index 1 and DSHOW backend."""
        cam = CameraSettings()
        assert cam.device_index == 1
        assert cam.backend == "DSHOW"

    @patch("cv2.VideoCapture")
    def test_discover_available_cameras_detection(self, mock_vc_cls):
        """Test scanning finds Camera 1 and 2 as open, and Camera 0 as closed."""
        def make_mock_cap(idx, backend):
            mock_cap = MagicMock()
            if idx in (1, 2):
                mock_cap.isOpened.return_value = True
                mock_cap.read.return_value = (True, MagicMock())
            else:
                mock_cap.isOpened.return_value = False
                mock_cap.read.return_value = (False, None)
            return mock_cap

        mock_vc_cls.side_effect = make_mock_cap

        cams = discover_available_cameras(max_indices=4, backend="DSHOW")
        assert len(cams) == 4
        assert cams[0].is_available is False
        assert cams[1].is_available is True
        assert cams[2].is_available is True
        assert cams[3].is_available is False

    @patch("cv2.VideoCapture")
    def test_open_camera_uses_cap_dshow_backend(self, mock_vc_cls):
        """Test _open_camera specifies cv2.CAP_DSHOW when backend is DSHOW."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 640
        mock_vc_cls.return_value = mock_cap

        settings = CameraSettings(device_index=1, backend="DSHOW")
        capture = CameraCapture(settings)
        cap = capture._open_camera()

        mock_vc_cls.assert_called_with(1, cv2.CAP_DSHOW)
        assert cap is mock_cap

    @patch("cv2.VideoCapture")
    def test_open_camera_fails_gracefully_when_unavailable(self, mock_vc_cls):
        """Test an unavailable camera raises RuntimeError with clear troubleshooting text."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_vc_cls.return_value = mock_cap

        settings = CameraSettings(device_index=0, backend="DSHOW")
        capture = CameraCapture(settings)

        with pytest.raises(RuntimeError, match="Cannot open camera at device index 0"):
            capture._open_camera()


class TestCameraUiIntegration:
    def test_camera_dropdown_populated_in_main_window(self, qapp, default_settings):
        """MainWindow must display camera selection dropdown with current camera selected."""
        default_settings.camera.device_index = 1
        default_settings.camera.use_mock_camera = False
        window = MainWindow(settings=default_settings)
        assert hasattr(window, "_combo_camera")
        assert hasattr(window, "_btn_refresh_cameras")
        assert window._combo_camera.count() > 0

        # Current device index 1 should be selected by default
        selected_data = window._combo_camera.currentData()
        assert selected_data is not None
        assert selected_data["index"] == 1
        window.close()

    def test_camera_switching_disables_mouse_control(self, qapp, project_root, tmp_path):
        """Never enable virtual mouse control during camera switching."""
        settings = Settings.load(
            defaults_path=project_root / "default_settings.json",
            user_path=tmp_path / "user_settings.json",
        )
        settings.camera.device_index = 1
        settings.camera.use_mock_camera = False

        window = MainWindow(settings=settings)
        # Mock save_user_settings only for this window's settings instance
        window._settings.save_user_settings = MagicMock()

        # Simulate mouse being enabled
        window._btn_mouse_enable.setEnabled(False)
        window._btn_mouse_disable.setEnabled(True)
        window._lbl_mouse_status.setText("MOUSE: ACTIVE (MOVING)")

        # Find target combo index for device index 2
        target_combo_idx = -1
        for idx in range(window._combo_camera.count()):
            data = window._combo_camera.itemData(idx)
            if data and data.get("index") == 2:
                target_combo_idx = idx
                break

        if target_combo_idx != -1:
            window._combo_camera.setCurrentIndex(target_combo_idx)

        assert window._settings.camera.device_index == 2
        # Virtual mouse must be disabled
        assert window._btn_mouse_disable.isEnabled() is False
        assert "DISABLED" in window._lbl_mouse_status.text()
        window.close()

    def test_save_user_camera_settings(self, project_root, tmp_path):
        """User settings persistence for camera device and backend."""
        user_file = tmp_path / "user_settings.json"
        settings = Settings.load(
            defaults_path=project_root / "default_settings.json",
            user_path=user_file,
        )
        settings.camera.device_index = 2
        settings.camera.backend = "DSHOW"
        settings.save_user_settings(user_file)

        assert user_file.exists()
        saved = json.loads(user_file.read_text(encoding="utf-8"))
        assert saved["camera"]["device_index"] == 2
        assert saved["camera"]["backend"] == "DSHOW"
