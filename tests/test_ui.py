"""
Tests for UI components (MainWindow, GestureFlowApp, PipelineWorker)
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pytest
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication

from gestureflow.config.settings import Settings
from gestureflow.ui.main_window import MainWindow
from gestureflow.ui.app import GestureFlowApp
from gestureflow.tracking.pipeline import PipelineFrame
from gestureflow.tracking.hand_landmarker import TrackingResult


@pytest.fixture
def app_instance(qapp):
    """Ensure QApplication instance is available from pytest-qt."""
    return qapp


class TestMainWindow:
    def test_window_initial_state(self, app_instance, default_settings):
        window = MainWindow(settings=default_settings)
        assert window.windowTitle() == "GestureFlow — AI Virtual Mouse"
        assert window._btn_start.isEnabled() is True
        assert window._btn_stop.isEnabled() is False
        assert window._lbl_status.text() == "Ready"
        assert "FPS:" in window._lbl_fps.text()
        window.close()

    def test_frame_ready_slot_updates_ui(self, app_instance, default_settings):
        window = MainWindow(settings=default_settings)
        dummy_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        pipeline_frame = PipelineFrame(
            display_image=dummy_bgr,
            tracking=TrackingResult(hands=[], timestamp_ms=50),
            fps=28.5,
            frame_index=1,
        )
        window._on_frame_ready(pipeline_frame)
        assert window._lbl_hands.text() == "Hands detected: 0"
        assert window._lbl_fps.text() == "FPS: 28.5"
        assert window._preview_label.pixmap() is not None
        window.close()

    def test_status_changed_slot(self, app_instance, default_settings):
        window = MainWindow(settings=default_settings)
        window._on_status_changed("Custom Status")
        assert window._lbl_status.text() == "Custom Status"
        window.close()

    def test_close_event_cleans_up(self, app_instance, default_settings):
        window = MainWindow(settings=default_settings)
        window.close()
        assert window._log_handler is None


class TestGestureFlowApp:
    def test_app_configuration(self, app_instance, default_settings):
        import pyautogui
        # Verify failsafe is strictly enforced
        assert pyautogui.FAILSAFE is True
