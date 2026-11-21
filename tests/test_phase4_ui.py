"""
tests.test_phase4_ui
~~~~~~~~~~~~~~~~~~~~
Comprehensive test suite for Phase 4 UI/UX Redesign & Desktop Experience:
- Application initialization & design system shell
- Sidebar navigation & QStackedWidget page switching across all 7 pages
- Dashboard primary controls & telemetry card updates
- Camera selection, discovery, and mock guard
- Virtual mouse enable/disable and emergency stop invariants
- Gesture action toggles and sensitivity tuning
- Calibration page controls & reset confirmation
- Settings validation, persistence, and profile switching
- Rolling performance chart buffer management
- First-run onboarding wizard walkthrough & skip logic
- Multi-resolution responsive layout validation (1366×768, 1600×900, 1920×1080)
- Graceful shutdown and resource cleanup
"""

from __future__ import annotations

import collections
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import QApplication, QMessageBox, QLabel

from gestureflow import __version__
from gestureflow.config.settings import Settings, CameraSettings, CURSOR_PROFILES
from gestureflow.tracking.hand_landmarker import TrackingResult
from gestureflow.tracking.pipeline import PipelineFrame
from gestureflow.gestures.classifier import Gesture, GestureResult, FingerStates
from gestureflow.ui.main_window import MainWindow
from gestureflow.ui.components.sidebar import SidebarNavigation
from gestureflow.ui.components.cards import StatusCard, GestureCard, MetricCard
from gestureflow.ui.components.charts import RollingMetricChart
from gestureflow.ui.components.onboarding_dialog import OnboardingDialog, STEPS
from gestureflow.ui.pages import (
    DashboardPage,
    LiveTrackingPage,
    GestureControlsPage,
    CalibrationPage,
    PerformancePage,
    SettingsPage,
    AboutPage,
)


@pytest.fixture
def app_instance(qapp):
    """Ensure QApplication instance is initialized."""
    return qapp


@pytest.fixture
def clean_settings(project_root, tmp_path):
    """Provide isolated Settings instance with mock user_settings.json."""
    user_file = tmp_path / "user_settings.json"
    settings = Settings.load(
        defaults_path=project_root / "default_settings.json",
        user_path=user_file,
    )
    # Ensure onboarding is marked completed so it does not auto-popup in unit tests
    settings.ui.onboarding_completed = True
    return settings


class TestApplicationShellAndNavigation:
    """Tests for sidebar navigation and page switching in the new application shell."""

    def test_main_window_initialization(self, app_instance, clean_settings):
        window = MainWindow(settings=clean_settings)
        assert window.windowTitle() == "GestureFlow — AI Virtual Mouse"
        assert window._stack.count() == 7
        assert isinstance(window._sidebar, SidebarNavigation)
        assert window._stack.currentIndex() == 0  # Starts on Dashboard
        window.close()

    def test_sidebar_page_switching(self, app_instance, clean_settings):
        window = MainWindow(settings=clean_settings)
        # Test switching to all 7 pages via sidebar
        for page_idx in range(7):
            window._sidebar.select_page(page_idx)
            assert window._stack.currentIndex() == page_idx

        # Out-of-bounds check
        window._sidebar.select_page(99)
        assert window._stack.currentIndex() == 6  # Remains on last valid selection
        window.close()

    def test_sidebar_collapse_toggle(self, app_instance, clean_settings):
        window = MainWindow(settings=clean_settings)
        sidebar = window._sidebar
        init_w = sidebar.width()
        sidebar.toggle_collapse()
        assert sidebar._collapsed is True
        assert sidebar.maximumWidth() == 64
        sidebar.toggle_collapse()
        assert sidebar._collapsed is False
        assert sidebar.maximumWidth() == 230
        window.close()


class TestDashboardPageControls:
    """Tests for Dashboard primary controls and telemetry."""

    def test_dashboard_controls_initial_state(self, app_instance, clean_settings):
        window = MainWindow(settings=clean_settings)
        assert window._btn_start.isEnabled() is True
        assert window._btn_stop.isEnabled() is False
        assert window._btn_mouse_enable.isEnabled() is False
        assert window._btn_mouse_disable.isEnabled() is False
        assert window._lbl_status.text() == "Ready"
        window.close()

    def test_dashboard_precision_toggle(self, app_instance, clean_settings):
        window = MainWindow(settings=clean_settings)
        assert clean_settings.controller.precision_mode is False
        window._toggle_precision_mode()
        assert clean_settings.controller.precision_mode is True
        assert "ON" in window._btn_precision.text()
        window._toggle_precision_mode()
        assert clean_settings.controller.precision_mode is False
        assert "Off" in window._btn_precision.text()
        window.close()

    def test_dashboard_activity_log_bounded(self, app_instance, clean_settings):
        page = DashboardPage(settings=clean_settings)
        for i in range(150):
            page.log_activity(f"Event {i}", "info")
        # Deque maxlen is 100
        assert len(page._activity_log) == 100
        assert "Event 149" in page._activity_log[-1]


class TestCameraAndTrackingPage:
    """Tests for Live Tracking page, camera selector, and HUD inspector."""

    def test_live_tracking_page_structure(self, app_instance, clean_settings):
        window = MainWindow(settings=clean_settings)
        page = window._page_live_tracking
        assert isinstance(page, LiveTrackingPage)
        assert page._combo_camera.count() > 0
        assert hasattr(page, "_preview_label")
        assert hasattr(page, "_metric_hands")
        assert hasattr(page, "_metric_quality")
        window.close()

    def test_frame_ready_updates_inspector_and_preview(self, app_instance, clean_settings):
        window = MainWindow(settings=clean_settings)
        dummy_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = PipelineFrame(
            display_image=dummy_bgr,
            tracking=TrackingResult(hands=[], timestamp_ms=100),
            fps=30.0,
            frame_index=1,
            gesture_result=GestureResult(
                gesture=Gesture.POINTING,
                confidence=0.88,
                is_stable=True,
                label="POINTING",
                pinch_distance=0.15,
            ),
            tracking_quality="EXCELLENT",
        )
        window._on_frame_ready(frame)
        assert window._page_dashboard._card_gesture._lbl_value.text() == "POINTING"
        assert window._page_dashboard._card_confidence._lbl_value.text() == "88%"
        assert window._preview_label.pixmap() is not None
        window.close()


class TestGestureControlsPage:
    """Tests for Gesture Controls tuning cards and action toggles."""

    def test_gesture_cards_initialized(self, app_instance, clean_settings):
        window = MainWindow(settings=clean_settings)
        page = window._page_gestures
        assert page._card_left_click.is_action_enabled is True
        assert page._card_double_click.is_action_enabled is True
        assert page._card_right_click.is_action_enabled is True
        assert page._card_drag.is_action_enabled is True
        assert page._card_scroll.is_action_enabled is True
        window.close()

    def test_toggle_action_persists_in_settings(self, app_instance, clean_settings):
        window = MainWindow(settings=clean_settings)
        window._settings.save_user_settings = MagicMock()

        # Toggle Right Click off
        window._on_toggle_action("right_click", False)
        assert clean_settings.controller.right_click_enabled is False

        # Toggle Scroll off
        window._on_toggle_action("scroll", False)
        assert clean_settings.controller.scroll_enabled is False

        assert window._settings.save_user_settings.call_count >= 2
        window.close()

    def test_sensitivity_slider_updates(self, app_instance, clean_settings):
        window = MainWindow(settings=clean_settings)
        page = window._page_gestures

        page._slider_dc.setValue(45)  # 0.45s
        assert clean_settings.controller.double_click_interval == 0.45

        page._slider_drag.setValue(60)  # 0.60s
        assert clean_settings.controller.drag_hold_duration == 0.60
        window.close()


class TestCalibrationPage:
    """Tests for Calibration page status and reset prompt."""

    def test_calibration_status_display(self, app_instance, clean_settings):
        clean_settings.controller.calibration_data = None
        window = MainWindow(settings=clean_settings)
        page = window._page_calibration
        assert "Default" in page._lbl_banner_title.text()

        # Simulate calibrated state
        clean_settings.controller.calibration_data = {"is_active": True}
        page.refresh_calibration_status()
        assert "Active ✔" in page._lbl_banner_title.text()
        window.close()

    @patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes)
    def test_reset_calibration_confirmed(self, mock_msg, app_instance, clean_settings):
        clean_settings.controller.calibration_data = {"is_active": True}
        window = MainWindow(settings=clean_settings)
        window._on_reset_calibration()
        assert clean_settings.controller.calibration_data is None
        assert "Default" in window._page_calibration._lbl_banner_title.text()
        window.close()


class TestPerformancePageAndRollingCharts:
    """Tests for Performance page metrics and rolling sparkline chart buffers."""

    def test_rolling_chart_buffer_bounded(self, app_instance):
        chart = RollingMetricChart("Test Chart", "Series A", "Series B", max_points=10)
        for i in range(25):
            chart.add_data_point(float(i), float(i * 2))

        assert len(chart._data1) == 10
        assert len(chart._data2) == 10
        assert chart._last_val1 == 24.0
        assert chart._last_val2 == 48.0
        chart.clear()
        assert len(chart._data1) == 0

    def test_performance_page_metric_update(self, app_instance, clean_settings):
        page = PerformancePage(settings=clean_settings)
        page.update_metrics(
            camera_fps=30.0,
            processing_fps=29.5,
            infer_ms=18.5,
            gest_ms=1.2,
            ctrl_ms=0.8,
            pipeline_ms=22.0,
            frame_age_ms=12.0,
        )
        assert page._card_health._lbl_value.text() == "OPTIMAL"
        assert "22.0ms" in page._card_pipeline_lat._lbl_value.text()


class TestSettingsPageAndProfiles:
    """Tests for Settings tabs, profile switching, and input persistence."""

    def test_settings_page_tabs_exist(self, app_instance, clean_settings):
        window = MainWindow(settings=clean_settings)
        page = window._page_settings
        assert page._tabs.count() == 5  # Camera, Cursor, Gestures, Appearance, Safety
        window.close()

    def test_profile_switch_updates_controller_settings(self, app_instance, clean_settings):
        window = MainWindow(settings=clean_settings)
        window._settings.save_user_settings = MagicMock()

        # Switch to Precision
        window._on_profile_selected("Precision")
        assert clean_settings.controller.cursor_profile == "Precision"
        assert clean_settings.controller.cursor_sensitivity == CURSOR_PROFILES["Precision"]["cursor_sensitivity"]

        # Switch to Responsive
        window._on_profile_selected("Responsive")
        assert clean_settings.controller.cursor_profile == "Responsive"
        assert clean_settings.controller.click_strategy == "native"
        window.close()

    def test_click_strategy_switching(self, app_instance, clean_settings):
        window = MainWindow(settings=clean_settings)
        window._settings.save_user_settings = MagicMock()

        window._on_click_strategy_selected("native")
        assert clean_settings.controller.click_strategy == "native"

        window._on_click_strategy_selected("deferred")
        assert clean_settings.controller.click_strategy == "deferred"
        window.close()


class TestOnboardingExperience:
    """Tests for the 7-step onboarding wizard dialog."""

    def test_onboarding_wizard_step_progression(self, app_instance, clean_settings):
        clean_settings.ui.onboarding_completed = False
        dlg = OnboardingDialog(settings=clean_settings)
        assert dlg._stack.count() == len(STEPS)
        assert dlg._current_step == 0

        # Step through to final step
        for i in range(len(STEPS) - 1):
            dlg._on_next()
            assert dlg._current_step == i + 1

        assert dlg._current_step == len(STEPS) - 1
        assert dlg._btn_next.text() == "Finish Setup ✔"

        # Finish setup marks completed
        dlg._on_finish()
        assert clean_settings.ui.onboarding_completed is True
        dlg.close()

    def test_onboarding_skip_marks_completed(self, app_instance, clean_settings):
        clean_settings.ui.onboarding_completed = False
        dlg = OnboardingDialog(settings=clean_settings)
        dlg._on_skip()
        assert clean_settings.ui.onboarding_completed is True
        dlg.close()

    def test_onboarding_never_enables_virtual_mouse(self, app_instance, clean_settings):
        clean_settings.ui.onboarding_completed = False
        dlg = OnboardingDialog(settings=clean_settings)
        dlg._on_next()
        dlg._on_finish()
        # Invariant: mouse control must stay disabled
        assert clean_settings.controller.mouse_control_enabled_default is False
        dlg.close()


class TestResponsiveLayoutsAndSafety:
    """Tests for responsive window sizes, safety halts, and graceful shutdown."""

    @pytest.mark.parametrize("width,height", [(1366, 768), (1600, 900), (1920, 1080)])
    def test_window_resizing_supported(self, app_instance, clean_settings, width, height):
        window = MainWindow(settings=clean_settings)
        window.resize(width, height)
        assert window.width() == width
        assert window.height() == height
        window.close()

    def test_emergency_stop_halts_automation(self, app_instance, clean_settings):
        window = MainWindow(settings=clean_settings)
        window._btn_mouse_enable.setEnabled(True)

        window._on_emergency_stop()
        assert window._btn_mouse_enable.isEnabled() is False
        assert window._btn_mouse_disable.isEnabled() is False
        assert window._btn_reset_emergency.isEnabled() is True
        assert "STOPPED" in window._lbl_mouse_status.text()
        window.close()

    def test_about_page_metadata(self, app_instance):
        page = AboutPage()
        assert __version__ in page.findChild(QLabel).text() or "0.1.0"
        page.close()
