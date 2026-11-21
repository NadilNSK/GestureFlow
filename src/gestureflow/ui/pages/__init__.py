"""
gestureflow.ui.pages
~~~~~~~~~~~~~~~~~~~~
Exports the 7 core pages of the GestureFlow application shell:
- DashboardPage
- LiveTrackingPage
- GestureControlsPage
- CalibrationPage
- PerformancePage
- SettingsPage
- AboutPage
"""

from gestureflow.ui.pages.dashboard_page import DashboardPage
from gestureflow.ui.pages.live_tracking_page import LiveTrackingPage
from gestureflow.ui.pages.gesture_controls_page import GestureControlsPage
from gestureflow.ui.pages.calibration_page import CalibrationPage
from gestureflow.ui.pages.performance_page import PerformancePage
from gestureflow.ui.pages.settings_page import SettingsPage
from gestureflow.ui.pages.about_page import AboutPage

__all__ = [
    "DashboardPage",
    "LiveTrackingPage",
    "GestureControlsPage",
    "CalibrationPage",
    "PerformancePage",
    "SettingsPage",
    "AboutPage",
]
