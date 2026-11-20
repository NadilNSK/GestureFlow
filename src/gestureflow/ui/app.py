"""
gestureflow.ui.app
~~~~~~~~~~~~~~~~~~~
Thin wrapper around QApplication that wires up settings and launches
the main window.
"""

from __future__ import annotations

import logging
import sys
from typing import Sequence

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from gestureflow.config.settings import Settings
from gestureflow.ui.main_window import MainWindow

log = logging.getLogger(__name__)


class GestureFlowApp:
    """
    Application host.

    Parameters
    ----------
    argv:
        Command-line arguments (typically ``sys.argv``).
    settings:
        Loaded application settings.
    """

    def __init__(self, argv: Sequence[str], settings: Settings) -> None:
        self._settings = settings

        # High-DPI support (PySide6 6.x enables it by default, but be explicit).
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
        self._qapp = QApplication(list(argv))
        self._qapp.setApplicationName("GestureFlow")
        self._qapp.setApplicationVersion("0.1.0")
        self._qapp.setOrganizationName("GestureFlow")

        # Apply a clean default font.
        font = QFont("Segoe UI", 10)
        self._qapp.setFont(font)

        # Configure PyAutoGUI safety settings NOW, before any gesture code runs.
        self._configure_pyautogui()

        self._window = MainWindow(settings=settings)

    def run(self) -> int:
        """Show the window and start the Qt event loop."""
        self._window.show()
        log.info("Qt event loop starting.")
        return self._qapp.exec()

    def _configure_pyautogui(self) -> None:
        """Apply PyAutoGUI safety settings from config."""
        try:
            import pyautogui
            # Always enforce fail-safe regardless of config.
            pyautogui.FAILSAFE = True
            pyautogui.PAUSE = self._settings.pyautogui.pause
            log.info(
                "PyAutoGUI configured — FAILSAFE=True  PAUSE=%.3f",
                self._settings.pyautogui.pause,
            )
        except ImportError:
            log.warning("PyAutoGUI is not installed — mouse control will be unavailable.")
