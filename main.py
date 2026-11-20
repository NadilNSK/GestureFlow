"""
GestureFlow — AI Virtual Mouse
Entry point for the application.
"""

import sys
import signal
import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure src/ is on the Python path when running from the project root.
# (PyInstaller bundles place the package directly on sys.path.)
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).parent / "src"
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from gestureflow.config.settings import Settings
from gestureflow.utils.logging_setup import configure_logging
from gestureflow.ui.app import GestureFlowApp


def _handle_sigint(signum, frame):  # noqa: ANN001
    """Allow Ctrl-C to close the app gracefully."""
    logging.getLogger(__name__).info("SIGINT received — shutting down.")
    # QApplication.quit() is safe to call from a signal handler on Windows.
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app:
        app.quit()


def main() -> int:
    """
    Application entry point.

    Returns
    -------
    int
        Exit code (0 = clean exit).
    """
    # ------------------------------------------------------------------ #
    # 1. Load settings                                                     #
    # ------------------------------------------------------------------ #
    settings = Settings.load()

    # ------------------------------------------------------------------ #
    # 2. Configure structured logging                                       #
    # ------------------------------------------------------------------ #
    configure_logging(settings.logging)

    log = logging.getLogger(__name__)
    log.info("GestureFlow starting up — version 0.1.0")
    log.debug("Settings: %s", settings)

    # ------------------------------------------------------------------ #
    # 3. Register SIGINT → graceful shutdown                               #
    # ------------------------------------------------------------------ #
    signal.signal(signal.SIGINT, _handle_sigint)

    # ------------------------------------------------------------------ #
    # 4. Launch PySide6 application                                         #
    # ------------------------------------------------------------------ #
    qt_app = GestureFlowApp(sys.argv, settings)
    exit_code = qt_app.run()

    log.info("GestureFlow exited cleanly with code %d.", exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
