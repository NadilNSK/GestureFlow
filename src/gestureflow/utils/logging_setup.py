"""
gestureflow.utils.logging_setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Configures the root logger with:
  - Colour-coded ``StreamHandler`` for the console.
  - Rotating ``FileHandler`` for persistent log files.
  - Structured JSON option (future).

Call ``configure_logging(settings)`` once at startup.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gestureflow.config.settings import LoggingSettings

# ---------------------------------------------------------------------------
# ANSI colour helpers (Windows 10+ supports VT sequences)
# ---------------------------------------------------------------------------
_RESET = "\x1b[0m"
_LEVEL_COLOURS: dict[int, str] = {
    logging.DEBUG: "\x1b[36m",    # Cyan
    logging.INFO: "\x1b[32m",     # Green
    logging.WARNING: "\x1b[33m",  # Yellow
    logging.ERROR: "\x1b[31m",    # Red
    logging.CRITICAL: "\x1b[35m", # Magenta
}


class _ColouredFormatter(logging.Formatter):
    """Formatter that adds ANSI colour codes to level names."""

    _FMT = "%(asctime)s %(colour)s%(levelname)-8s%(reset)s %(name)s — %(message)s"
    _DATE_FMT = "%H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        colour = _LEVEL_COLOURS.get(record.levelno, "")
        record.colour = colour
        record.reset = _RESET if colour else ""
        formatter = logging.Formatter(self._FMT, datefmt=self._DATE_FMT)
        return formatter.format(record)


class _PlainFormatter(logging.Formatter):
    """Plain formatter for file output (no ANSI escapes)."""

    _FMT = "%(asctime)s %(levelname)-8s %(name)s — %(message)s"
    _DATE_FMT = "%Y-%m-%d %H:%M:%S"

    def __init__(self) -> None:
        super().__init__(fmt=self._FMT, datefmt=self._DATE_FMT)


def configure_logging(settings: "LoggingSettings") -> None:
    """
    Set up the application-wide logging configuration.

    Parameters
    ----------
    settings:
        The ``LoggingSettings`` dataclass from ``Settings.load()``.
    """
    root = logging.getLogger()
    numeric_level = logging.getLevelName(settings.level.upper())
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    root.setLevel(numeric_level)

    # Remove any existing handlers (e.g. from a previous call or test fixture).
    root.handlers.clear()

    # ------------------------------------------------------------------ #
    # Console handler                                                       #
    # ------------------------------------------------------------------ #
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    # Enable VT sequences on Windows so colours work.
    _enable_windows_vt()
    console_handler.setFormatter(_ColouredFormatter())
    root.addHandler(console_handler)

    # ------------------------------------------------------------------ #
    # Rotating file handler                                                 #
    # ------------------------------------------------------------------ #
    log_path = Path(settings.file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_path,
        maxBytes=settings.max_bytes,
        backupCount=settings.backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(_PlainFormatter())
    root.addHandler(file_handler)

    # Silence noisy third-party loggers.
    for noisy in ("PIL", "matplotlib", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "Logging initialised — level=%s file=%s", settings.level, log_path
    )


def _enable_windows_vt() -> None:
    """Enable VT-100 escape codes on Windows 10+ consoles."""
    try:
        import ctypes
        import ctypes.wintypes

        STD_OUTPUT_HANDLE = -11
        ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
        mode = ctypes.wintypes.DWORD()
        kernel32.GetConsoleMode(handle, ctypes.byref(mode))
        kernel32.SetConsoleMode(handle, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING)
    except Exception:
        pass  # Not on Windows or VT not supported — colours simply won't appear.
