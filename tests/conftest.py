"""
Pytest configuration and shared fixtures for the GestureFlow test suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure src/ is on sys.path for imports.
_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def default_settings(project_root):
    """Return a Settings instance loaded from the project defaults."""
    from gestureflow.config.settings import Settings
    return Settings.load(
        defaults_path=project_root / "default_settings.json",
    )
