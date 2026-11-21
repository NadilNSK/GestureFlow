"""
gestureflow.ui.components.sidebar
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Collapsible left navigation sidebar with:
- GestureFlow branded header and logo badge
- Navigation items for the 7 application pages
- Active indicator pill and hover states
- Keyboard accessibility and tooltips
- Expand / collapse toggle for compact desktop layouts
"""

from __future__ import annotations

from typing import Optional
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFrame,
    QSizePolicy,
)

from gestureflow.ui.components.theme import (
    SIDEBAR_BG,
    BORDER_COLOR,
    ACCENT_EMERALD,
    ACCENT_CYAN,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_MUTED,
)

NAV_ITEMS = [
    ("📊", "Dashboard", "System overview, quick controls, and live HUD"),
    ("📷", "Live Tracking", "Expanded camera view and landmark inspection"),
    ("✋", "Gesture Controls", "Mouse actions, gesture cards, and triggers"),
    ("🎯", "Calibration", "4-point perspective coordinate calibration"),
    ("⚡", "Performance", "Real-time latency charts and pipeline diagnostics"),
    ("⚙", "Settings", "Configure camera, cursor, gestures, and safety"),
    ("ℹ", "About", "Version info, shortcuts, and setup guide"),
]


class _NavButton(QPushButton):
    """Individual navigation button with icon, text, and active indicator."""

    def __init__(
        self,
        index: int,
        icon: str,
        title: str,
        tooltip: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._index = index
        self._icon = icon
        self._title = title
        self._is_active = False
        self._collapsed = False

        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(tooltip)
        self.setFixedHeight(40)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._update_text()
        self._apply_style()

    def set_collapsed(self, collapsed: bool) -> None:
        self._collapsed = collapsed
        self._update_text()
        self.setFixedWidth(44 if collapsed else QWIDGETSIZE_MAX)

    def _update_text(self) -> None:
        if self._collapsed:
            self.setText(self._icon)
        else:
            self.setText(f"  {self._icon}   {self._title}")

    def set_active(self, active: bool) -> None:
        self._is_active = active
        self.setChecked(active)
        self._apply_style()

    def _apply_style(self) -> None:
        if self._is_active:
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: rgba(52, 211, 153, 0.15);
                    color: {ACCENT_EMERALD};
                    border: none;
                    border-left: 3px solid {ACCENT_EMERALD};
                    border-radius: 4px;
                    text-align: left;
                    padding-left: 10px;
                    font-size: 13px;
                    font-weight: 700;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: transparent;
                    color: {TEXT_SECONDARY};
                    border: none;
                    border-left: 3px solid transparent;
                    border-radius: 4px;
                    text-align: left;
                    padding-left: 10px;
                    font-size: 13px;
                    font-weight: 500;
                }}
                QPushButton:hover {{
                    background-color: rgba(255, 255, 255, 0.05);
                    color: {TEXT_PRIMARY};
                }}
            """)


class SidebarNavigation(QWidget):
    """
    Main left-hand application navigation sidebar.
    """

    page_selected = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._collapsed = False
        self._current_index = 0
        self._buttons: list[_NavButton] = []

        self.setObjectName("sidebar_nav")
        self.setMinimumWidth(210)
        self.setMaximumWidth(230)
        self.setStyleSheet(f"""
            QWidget#sidebar_nav {{
                background-color: {SIDEBAR_BG};
                border-right: 1px solid {BORDER_COLOR};
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 16, 10, 16)
        layout.setSpacing(6)

        # Header branding
        self._header_layout = QHBoxLayout()
        self._header_layout.setContentsMargins(4, 0, 4, 10)

        self._lbl_logo = QLabel("🖐")
        self._lbl_logo.setStyleSheet("font-size: 22px;")
        self._header_layout.addWidget(self._lbl_logo)

        self._brand_title = QLabel("GestureFlow")
        self._brand_title.setStyleSheet(f"""
            color: {TEXT_PRIMARY};
            font-size: 16px;
            font-weight: 800;
            letter-spacing: 0.5px;
        """)
        self._header_layout.addWidget(self._brand_title)
        self._header_layout.addStretch()

        # Collapse / Expand toggle
        self._btn_toggle = QPushButton("◀")
        self._btn_toggle.setFixedSize(26, 26)
        self._btn_toggle.setToolTip("Toggle sidebar width")
        self._btn_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_toggle.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                border: 1px solid {BORDER_COLOR};
                border-radius: 4px;
                color: {TEXT_MUTED};
                font-size: 10px;
            }}
            QPushButton:hover {{
                color: {TEXT_PRIMARY};
                background: rgba(255, 255, 255, 0.05);
            }}
        """)
        self._btn_toggle.clicked.connect(self.toggle_collapse)
        self._header_layout.addWidget(self._btn_toggle)

        layout.addLayout(self._header_layout)

        # Separator line
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"background-color: {BORDER_COLOR}; max-height: 1px;")
        layout.addWidget(sep)
        layout.addSpacing(6)

        # Nav Buttons
        for idx, (icon, title, tip) in enumerate(NAV_ITEMS):
            btn = _NavButton(idx, icon, title, tip, self)
            btn.clicked.connect(lambda checked, i=idx: self.select_page(i))
            layout.addWidget(btn)
            self._buttons.append(btn)

        layout.addStretch()

        # Bottom version / status footer
        self._footer_lbl = QLabel("v0.1.0 • Desktop")
        self._footer_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._footer_lbl.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        layout.addWidget(self._footer_lbl)

        # Default selection: Dashboard (index 0)
        self.select_page(0)

    def select_page(self, index: int) -> None:
        """Set the active page and emit page_selected."""
        if not (0 <= index < len(self._buttons)):
            return
        self._current_index = index
        for i, btn in enumerate(self._buttons):
            btn.set_active(i == index)
        self.page_selected.emit(index)

    def current_index(self) -> int:
        return self._current_index

    def toggle_collapse(self) -> None:
        """Toggle collapsed (icon-only) mode."""
        self._collapsed = not self._collapsed
        if self._collapsed:
            self.setMinimumWidth(64)
            self.setMaximumWidth(64)
            self._brand_title.hide()
            self._btn_toggle.setText("▶")
            self._footer_lbl.setText("v0.1")
        else:
            self.setMinimumWidth(210)
            self.setMaximumWidth(230)
            self._brand_title.show()
            self._btn_toggle.setText("◀")
            self._footer_lbl.setText("v0.1.0 • Desktop")

        for btn in self._buttons:
            btn.set_collapsed(self._collapsed)


QWIDGETSIZE_MAX = 16777215
