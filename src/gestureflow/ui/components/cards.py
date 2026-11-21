"""
gestureflow.ui.components.cards
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Reusable card components conforming to the GestureFlow design system:
- StatusCard: High-level metric or status display
- GestureCard: Visual card for individual mouse gesture configuration & recognition feedback
- MetricCard: Compact key-value telemetry card
- SectionHeader: Consistent section heading with title, subtitle, and action slot
"""

from __future__ import annotations

from typing import Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QFrame,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QCheckBox,
    QSlider,
    QPushButton,
    QSizePolicy,
)

from gestureflow.ui.components.theme import (
    CARD_BG,
    CARD_BG_HOVER,
    BORDER_COLOR,
    BORDER_COLOR_LIGHT,
    ACCENT_EMERALD,
    ACCENT_CYAN,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_MUTED,
    DANGER_RED,
    WARNING_AMBER,
)


class StatusCard(QFrame):
    """
    Card displaying a primary metric or status value with title and icon.
    """

    def __init__(
        self,
        title: str,
        initial_value: str = "—",
        subtitle: str = "",
        icon: str = "",
        accent_color: str = ACCENT_EMERALD,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("status_card")
        self._accent_color = accent_color

        self.setStyleSheet(f"""
            QFrame#status_card {{
                background-color: {CARD_BG};
                border: 1px solid {BORDER_COLOR};
                border-radius: 8px;
                padding: 10px;
            }}
            QFrame#status_card:hover {{
                border-color: {BORDER_COLOR_LIGHT};
                background-color: {CARD_BG_HOVER};
            }}
        """)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)

        # Header row: Title and Icon
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)

        self._lbl_title = QLabel(title.upper())
        self._lbl_title.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; font-weight: 700; letter-spacing: 0.5px;")
        header_row.addWidget(self._lbl_title)
        header_row.addStretch()

        if icon:
            self._lbl_icon = QLabel(icon)
            self._lbl_icon.setStyleSheet(f"font-size: 16px; color: {accent_color};")
            header_row.addWidget(self._lbl_icon)
        else:
            self._lbl_icon = None

        layout.addLayout(header_row)

        # Value
        self._lbl_value = QLabel(initial_value)
        self._lbl_value.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 20px; font-weight: 700;")
        layout.addWidget(self._lbl_value)

        # Subtitle / status note
        self._lbl_subtitle = QLabel(subtitle)
        self._lbl_subtitle.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px;")
        if not subtitle:
            self._lbl_subtitle.hide()
        layout.addWidget(self._lbl_subtitle)

    def set_value(self, val: str, color: Optional[str] = None) -> None:
        """Update displayed value string and optional color."""
        self._lbl_value.setText(val)
        if color:
            self._lbl_value.setStyleSheet(f"color: {color}; font-size: 20px; font-weight: 700;")

    def set_subtitle(self, text: str, color: Optional[str] = None) -> None:
        """Update subtitle text."""
        self._lbl_subtitle.setText(text)
        if text:
            self._lbl_subtitle.show()
            if color:
                self._lbl_subtitle.setStyleSheet(f"color: {color}; font-size: 11px;")
        else:
            self._lbl_subtitle.hide()


class MetricCard(QFrame):
    """
    Compact horizontal key-value card with optional indicator badge.
    """

    def __init__(
        self,
        label: str,
        value: str = "—",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {CARD_BG};
                border: 1px solid {BORDER_COLOR};
                border-radius: 6px;
                padding: 6px 10px;
            }}
        """)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)

        self._lbl_label = QLabel(label)
        self._lbl_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px; font-weight: 500;")
        layout.addWidget(self._lbl_label)
        layout.addStretch()

        self._lbl_value = QLabel(value)
        self._lbl_value.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 12px; font-weight: 700;")
        layout.addWidget(self._lbl_value)

    def set_value(self, text: str, color: Optional[str] = None) -> None:
        self._lbl_value.setText(text)
        if color:
            self._lbl_value.setStyleSheet(f"color: {color}; font-size: 12px; font-weight: 700;")


class SectionHeader(QWidget):
    """
    Consistent section header with title, subtitle, and an optional action slot.
    """

    def __init__(
        self,
        title: str,
        subtitle: str = "",
        action_button: Optional[QPushButton] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 8)

        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(2)

        self._lbl_title = QLabel(title)
        self._lbl_title.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 16px; font-weight: 700;")
        text_layout.addWidget(self._lbl_title)

        if subtitle:
            self._lbl_subtitle = QLabel(subtitle)
            self._lbl_subtitle.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px;")
            text_layout.addWidget(self._lbl_subtitle)
        else:
            self._lbl_subtitle = None

        layout.addLayout(text_layout)
        layout.addStretch()

        if action_button:
            layout.addWidget(action_button)


class GestureCard(QFrame):
    """
    Card displaying a mouse gesture, its trigger, icon, enable toggle, and status.
    """

    toggled = Signal(bool)

    def __init__(
        self,
        gesture_name: str,
        action_description: str,
        trigger_gesture: str,
        icon: str = "✋",
        is_enabled: bool = True,
        show_toggle: bool = True,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("gesture_card")
        self._gesture_name = gesture_name

        self.setStyleSheet(f"""
            QFrame#gesture_card {{
                background-color: {CARD_BG};
                border: 1px solid {BORDER_COLOR};
                border-radius: 8px;
            }}
            QFrame#gesture_card:hover {{
                border-color: {BORDER_COLOR_LIGHT};
            }}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        # Top row: Icon, Name, and Status Badge
        top_row = QHBoxLayout()
        self._lbl_icon = QLabel(icon)
        self._lbl_icon.setStyleSheet(f"font-size: 24px; padding-right: 6px;")
        top_row.addWidget(self._lbl_icon)

        title_box = QVBoxLayout()
        title_box.setSpacing(1)
        self._lbl_title = QLabel(gesture_name)
        self._lbl_title.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 14px; font-weight: 700;")
        self._lbl_trigger = QLabel(f"Trigger: {trigger_gesture}")
        self._lbl_trigger.setStyleSheet(f"color: {ACCENT_CYAN}; font-size: 11px; font-weight: 600;")
        title_box.addWidget(self._lbl_title)
        title_box.addWidget(self._lbl_trigger)
        top_row.addLayout(title_box)

        top_row.addStretch()

        # Status badge
        self._lbl_status_badge = QLabel("IDLE")
        self._lbl_status_badge.setStyleSheet(f"""
            background-color: rgba(156, 163, 175, 0.15);
            color: {TEXT_SECONDARY};
            border-radius: 4px;
            padding: 3px 8px;
            font-size: 10px;
            font-weight: 700;
        """)
        top_row.addWidget(self._lbl_status_badge)
        layout.addLayout(top_row)

        # Description
        self._lbl_desc = QLabel(action_description)
        self._lbl_desc.setWordWrap(True)
        self._lbl_desc.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px;")
        layout.addWidget(self._lbl_desc)

        # Bottom row: Enable Checkbox & optional controls slot
        bottom_row = QHBoxLayout()
        if show_toggle:
            self._chk_enable = QCheckBox("Enable Action")
            self._chk_enable.setChecked(is_enabled)
            self._chk_enable.toggled.connect(self.toggled.emit)
            bottom_row.addWidget(self._chk_enable)
        else:
            self._chk_enable = None

        bottom_row.addStretch()
        self._custom_control_layout = QHBoxLayout()
        bottom_row.addLayout(self._custom_control_layout)
        layout.addLayout(bottom_row)

    @property
    def is_action_enabled(self) -> bool:
        return self._chk_enable.isChecked() if self._chk_enable else True

    def set_action_enabled(self, enabled: bool) -> None:
        if self._chk_enable:
            self._chk_enable.blockSignals(True)
            self._chk_enable.setChecked(enabled)
            self._chk_enable.blockSignals(False)

    def set_status(self, status: str, state_type: str = "idle") -> None:
        """Update status badge text and color based on state_type ('idle', 'active', 'triggered')."""
        self._lbl_status_badge.setText(status.upper())
        if state_type == "active":
            self._lbl_status_badge.setStyleSheet(f"""
                background-color: rgba(52, 211, 153, 0.2);
                color: {ACCENT_EMERALD};
                border: 1px solid {ACCENT_EMERALD};
                border-radius: 4px;
                padding: 3px 8px;
                font-size: 10px;
                font-weight: 700;
            """)
        elif state_type == "triggered":
            self._lbl_status_badge.setStyleSheet(f"""
                background-color: rgba(34, 211, 238, 0.25);
                color: {ACCENT_CYAN};
                border: 1px solid {ACCENT_CYAN};
                border-radius: 4px;
                padding: 3px 8px;
                font-size: 10px;
                font-weight: 700;
            """)
        elif state_type == "warning":
            self._lbl_status_badge.setStyleSheet(f"""
                background-color: rgba(245, 158, 11, 0.2);
                color: {WARNING_AMBER};
                border: 1px solid {WARNING_AMBER};
                border-radius: 4px;
                padding: 3px 8px;
                font-size: 10px;
                font-weight: 700;
            """)
        else:
            self._lbl_status_badge.setStyleSheet(f"""
                background-color: rgba(156, 163, 175, 0.15);
                color: {TEXT_SECONDARY};
                border-radius: 4px;
                padding: 3px 8px;
                font-size: 10px;
                font-weight: 700;
            """)
