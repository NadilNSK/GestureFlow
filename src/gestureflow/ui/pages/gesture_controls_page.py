"""
gestureflow.ui.pages.gesture_controls_page
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Dedicated gesture configuration and tuning page displaying visual cards for:
1. Cursor Movement (POINTING)
2. Left Click (Short Thumb-Index Pinch)
3. Double Click (Two Rapid Thumb-Index Pinches)
4. Right Click (Thumb-Middle Pinch)
5. Drag and Drop (Sustained Thumb-Index Pinch)
6. Vertical Scrolling (TWO_FINGER Hand Motion)
"""

from __future__ import annotations

from typing import Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QSlider,
    QScrollArea,
    QFrame,
)

from gestureflow.config.settings import Settings
from gestureflow.ui.components.theme import (
    CARD_BG,
    BORDER_COLOR,
    ACCENT_EMERALD,
    ACCENT_CYAN,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_MUTED,
)
from gestureflow.ui.components.cards import GestureCard, SectionHeader


class GestureControlsPage(QWidget):
    """
    Gesture Controls page providing visual tuning and toggles for each mouse gesture.
    """

    action_toggled = Signal(str, bool)  # ("left_click", True)
    drag_hold_changed = Signal(float)
    double_click_interval_changed = Signal(float)
    scroll_sensitivity_changed = Signal(float)
    tracking_margin_changed = Signal(float)

    def __init__(self, settings: Settings, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._settings = settings

        self._build_ui()

    def _build_ui(self) -> None:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(14)

        header = SectionHeader(
            "Gesture Mouse Controls",
            "Configure, enable/disable, and tune sensitivity for each computer vision mouse action",
        )
        layout.addWidget(header)

        # 2x3 Responsive Grid of Gesture Cards
        grid = QGridLayout()
        grid.setSpacing(14)

        # 1. Cursor Movement Card (POINTING)
        self._card_pointing = GestureCard(
            gesture_name="Cursor Movement",
            action_description="Extend index finger while folding remaining fingers. Moves OS cursor smoothly across your display with adaptive filtering.",
            trigger_gesture="POINTING (Index Tip)",
            icon="👆",
            is_enabled=True,
            show_toggle=False,
        )
        margin_row = QHBoxLayout()
        self._lbl_margin = QLabel(f"Tracking Margin: {int(self._settings.controller.tracking_region_margin * 100)}%")
        self._lbl_margin.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        self._slider_margin = QSlider(Qt.Orientation.Horizontal)
        self._slider_margin.setRange(5, 35)
        self._slider_margin.setValue(int(self._settings.controller.tracking_region_margin * 100))
        self._slider_margin.valueChanged.connect(self._on_margin_changed)
        margin_row.addWidget(self._lbl_margin)
        margin_row.addWidget(self._slider_margin)
        self._card_pointing._custom_control_layout.addLayout(margin_row)
        grid.addWidget(self._card_pointing, 0, 0)

        # 2. Left Click Card (Short Pinch)
        self._card_left_click = GestureCard(
            gesture_name="Left Click",
            action_description="Bring thumb tip and index fingertip together for a quick touch (< 450ms). Dispatches a single left mouse click on release.",
            trigger_gesture="LEFT_PINCH (Thumb + Index)",
            icon="👌",
            is_enabled=self._settings.controller.left_click_enabled,
            show_toggle=True,
        )
        self._card_left_click.toggled.connect(lambda v: self.action_toggled.emit("left_click", v))
        grid.addWidget(self._card_left_click, 0, 1)

        # 3. Double Click Card (Rapid Pinch)
        self._card_double_click = GestureCard(
            gesture_name="Double Click",
            action_description="Perform two quick consecutive pinches between thumb and index finger within the double-click window to open files or folders.",
            trigger_gesture="2x PINCH (< 350ms)",
            icon="✌️",
            is_enabled=self._settings.controller.double_click_enabled,
            show_toggle=True,
        )
        self._card_double_click.toggled.connect(lambda v: self.action_toggled.emit("double_click", v))

        dc_row = QHBoxLayout()
        self._lbl_dc_val = QLabel(f"Window: {self._settings.controller.double_click_interval:.2f}s")
        self._lbl_dc_val.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        self._slider_dc = QSlider(Qt.Orientation.Horizontal)
        self._slider_dc.setRange(15, 80)
        self._slider_dc.setValue(int(self._settings.controller.double_click_interval * 100))
        self._slider_dc.valueChanged.connect(self._on_double_click_changed)
        dc_row.addWidget(self._lbl_dc_val)
        dc_row.addWidget(self._slider_dc)
        self._card_double_click._custom_control_layout.addLayout(dc_row)
        grid.addWidget(self._card_double_click, 1, 0)

        # 4. Right Click Card (Thumb-Middle Pinch)
        self._card_right_click = GestureCard(
            gesture_name="Right Click (Context Menu)",
            action_description="Pinch thumb tip and middle finger tip together. Dispatches an immediate right-click to open Windows context menus.",
            trigger_gesture="RIGHT_PINCH (Thumb + Middle)",
            icon="🤏",
            is_enabled=self._settings.controller.right_click_enabled,
            show_toggle=True,
        )
        self._card_right_click.toggled.connect(lambda v: self.action_toggled.emit("right_click", v))
        grid.addWidget(self._card_right_click, 1, 1)

        # 5. Drag and Drop Card (Sustained Pinch)
        self._card_drag = GestureCard(
            gesture_name="Drag and Drop",
            action_description="Pinch thumb and index together and hold for sustained duration. Virtual mouse latches mouseDown until fingers are opened.",
            trigger_gesture="SUSTAINED PINCH (> 450ms)",
            icon="✊",
            is_enabled=self._settings.controller.drag_enabled,
            show_toggle=True,
        )
        self._card_drag.toggled.connect(lambda v: self.action_toggled.emit("drag", v))

        drag_row = QHBoxLayout()
        self._lbl_drag_val = QLabel(f"Hold: {self._settings.controller.drag_hold_duration:.2f}s")
        self._lbl_drag_val.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        self._slider_drag = QSlider(Qt.Orientation.Horizontal)
        self._slider_drag.setRange(20, 100)
        self._slider_drag.setValue(int(self._settings.controller.drag_hold_duration * 100))
        self._slider_drag.valueChanged.connect(self._on_drag_hold_changed)
        drag_row.addWidget(self._lbl_drag_val)
        drag_row.addWidget(self._slider_drag)
        self._card_drag._custom_control_layout.addLayout(drag_row)
        grid.addWidget(self._card_drag, 2, 0)

        # 6. Vertical Scrolling Card (TWO_FINGER)
        self._card_scroll = GestureCard(
            gesture_name="Vertical Scrolling",
            action_description="Raise both index and middle fingers side-by-side. Moving your hand up scrolls up; moving hand down scrolls down smoothly.",
            trigger_gesture="TWO_FINGER (Index + Middle)",
            icon="↕️",
            is_enabled=self._settings.controller.scroll_enabled,
            show_toggle=True,
        )
        self._card_scroll.toggled.connect(lambda v: self.action_toggled.emit("scroll", v))

        scroll_row = QHBoxLayout()
        self._lbl_scroll_val = QLabel(f"Sens: {self._settings.controller.scroll_sensitivity:.1f}x")
        self._lbl_scroll_val.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        self._slider_scroll = QSlider(Qt.Orientation.Horizontal)
        self._slider_scroll.setRange(2, 30)
        self._slider_scroll.setValue(int(self._settings.controller.scroll_sensitivity * 10))
        self._slider_scroll.valueChanged.connect(self._on_scroll_sens_changed)
        scroll_row.addWidget(self._lbl_scroll_val)
        scroll_row.addWidget(self._slider_scroll)
        self._card_scroll._custom_control_layout.addLayout(scroll_row)
        grid.addWidget(self._card_scroll, 2, 1)

        layout.addLayout(grid)
        layout.addStretch()

        scroll.setWidget(container)

        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.addWidget(scroll)

    def _on_margin_changed(self, value: int) -> None:
        m = value / 100.0
        self._lbl_margin.setText(f"Tracking Margin: {value}%")
        self._settings.controller.tracking_region_margin = m
        self.tracking_margin_changed.emit(m)

    def _on_drag_hold_changed(self, value: int) -> None:
        val = value / 100.0
        self._lbl_drag_val.setText(f"Hold: {val:.2f}s")
        self._settings.controller.drag_hold_duration = val
        self.drag_hold_changed.emit(val)

    def _on_double_click_changed(self, value: int) -> None:
        val = value / 100.0
        self._lbl_dc_val.setText(f"Window: {val:.2f}s")
        self._settings.controller.double_click_interval = val
        self.double_click_interval_changed.emit(val)

    def _on_scroll_sens_changed(self, value: int) -> None:
        val = value / 10.0
        self._lbl_scroll_val.setText(f"Sens: {val:.1f}x")
        self._settings.controller.scroll_sensitivity = val
        self.scroll_sensitivity_changed.emit(val)

    def update_live_gestures(self, gesture_name: str, action_state: str) -> None:
        """Update live status badges on each gesture card in real time."""
        g = gesture_name.upper()
        act = action_state.upper()

        # Reset all cards to idle
        self._card_pointing.set_status("IDLE", "idle")
        self._card_left_click.set_status("IDLE", "idle")
        self._card_double_click.set_status("IDLE", "idle")
        self._card_right_click.set_status("IDLE", "idle")
        self._card_drag.set_status("IDLE", "idle")
        self._card_scroll.set_status("IDLE", "idle")

        if g == "POINTING":
            self._card_pointing.set_status("TRACKING", "active")
        elif g == "LEFT_PINCH":
            if act == "DRAGGING":
                self._card_drag.set_status("DRAGGING ✊", "triggered")
            else:
                self._card_left_click.set_status("PINCHING", "active")
        elif g == "RIGHT_PINCH":
            self._card_right_click.set_status("CLICKING", "triggered")
        elif g == "TWO_FINGER":
            self._card_scroll.set_status("SCROLLING ↕", "active")
