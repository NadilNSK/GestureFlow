"""
gestureflow.ui.pages.dashboard_page
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Main home dashboard page featuring:
- Header with branding, status badges, and Emergency Stop
- Primary action buttons (Start/Stop, Enable/Disable Mouse, Calibrate, Precision)
- Status cards (FPS, Gesture, Confidence, Action, Profile)
- Embedded live camera preview with landmark HUD
- Recent activity log (bounded in-memory queue)
"""

from __future__ import annotations

import collections
import datetime
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
)

from gestureflow.config.settings import Settings
from gestureflow.ui.components.theme import (
    CARD_BG,
    CARD_BG_HOVER,
    BORDER_COLOR,
    ACCENT_EMERALD,
    ACCENT_CYAN,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_MUTED,
    DANGER_RED,
    WARNING_AMBER,
)
from gestureflow.ui.components.cards import StatusCard


class DashboardPage(QWidget):
    """
    Dashboard home page providing central control and system overview.
    """

    # Signals for parent window to connect
    start_requested = Signal()
    stop_requested = Signal()
    enable_mouse_requested = Signal()
    disable_mouse_requested = Signal()
    calibrate_requested = Signal()
    precision_toggled = Signal()
    emergency_stop_requested = Signal()

    def __init__(self, settings: Settings, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._activity_log: collections.deque[str] = collections.deque(maxlen=100)

        self._build_ui()

    def _build_ui(self) -> None:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(14)

        # --------------------------------------------------------------
        # 1. Header Bar: Title, Status Pills, Emergency Stop
        # --------------------------------------------------------------
        header_bar = QHBoxLayout()
        header_bar.setContentsMargins(0, 0, 0, 0)

        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        lbl_title = QLabel("GestureFlow Dashboard")
        lbl_title.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 20px; font-weight: 800;")
        lbl_subtitle = QLabel("AI Virtual Mouse • Real-Time Human-Computer Interface")
        lbl_subtitle.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px;")
        title_box.addWidget(lbl_title)
        title_box.addWidget(lbl_subtitle)
        header_bar.addLayout(title_box)

        header_bar.addStretch()

        # Status Pills
        self._pill_camera = QLabel("CAMERA: OFF")
        self._pill_camera.setStyleSheet(self._pill_style("idle"))
        header_bar.addWidget(self._pill_camera)

        self._pill_tracking = QLabel("TRACKING: IDLE")
        self._pill_tracking.setStyleSheet(self._pill_style("idle"))
        header_bar.addWidget(self._pill_tracking)

        self._pill_mouse = QLabel("MOUSE: DISABLED")
        self._pill_mouse.setStyleSheet(self._pill_style("idle"))
        header_bar.addWidget(self._pill_mouse)

        # Emergency Stop Button
        self._btn_emergency_stop = QPushButton("🛑 EMERGENCY STOP (Esc)")
        self._btn_emergency_stop.setObjectName("btn_danger")
        self._btn_emergency_stop.setFixedHeight(34)
        self._btn_emergency_stop.setStyleSheet(f"""
            QPushButton#btn_danger {{
                background-color: {DANGER_RED};
                color: #FFFFFF;
                border: none;
                font-weight: 800;
                font-size: 12px;
                padding: 6px 16px;
                border-radius: 6px;
            }}
            QPushButton#btn_danger:hover {{
                background-color: #DC2626;
            }}
        """)
        self._btn_emergency_stop.clicked.connect(self.emergency_stop_requested.emit)
        header_bar.addWidget(self._btn_emergency_stop)

        layout.addLayout(header_bar)

        # --------------------------------------------------------------
        # 2. Primary Control Strip
        # --------------------------------------------------------------
        ctrl_card = QFrame()
        ctrl_card.setStyleSheet(f"""
            QFrame {{
                background-color: {CARD_BG};
                border: 1px solid {BORDER_COLOR};
                border-radius: 8px;
                padding: 6px;
            }}
        """)
        ctrl_layout = QHBoxLayout(ctrl_card)
        ctrl_layout.setContentsMargins(12, 8, 12, 8)
        ctrl_layout.setSpacing(10)

        self._btn_start = QPushButton("▶  Start Tracking")
        self._btn_start.setObjectName("btn_primary")
        self._btn_start.setFixedHeight(34)
        self._btn_start.clicked.connect(self.start_requested.emit)
        ctrl_layout.addWidget(self._btn_start)

        self._btn_stop = QPushButton("■  Stop Tracking")
        self._btn_stop.setFixedHeight(34)
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self.stop_requested.emit)
        ctrl_layout.addWidget(self._btn_stop)

        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.VLine)
        sep1.setStyleSheet(f"background-color: {BORDER_COLOR}; max-width: 1px;")
        ctrl_layout.addWidget(sep1)

        self._btn_mouse_enable = QPushButton("🖱  Enable Mouse")
        self._btn_mouse_enable.setFixedHeight(34)
        self._btn_mouse_enable.setEnabled(False)
        self._btn_mouse_enable.clicked.connect(self.enable_mouse_requested.emit)
        ctrl_layout.addWidget(self._btn_mouse_enable)

        self._btn_mouse_disable = QPushButton("Disable Mouse")
        self._btn_mouse_disable.setFixedHeight(34)
        self._btn_mouse_disable.setEnabled(False)
        self._btn_mouse_disable.clicked.connect(self.disable_mouse_requested.emit)
        ctrl_layout.addWidget(self._btn_mouse_disable)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.VLine)
        sep2.setStyleSheet(f"background-color: {BORDER_COLOR}; max-width: 1px;")
        ctrl_layout.addWidget(sep2)

        self._btn_precision = QPushButton("🎯 Precision (Off)")
        self._btn_precision.setFixedHeight(34)
        self._btn_precision.setCheckable(True)
        self._btn_precision.clicked.connect(self.precision_toggled.emit)
        ctrl_layout.addWidget(self._btn_precision)

        self._btn_calibrate = QPushButton("🎯 Calibrate")
        self._btn_calibrate.setFixedHeight(34)
        self._btn_calibrate.clicked.connect(self.calibrate_requested.emit)
        ctrl_layout.addWidget(self._btn_calibrate)

        ctrl_layout.addStretch()
        layout.addWidget(ctrl_card)

        # --------------------------------------------------------------
        # 3. Status Cards Row (6 Metric Cards)
        # --------------------------------------------------------------
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(10)

        self._card_cam_fps = StatusCard("Camera FPS", "—", "Input stream", "📷", ACCENT_CYAN)
        self._card_proc_fps = StatusCard("Pipeline FPS", "—", "Processing rate", "⚡", ACCENT_EMERALD)
        self._card_gesture = StatusCard("Gesture", "NONE", "Active gesture", "✋", ACCENT_EMERALD)
        self._card_confidence = StatusCard("Confidence", "—", "Classification", "🎯", ACCENT_CYAN)
        self._card_action = StatusCard("Mouse Action", "IDLE", "State machine", "🖱", ACCENT_CYAN)
        self._card_profile = StatusCard("Profile", getattr(self._settings.controller, "cursor_profile", "Balanced"), "Response mode", "⚙", ACCENT_EMERALD)

        for card in (
            self._card_cam_fps,
            self._card_proc_fps,
            self._card_gesture,
            self._card_confidence,
            self._card_action,
            self._card_profile,
        ):
            cards_layout.addWidget(card)

        layout.addLayout(cards_layout)

        # --------------------------------------------------------------
        # 4. Main Body: Live Feed Preview (Left) + Activity & Telemetry (Right)
        # --------------------------------------------------------------
        content_layout = QHBoxLayout()
        content_layout.setSpacing(12)

        # Left: Camera Preview Container
        preview_box = QFrame()
        preview_box.setStyleSheet(f"""
            QFrame {{
                background-color: {CARD_BG};
                border: 1px solid {BORDER_COLOR};
                border-radius: 8px;
            }}
        """)
        preview_layout = QVBoxLayout(preview_box)
        preview_layout.setContentsMargins(10, 10, 10, 10)

        preview_header = QHBoxLayout()
        lbl_feed_title = QLabel("Live Camera HUD")
        lbl_feed_title.setStyleSheet(f"color: {TEXT_PRIMARY}; font-weight: 700; font-size: 13px;")
        preview_header.addWidget(lbl_feed_title)
        preview_header.addStretch()

        self._lbl_hud_info = QLabel("DirectShow Camera • 640×480")
        self._lbl_hud_info.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        preview_header.addWidget(self._lbl_hud_info)
        preview_layout.addLayout(preview_header)

        self._preview_label = QLabel()
        self._preview_label.setObjectName("preview_label")
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setMinimumSize(480, 360)
        self._preview_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._preview_label.setStyleSheet("""
            QLabel#preview_label {
                background-color: #05080E;
                border: 1px solid #1E2838;
                border-radius: 6px;
            }
        """)
        self._preview_label.setText(
            '<span style="color:#6B7280; font-size:14px; font-weight:500;">'
            '📷 Click "▶ Start Tracking" to initialize webcam stream</span>'
        )
        self._preview_label.setTextFormat(Qt.TextFormat.RichText)
        preview_layout.addWidget(self._preview_label)

        content_layout.addWidget(preview_box, stretch=3)

        # Right: Activity Log & Quick Telemetry
        right_box = QFrame()
        right_box.setStyleSheet(f"""
            QFrame {{
                background-color: {CARD_BG};
                border: 1px solid {BORDER_COLOR};
                border-radius: 8px;
            }}
        """)
        right_layout = QVBoxLayout(right_box)
        right_layout.setContentsMargins(12, 10, 12, 10)
        right_layout.setSpacing(8)

        lbl_log_title = QLabel("Recent Activity & Safety Events")
        lbl_log_title.setStyleSheet(f"color: {TEXT_PRIMARY}; font-weight: 700; font-size: 13px;")
        right_layout.addWidget(lbl_log_title)

        self._log_view = QTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.document().setMaximumBlockCount(200)
        self._log_view.setStyleSheet(f"""
            QTextEdit {{
                background-color: #070B12;
                color: {TEXT_SECONDARY};
                font-family: Consolas, 'Courier New', monospace;
                font-size: 11px;
                border: 1px solid {BORDER_COLOR};
                border-radius: 6px;
                padding: 6px;
            }}
        """)
        right_layout.addWidget(self._log_view)

        # Quick stats line at bottom of right panel
        self._lbl_cursor_coords = QLabel("Cursor: (—, —)")
        self._lbl_cursor_coords.setStyleSheet(f"color: {ACCENT_CYAN}; font-size: 11px; font-weight: 600;")
        right_layout.addWidget(self._lbl_cursor_coords)

        content_layout.addWidget(right_box, stretch=2)
        layout.addLayout(content_layout)

        scroll.setWidget(container)

        # Set central widget
        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.addWidget(scroll)

    def log_activity(self, message: str, level: str = "info") -> None:
        """Append an in-memory activity log event with timestamp."""
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        color = {
            "info": TEXT_SECONDARY,
            "success": ACCENT_EMERALD,
            "warning": WARNING_AMBER,
            "danger": DANGER_RED,
            "cyan": ACCENT_CYAN,
        }.get(level, TEXT_SECONDARY)

        html = f'<span style="color:{TEXT_MUTED}">[{ts}]</span> <span style="color:{color}">{message}</span>'
        self._activity_log.append(html)
        self._log_view.append(html)
        sb = self._log_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def update_telemetry(
        self,
        camera_fps: float,
        processing_fps: float,
        gesture_name: str,
        confidence: float,
        mouse_action: str,
        profile_name: str,
        cursor_coords: Optional[tuple[int, int]],
        mouse_state: str,
        tracking_quality: str,
    ) -> None:
        """Update dashboard status cards and telemetry pills."""
        self._card_cam_fps.set_value(f"{camera_fps:.1f}", ACCENT_CYAN)
        self._card_proc_fps.set_value(f"{processing_fps:.1f}", ACCENT_EMERALD)
        self._card_gesture.set_value(gesture_name.upper())
        self._card_confidence.set_value(f"{confidence * 100:.0f}%" if confidence > 0 else "—")
        self._card_action.set_value(mouse_action.upper())
        self._card_profile.set_value(profile_name)

        if cursor_coords:
            self._lbl_cursor_coords.setText(f"Cursor: ({cursor_coords[0]}, {cursor_coords[1]})")

        # Update Pills
        if mouse_state in ("MOVING", "DRAGGING", "SCROLLING", "IDLE", "CLICK_PENDING"):
            self._pill_mouse.setText(f"MOUSE: {mouse_state}")
            self._pill_mouse.setStyleSheet(self._pill_style("active"))
        elif mouse_state == "EMERGENCY_STOPPED":
            self._pill_mouse.setText("MOUSE: STOPPED")
            self._pill_mouse.setStyleSheet(self._pill_style("danger"))
        else:
            self._pill_mouse.setText("MOUSE: DISABLED")
            self._pill_mouse.setStyleSheet(self._pill_style("idle"))

        if tracking_quality in ("EXCELLENT", "GOOD"):
            self._pill_tracking.setText(f"TRACK: {tracking_quality}")
            self._pill_tracking.setStyleSheet(self._pill_style("active"))
        elif tracking_quality == "DEGRADED":
            self._pill_tracking.setText("TRACK: DEGRADED")
            self._pill_tracking.setStyleSheet(self._pill_style("warning"))
        else:
            self._pill_tracking.setText("TRACK: LOST")
            self._pill_tracking.setStyleSheet(self._pill_style("idle"))

    def set_camera_active(self, active: bool, device_text: str = "") -> None:
        if active:
            self._pill_camera.setText("CAMERA: ACTIVE")
            self._pill_camera.setStyleSheet(self._pill_style("active"))
            if device_text:
                self._lbl_hud_info.setText(device_text)
        else:
            self._pill_camera.setText("CAMERA: OFF")
            self._pill_camera.setStyleSheet(self._pill_style("idle"))

    def _pill_style(self, variant: str) -> str:
        if variant == "active":
            return f"""
                background-color: rgba(52, 211, 153, 0.15);
                color: {ACCENT_EMERALD};
                border: 1px solid {ACCENT_EMERALD};
                border-radius: 4px;
                padding: 4px 10px;
                font-size: 11px;
                font-weight: 700;
            """
        elif variant == "warning":
            return f"""
                background-color: rgba(245, 158, 11, 0.15);
                color: {WARNING_AMBER};
                border: 1px solid {WARNING_AMBER};
                border-radius: 4px;
                padding: 4px 10px;
                font-size: 11px;
                font-weight: 700;
            """
        elif variant == "danger":
            return f"""
                background-color: rgba(239, 68, 68, 0.2);
                color: {DANGER_RED};
                border: 1px solid {DANGER_RED};
                border-radius: 4px;
                padding: 4px 10px;
                font-size: 11px;
                font-weight: 700;
            """
        return f"""
            background-color: rgba(156, 163, 175, 0.1);
            color: {TEXT_MUTED};
            border: 1px solid {BORDER_COLOR};
            border-radius: 4px;
            padding: 4px 10px;
            font-size: 11px;
            font-weight: 700;
        """
