"""
gestureflow.ui.pages.live_tracking_page
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Expanded camera view and computer vision inspector:
- Large high-resolution camera HUD with landmark overlays
- Camera device selection dropdown and scan/refresh action
- Visualization toggles (Landmarks, Hand Skeleton, Calibration Box)
- Detailed hand tracking telemetry panel (Handedness, Score, Finger states, Pinch ratio)
- Thread-safe frame delivery with zero main-thread processing
"""

from __future__ import annotations

from typing import Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QPixmap
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QCheckBox,
    QFrame,
    QScrollArea,
    QSizePolicy,
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
    DANGER_RED,
    WARNING_AMBER,
)
from gestureflow.ui.components.cards import MetricCard, SectionHeader


class LiveTrackingPage(QWidget):
    """
    Expanded Live Tracking page dedicated to computer vision and landmark inspection.
    """

    camera_changed = Signal(int)
    refresh_cameras_requested = Signal()
    start_requested = Signal()
    stop_requested = Signal()
    toggle_landmarks_changed = Signal(bool)
    toggle_skeleton_changed = Signal(bool)

    def __init__(self, settings: Settings, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._finger_labels: dict[str, QLabel] = {}

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        # --------------------------------------------------------------
        # Top Controls Bar: Camera selector, Refresh, Start/Stop, Toggles
        # --------------------------------------------------------------
        top_bar = QFrame()
        top_bar.setStyleSheet(f"""
            QFrame {{
                background-color: {CARD_BG};
                border: 1px solid {BORDER_COLOR};
                border-radius: 8px;
                padding: 4px;
            }}
        """)
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(12, 6, 12, 6)
        top_layout.setSpacing(10)

        lbl_cam = QLabel("Camera:")
        lbl_cam.setStyleSheet(f"color: {TEXT_SECONDARY}; font-weight: 700; font-size: 12px;")
        top_layout.addWidget(lbl_cam)

        self._combo_camera = QComboBox()
        self._combo_camera.setObjectName("combo_camera")
        self._combo_camera.setMinimumWidth(260)
        self._combo_camera.setFixedHeight(32)
        self._combo_camera.currentIndexChanged.connect(self.camera_changed.emit)
        top_layout.addWidget(self._combo_camera)

        self._btn_refresh = QPushButton("🔄 Refresh")
        self._btn_refresh.setObjectName("btn_refresh_cameras")
        self._btn_refresh.setFixedHeight(32)
        self._btn_refresh.clicked.connect(self.refresh_cameras_requested.emit)
        top_layout.addWidget(self._btn_refresh)

        top_layout.addSpacing(10)

        self._btn_start = QPushButton("▶ Start")
        self._btn_start.setObjectName("btn_primary")
        self._btn_start.setFixedHeight(32)
        self._btn_start.clicked.connect(self.start_requested.emit)
        top_layout.addWidget(self._btn_start)

        self._btn_stop = QPushButton("■ Stop")
        self._btn_stop.setFixedHeight(32)
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self.stop_requested.emit)
        top_layout.addWidget(self._btn_stop)

        top_layout.addStretch()

        # Visualization Toggles
        self._chk_landmarks = QCheckBox("Landmarks")
        self._chk_landmarks.setChecked(getattr(self._settings.ui, "show_landmarks", True))
        self._chk_landmarks.toggled.connect(self.toggle_landmarks_changed.emit)
        top_layout.addWidget(self._chk_landmarks)

        self._chk_skeleton = QCheckBox("Skeleton")
        self._chk_skeleton.setChecked(getattr(self._settings.ui, "show_skeleton", True))
        self._chk_skeleton.toggled.connect(self.toggle_skeleton_changed.emit)
        top_layout.addWidget(self._chk_skeleton)

        layout.addWidget(top_bar)

        # --------------------------------------------------------------
        # Main Split: Expanded Preview (Left) + Detailed Telemetry (Right)
        # --------------------------------------------------------------
        main_body = QHBoxLayout()
        main_body.setSpacing(14)

        # Left: Large Preview
        preview_frame = QFrame()
        preview_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {CARD_BG};
                border: 1px solid {BORDER_COLOR};
                border-radius: 8px;
            }}
        """)
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.setContentsMargins(10, 10, 10, 10)

        self._preview_label = QLabel()
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._preview_label.setMinimumSize(540, 400)
        self._preview_label.setStyleSheet("""
            QLabel {
                background-color: #05080E;
                border: 1px solid #1E2838;
                border-radius: 6px;
            }
        """)
        self._preview_label.setText(
            '<span style="color:#6B7280; font-size:14px;">📷 Camera stream inactive</span>'
        )
        self._preview_label.setTextFormat(Qt.TextFormat.RichText)
        preview_layout.addWidget(self._preview_label)

        main_body.addWidget(preview_frame, stretch=4)

        # Right: Hand & Landmark Telemetry Card
        side_panel = QFrame()
        side_panel.setMinimumWidth(280)
        side_panel.setMaximumWidth(340)
        side_panel.setStyleSheet(f"""
            QFrame {{
                background-color: {CARD_BG};
                border: 1px solid {BORDER_COLOR};
                border-radius: 8px;
            }}
        """)
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(14, 12, 14, 12)
        side_layout.setSpacing(10)

        hdr = SectionHeader("Tracking Inspector", "MediaPipe 21 Hand Landmarks")
        side_layout.addWidget(hdr)

        # Active Gesture Badge
        self._badge_gesture = QLabel("NO GESTURE")
        self._badge_gesture.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._badge_gesture.setStyleSheet(self._gesture_badge_style("idle"))
        side_layout.addWidget(self._badge_gesture)

        # Hand Telemetry Metrics
        self._metric_hands = MetricCard("Hands Detected", "0")
        self._metric_active_hand = MetricCard("Active Hand", "None")
        self._metric_quality = MetricCard("Tracking Quality", "LOST")
        self._metric_confidence = MetricCard("Gesture Confidence", "—")
        self._metric_pinch = MetricCard("Pinch Ratio", "—")
        self._metric_action = MetricCard("Mouse Action", "IDLE")

        for m in (
            self._metric_hands,
            self._metric_active_hand,
            self._metric_quality,
            self._metric_confidence,
            self._metric_pinch,
            self._metric_action,
        ):
            side_layout.addWidget(m)

        # Finger State Knuckles
        finger_box = QFrame()
        finger_box.setStyleSheet(f"""
            QFrame {{
                background-color: rgba(11, 16, 24, 0.6);
                border: 1px solid {BORDER_COLOR};
                border-radius: 6px;
                padding: 6px;
            }}
        """)
        f_layout = QVBoxLayout(finger_box)
        f_layout.setContentsMargins(8, 6, 8, 6)
        f_layout.setSpacing(4)

        lbl_f_title = QLabel("FINGER EXTENSION STATES")
        lbl_f_title.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 10px; font-weight: 700;")
        f_layout.addWidget(lbl_f_title)

        for fname in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            lbl_name = QLabel(f"{fname}:")
            lbl_name.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px;")
            lbl_val = QLabel("FOLDED")
            lbl_val.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; font-weight: 700;")
            row.addWidget(lbl_name)
            row.addStretch()
            row.addWidget(lbl_val)
            f_layout.addLayout(row)
            self._finger_labels[fname] = lbl_val

        side_layout.addWidget(finger_box)
        side_layout.addStretch()

        # Calibration Box Info Note
        self._lbl_calib_info = QLabel("Target Tracking Zone: 15% Margin")
        self._lbl_calib_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_calib_info.setStyleSheet(f"color: {ACCENT_CYAN}; font-size: 11px; font-weight: 600;")
        side_layout.addWidget(self._lbl_calib_info)

        main_body.addWidget(side_panel, stretch=1)
        layout.addLayout(main_body)

    def update_frame(self, pixmap: QPixmap) -> None:
        """Update live camera preview with scaled pixmap."""
        scaled = pixmap.scaled(
            self._preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        self._preview_label.setPixmap(scaled)

    def update_inspector(
        self,
        gesture_name: str,
        is_stable: bool,
        confidence: float,
        num_hands: int,
        active_hand: str,
        quality: str,
        pinch_ratio: float,
        action_name: str,
        finger_states: Optional[dict[str, bool]] = None,
    ) -> None:
        """Update detailed tracking inspector readouts."""
        # Update Gesture Badge
        state_tag = " (CONFIRMED)" if is_stable and gesture_name != "UNKNOWN" else ""
        self._badge_gesture.setText(f"{gesture_name.upper()}{state_tag}")
        self._badge_gesture.setStyleSheet(self._gesture_badge_style(gesture_name))

        self._metric_hands.set_value(str(num_hands))
        self._metric_active_hand.set_value(active_hand)

        # Quality Color
        q_col = (
            ACCENT_EMERALD if quality == "EXCELLENT"
            else (ACCENT_CYAN if quality == "GOOD"
            else (WARNING_AMBER if quality == "DEGRADED"
            else DANGER_RED))
        )
        self._metric_quality.set_value(quality, q_col)
        self._metric_confidence.set_value(f"{confidence * 100:.0f}%" if confidence > 0 else "—")
        self._metric_pinch.set_value(f"{pinch_ratio:.3f}" if pinch_ratio > 0 else "—")
        self._metric_action.set_value(action_name.upper())

        # Fingers
        if finger_states:
            for fname, is_ext in finger_states.items():
                lbl = self._finger_labels.get(fname)
                if lbl:
                    if is_ext:
                        lbl.setText("EXTENDED")
                        lbl.setStyleSheet(f"color: {ACCENT_EMERALD}; font-size: 11px; font-weight: 700;")
                    else:
                        lbl.setText("FOLDED")
                        lbl.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; font-weight: 500;")
        else:
            for lbl in self._finger_labels.values():
                lbl.setText("—")
                lbl.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")

    def _gesture_badge_style(self, gesture_name: str) -> str:
        g = gesture_name.upper()
        if "POINTING" in g:
            return f"background: rgba(52, 211, 153, 0.2); color: {ACCENT_EMERALD}; font-weight: 800; font-size: 14px; padding: 8px; border-radius: 6px; border: 1px solid {ACCENT_EMERALD};"
        elif "PINCH" in g:
            return f"background: rgba(34, 211, 238, 0.2); color: {ACCENT_CYAN}; font-weight: 800; font-size: 14px; padding: 8px; border-radius: 6px; border: 1px solid {ACCENT_CYAN};"
        elif "TWO_FINGER" in g:
            return f"background: rgba(59, 130, 246, 0.2); color: #60A5FA; font-weight: 800; font-size: 14px; padding: 8px; border-radius: 6px; border: 1px solid #3B82F6;"
        elif "OPEN_PALM" in g:
            return f"background: rgba(236, 72, 153, 0.2); color: #F472B6; font-weight: 800; font-size: 14px; padding: 8px; border-radius: 6px; border: 1px solid #EC4899;"
        elif "FIST" in g:
            return f"background: rgba(245, 158, 11, 0.2); color: {WARNING_AMBER}; font-weight: 800; font-size: 14px; padding: 8px; border-radius: 6px; border: 1px solid {WARNING_AMBER};"
        return f"background: rgba(156, 163, 175, 0.1); color: {TEXT_MUTED}; font-weight: 700; font-size: 14px; padding: 8px; border-radius: 6px; border: 1px solid {BORDER_COLOR};"
