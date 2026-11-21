"""
gestureflow.ui.pages.calibration_page
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Screen calibration dashboard and interactive guidance:
- Active calibration status banner and matrix summary
- Step-by-step visual instructions for 4-point corner calibration
- "Start 4-Point Calibration" launcher (modal overlay)
- "Reset Calibration" with safety confirmation dialog
- Explanation of coordinate homography & tracking zones
"""

from __future__ import annotations

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
    QMessageBox,
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
from gestureflow.ui.components.cards import SectionHeader, MetricCard


class CalibrationPage(QWidget):
    """
    Dedicated Calibration Page for four-point perspective calibration.
    """

    start_calibration_requested = Signal()
    reset_calibration_requested = Signal()

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
        layout.setSpacing(16)

        hdr = SectionHeader(
            "Screen Perspective Calibration",
            "Map your natural hand pointing angles directly to your Windows desktop display coordinates",
        )
        layout.addWidget(hdr)

        # --------------------------------------------------------------
        # 1. Status Banner
        # --------------------------------------------------------------
        self._banner = QFrame()
        self._banner_layout = QHBoxLayout(self._banner)
        self._banner_layout.setContentsMargins(16, 14, 16, 14)

        self._lbl_status_icon = QLabel("🎯")
        self._lbl_status_icon.setStyleSheet("font-size: 28px;")
        self._banner_layout.addWidget(self._lbl_status_icon)

        status_text_box = QVBoxLayout()
        status_text_box.setSpacing(2)
        self._lbl_banner_title = QLabel("Calibration Status: Ready")
        self._lbl_banner_title.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 15px; font-weight: 800;")
        self._lbl_banner_sub = QLabel("Using default 15% viewport bounding box")
        self._lbl_banner_sub.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px;")
        status_text_box.addWidget(self._lbl_banner_title)
        status_text_box.addWidget(self._lbl_banner_sub)
        self._banner_layout.addLayout(status_text_box)
        self._banner_layout.addStretch()

        self._btn_start_calib = QPushButton("Start 4-Point Calibration ▶")
        self._btn_start_calib.setObjectName("btn_primary")
        self._btn_start_calib.setFixedHeight(36)
        self._btn_start_calib.clicked.connect(self.start_calibration_requested.emit)
        self._banner_layout.addWidget(self._btn_start_calib)

        self._btn_reset_calib = QPushButton("Reset to Defaults")
        self._btn_reset_calib.setFixedHeight(36)
        self._btn_reset_calib.clicked.connect(self._on_reset_clicked)
        self._banner_layout.addWidget(self._btn_reset_calib)

        layout.addWidget(self._banner)

        # --------------------------------------------------------------
        # 2. Step-by-Step Instructions & Visual Diagram
        # --------------------------------------------------------------
        body_row = QHBoxLayout()
        body_row.setSpacing(14)

        # Instructions Card
        instr_card = QFrame()
        instr_card.setStyleSheet(f"""
            QFrame {{
                background-color: {CARD_BG};
                border: 1px solid {BORDER_COLOR};
                border-radius: 8px;
                padding: 14px;
            }}
        """)
        instr_layout = QVBoxLayout(instr_card)
        instr_layout.setContentsMargins(14, 12, 14, 12)
        instr_layout.setSpacing(10)

        lbl_steps_title = QLabel("How 4-Point Calibration Works")
        lbl_steps_title.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 14px; font-weight: 700;")
        instr_layout.addWidget(lbl_steps_title)

        steps = [
            ("1. Position Yourself", "Sit comfortably in your regular workspace chair facing your webcam."),
            ("2. Start the Routine", "Click 'Start 4-Point Calibration' to launch the fullscreen target overlay."),
            ("3. Point at Targets", "Point your index finger toward each of the 4 targets shown on screen:"),
            ("   • Top-Left (TL)", "Hold your index finger pointing at the top-left circle until complete (1 sec)."),
            ("   • Top-Right (TR)", "Move to top-right and hold steady."),
            ("   • Bottom-Right (BR)", "Move to bottom-right and hold steady."),
            ("   • Bottom-Left (BL)", "Move to bottom-left and hold steady."),
            ("4. Apply Transformation", "GestureFlow computes an optimal perspective homography matrix to reach every corner."),
        ]
        for step_title, step_desc in steps:
            s_box = QVBoxLayout()
            s_box.setSpacing(1)
            t = QLabel(step_title)
            t.setStyleSheet(f"color: {ACCENT_CYAN}; font-size: 12px; font-weight: 700;")
            d = QLabel(step_desc)
            d.setWordWrap(True)
            d.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px;")
            s_box.addWidget(t)
            s_box.addWidget(d)
            instr_layout.addLayout(s_box)

        body_row.addWidget(instr_card, stretch=3)

        # Safety & Quality Guidelines Card
        guide_card = QFrame()
        guide_card.setStyleSheet(f"""
            QFrame {{
                background-color: {CARD_BG};
                border: 1px solid {BORDER_COLOR};
                border-radius: 8px;
                padding: 14px;
            }}
        """)
        guide_layout = QVBoxLayout(guide_card)
        guide_layout.setContentsMargins(14, 12, 14, 12)
        guide_layout.setSpacing(10)

        lbl_guide_title = QLabel("Calibration Guidelines & Safety")
        lbl_guide_title.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 14px; font-weight: 700;")
        guide_layout.addWidget(lbl_guide_title)

        # Metric cards for calibration details
        self._metric_calib_type = MetricCard("Mapping Mode", "Default Bounding Box")
        self._metric_quality = MetricCard("Calibration Quality", "Uncalibrated")
        self._metric_matrix = MetricCard("Homography Status", "Identity Matrix")
        guide_layout.addWidget(self._metric_calib_type)
        guide_layout.addWidget(self._metric_quality)
        guide_layout.addWidget(self._metric_matrix)

        notice_box = QFrame()
        notice_box.setStyleSheet(f"""
            QFrame {{
                background-color: rgba(52, 211, 153, 0.08);
                border-left: 3px solid {ACCENT_EMERALD};
                border-radius: 4px;
                padding: 8px 10px;
            }}
        """)
        n_layout = QVBoxLayout(notice_box)
        n_layout.setContentsMargins(6, 4, 6, 4)
        n_lbl1 = QLabel("Safety Interlock Active")
        n_lbl1.setStyleSheet(f"color: {ACCENT_EMERALD}; font-weight: 700; font-size: 11px;")
        n_lbl2 = QLabel("Virtual mouse cursor automation is unconditionally disabled during calibration sampling.")
        n_lbl2.setWordWrap(True)
        n_lbl2.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 11px;")
        n_layout.addWidget(n_lbl1)
        n_layout.addWidget(n_lbl2)
        guide_layout.addWidget(notice_box)

        guide_layout.addStretch()
        body_row.addWidget(guide_card, stretch=2)

        layout.addLayout(body_row)
        layout.addStretch()

        scroll.setWidget(container)

        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.addWidget(scroll)

        self.refresh_calibration_status()

    def refresh_calibration_status(self) -> None:
        """Update banner and metrics from settings calibration_data."""
        c_data = getattr(self._settings.controller, "calibration_data", None)
        is_active = bool(c_data and c_data.get("is_active"))

        if is_active:
            self._banner.setStyleSheet(f"""
                QFrame {{
                    background-color: rgba(52, 211, 153, 0.12);
                    border: 1px solid {ACCENT_EMERALD};
                    border-radius: 8px;
                }}
            """)
            self._lbl_banner_title.setText("Calibration Status: 4-Point Active ✔")
            self._lbl_banner_title.setStyleSheet(f"color: {ACCENT_EMERALD}; font-size: 15px; font-weight: 800;")
            self._lbl_banner_sub.setText("Perspective homography matrix is actively mapping coordinates")
            self._metric_calib_type.set_value("4-Point Homography", ACCENT_EMERALD)
            self._metric_quality.set_value("Calibrated (High)", ACCENT_EMERALD)
            self._metric_matrix.set_value("Matrix Active (3×3)", ACCENT_CYAN)
        else:
            self._banner.setStyleSheet(f"""
                QFrame {{
                    background-color: {CARD_BG};
                    border: 1px solid {BORDER_COLOR};
                    border-radius: 8px;
                }}
            """)
            self._lbl_banner_title.setText("Calibration Status: Ready (Default Bounds)")
            self._lbl_banner_title.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 15px; font-weight: 800;")
            self._lbl_banner_sub.setText("Using default 15% viewport margin — run calibration to fit your setup")
            self._metric_calib_type.set_value("Default Bounds", TEXT_SECONDARY)
            self._metric_quality.set_value("Uncalibrated", WARNING_AMBER)
            self._metric_matrix.set_value("Default Box", TEXT_MUTED)

    def _on_reset_clicked(self) -> None:
        """Prompt confirmation before clearing calibration data."""
        reply = QMessageBox.question(
            self,
            "Confirm Reset Calibration",
            "Are you sure you want to reset coordinate calibration back to default boundaries?\n\n"
            "Your existing 4-point perspective matrix will be cleared.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.reset_calibration_requested.emit()
            self.refresh_calibration_status()
