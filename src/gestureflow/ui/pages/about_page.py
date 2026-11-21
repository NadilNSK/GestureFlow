"""
gestureflow.ui.pages.about_page
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
About GestureFlow desktop application:
- Product branding, version metadata, and developer attribution
- Comprehensive technical stack badges
- Keyboard shortcut quick reference
- Troubleshooting and lighting guide
- One-click trigger to launch the First-Run Setup Guide
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
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
)

from gestureflow import __version__, __author__, __license__
from gestureflow.ui.components.theme import (
    CARD_BG,
    BORDER_COLOR,
    ACCENT_EMERALD,
    ACCENT_CYAN,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_MUTED,
)
from gestureflow.ui.components.cards import SectionHeader, MetricCard


class AboutPage(QWidget):
    """
    Dedicated About Page with product info, shortcuts, tech stack, and onboarding launcher.
    """

    launch_onboarding_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(16)

        # --------------------------------------------------------------
        # 1. Branding Hero Banner
        # --------------------------------------------------------------
        hero_card = QFrame()
        hero_card.setStyleSheet(f"""
            QFrame {{
                background-color: {CARD_BG};
                border: 1px solid {BORDER_COLOR};
                border-radius: 10px;
                padding: 16px;
            }}
        """)
        hero_layout = QHBoxLayout(hero_card)
        hero_layout.setContentsMargins(16, 12, 16, 12)

        lbl_icon = QLabel("🖐")
        lbl_icon.setStyleSheet("font-size: 48px; padding-right: 12px;")
        hero_layout.addWidget(lbl_icon)

        hero_text = QVBoxLayout()
        hero_text.setSpacing(2)
        lbl_title = QLabel(f"GestureFlow — AI Virtual Mouse")
        lbl_title.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 20px; font-weight: 800;")
        lbl_desc = QLabel(
            f"Version {__version__} • Built by {__author__} • Licensed under {__license__}\n"
            "High-precision touchless human-computer interface for Windows."
        )
        lbl_desc.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px; line-height: 1.4;")
        hero_text.addWidget(lbl_title)
        hero_text.addWidget(lbl_desc)
        hero_layout.addLayout(hero_text)
        hero_layout.addStretch()

        self._btn_launch_guide = QPushButton("🚀 Launch Setup Guide")
        self._btn_launch_guide.setObjectName("btn_primary")
        self._btn_launch_guide.setFixedHeight(36)
        self._btn_launch_guide.setStyleSheet(f"""
            QPushButton#btn_primary {{
                background-color: {ACCENT_EMERALD};
                color: #064E3B;
                border: none;
                font-weight: 700;
                padding: 6px 18px;
                border-radius: 6px;
            }}
            QPushButton#btn_primary:hover {{
                background-color: #10B981;
            }}
        """)
        self._btn_launch_guide.clicked.connect(self.launch_onboarding_requested.emit)
        hero_layout.addWidget(self._btn_launch_guide)

        layout.addWidget(hero_card)

        # --------------------------------------------------------------
        # 2. Technology Stack & Architecture
        # --------------------------------------------------------------
        tech_card = QFrame()
        tech_card.setStyleSheet(f"""
            QFrame {{
                background-color: {CARD_BG};
                border: 1px solid {BORDER_COLOR};
                border-radius: 8px;
                padding: 14px;
            }}
        """)
        tech_layout = QVBoxLayout(tech_card)
        tech_layout.setContentsMargins(12, 10, 12, 10)
        tech_layout.setSpacing(8)

        lbl_tech_title = QLabel("Core Technologies & Libraries")
        lbl_tech_title.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 14px; font-weight: 700;")
        tech_layout.addWidget(lbl_tech_title)

        tech_row = QHBoxLayout()
        tech_row.addWidget(MetricCard("Computer Vision Engine", "Google MediaPipe Tasks API"))
        tech_row.addWidget(MetricCard("Webcam Backend", "OpenCV cv2.CAP_DSHOW"))
        tech_row.addWidget(MetricCard("Desktop Interface", "PySide6 (Qt 6.12)"))
        tech_layout.addLayout(tech_row)

        tech_row2 = QHBoxLayout()
        tech_row2.addWidget(MetricCard("OS Mouse Automation", "PyAutoGUI (FAILSAFE Enabled)"))
        tech_row2.addWidget(MetricCard("Mathematics & DSP", "NumPy & One Euro Adaptive Filter"))
        tech_row2.addWidget(MetricCard("Runtime Platform", "Python 3.14 64-bit on Windows"))
        tech_layout.addLayout(tech_row2)

        layout.addWidget(tech_card)

        # --------------------------------------------------------------
        # 3. Keyboard Shortcuts Reference Table
        # --------------------------------------------------------------
        shortcuts_card = QFrame()
        shortcuts_card.setStyleSheet(f"""
            QFrame {{
                background-color: {CARD_BG};
                border: 1px solid {BORDER_COLOR};
                border-radius: 8px;
                padding: 14px;
            }}
        """)
        sc_layout = QVBoxLayout(shortcuts_card)
        sc_layout.setContentsMargins(12, 10, 12, 10)
        sc_layout.setSpacing(10)

        lbl_sc_title = QLabel("Keyboard Shortcuts Reference")
        lbl_sc_title.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 14px; font-weight: 700;")
        sc_layout.addWidget(lbl_sc_title)

        shortcuts = [
            ("Esc", "Emergency Stop: Abort all mouse actions immediately and release held buttons"),
            ("P", "Precision Mode: Toggle cursor precision aiming multiplier (0.35x speed)"),
            ("Screen Corners", "PyAutoGUI Fail-Safe: Flick physical mouse to any monitor corner to halt"),
        ]

        for key, desc in shortcuts:
            row = QHBoxLayout()
            lbl_k = QLabel(key)
            lbl_k.setStyleSheet(f"""
                background-color: rgba(34, 211, 238, 0.15);
                color: {ACCENT_CYAN};
                border: 1px solid {ACCENT_CYAN};
                border-radius: 4px;
                padding: 3px 10px;
                font-weight: 700;
                font-size: 11px;
                min-width: 100px;
            """)
            lbl_d = QLabel(desc)
            lbl_d.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px;")
            row.addWidget(lbl_k)
            row.addWidget(lbl_d)
            row.addStretch()
            sc_layout.addLayout(row)

        layout.addWidget(shortcuts_card)

        # --------------------------------------------------------------
        # 4. Troubleshooting & Tips
        # --------------------------------------------------------------
        help_card = QFrame()
        help_card.setStyleSheet(f"""
            QFrame {{
                background-color: {CARD_BG};
                border: 1px solid {BORDER_COLOR};
                border-radius: 8px;
                padding: 14px;
            }}
        """)
        h_layout = QVBoxLayout(help_card)
        h_layout.setContentsMargins(12, 10, 12, 10)
        h_layout.setSpacing(8)

        lbl_h_title = QLabel("Troubleshooting & Lighting Tips")
        lbl_h_title.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 14px; font-weight: 700;")
        h_layout.addWidget(lbl_h_title)

        tips = [
            "• Camera Not Opening: Confirm camera index (Camera 0 by default) in Settings and verify no other app is using it.",
            "• Jittery Movement: Switch Cursor Profile to 'Precision' or increase One Euro filter beta in Settings.",
            "• Unintended Double Clicks: Increase the Double-Click window duration or switch to 'Safe Deferred' click mode.",
            "• Off-Screen Cursor Reach: Run 4-Point Screen Calibration to map hand movement to your display boundaries.",
        ]
        for tip in tips:
            t_lbl = QLabel(tip)
            t_lbl.setWordWrap(True)
            t_lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 12px; line-height: 1.4;")
            h_layout.addWidget(t_lbl)

        layout.addWidget(help_card)
        layout.addStretch()

        scroll.setWidget(container)

        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.addWidget(scroll)
