"""
gestureflow.ui.components.onboarding_dialog
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
First-run setup wizard dialog guiding users through:
1. Welcome to GestureFlow
2. Camera Selection & Lighting
3. Hand Tracking Verification
4. 4-Point Screen Calibration
5. Core Mouse Gestures (Point, Click, Drag, Scroll)
6. Safety & Emergency Stop Invariants
7. Ready to Flow!

Safety Guarantee: Never activates virtual mouse control automatically.
"""

from __future__ import annotations

from typing import Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QDialog,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
    QStackedWidget,
    QFrame,
    QSizePolicy,
)

from gestureflow.config.settings import Settings
from gestureflow.ui.components.theme import (
    BG_DARK,
    SIDEBAR_BG,
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

STEPS = [
    {
        "title": "Welcome to GestureFlow",
        "subtitle": "Touchless AI Virtual Mouse for Windows",
        "icon": "👋",
        "body": (
            "GestureFlow lets you control your Windows desktop completely hands-free using your webcam and "
            "Google MediaPipe Hand Landmark detection.\n\n"
            "This quick 2-minute walkthrough introduces camera setup, core gestures, calibration, and "
            "essential safety mechanisms to give you complete confidence."
        ),
        "highlight": "Tip: You can reopen this guide anytime from the Settings or About page.",
    },
    {
        "title": "Select Camera & Environment",
        "subtitle": "Optimizing Computer Vision Tracking",
        "icon": "📷",
        "body": (
            "• Physical Webcam: Ensure your preferred camera is connected. GestureFlow defaults to Camera 0 "
            "using the high-performance Windows DirectShow (DSHOW) backend.\n\n"
            "• Lighting: Ensure your hand is evenly illuminated with minimal backlighting or glare.\n\n"
            "• Positioning: Keep your hand 18–30 inches (45–75 cm) away from the lens, facing the camera naturally."
        ),
        "highlight": "Check the Live Tracking tab at any time to verify camera health and frame rate.",
    },
    {
        "title": "Hand Tracking & Landmarker",
        "subtitle": "21 3D Coordinates Tracked at 30 FPS",
        "icon": "✋",
        "body": (
            "GestureFlow continuously predicts 21 3D joint landmarks for up to two hands:\n\n"
            "• Wrist, palm base, and all 5 finger MCP knuckles.\n"
            "• PIP, DIP, and fingertip coordinates.\n"
            "• Automatic primary hand selection with spatial centroid continuity.\n\n"
            "Raise your hand in front of the camera: when detected, a green skeleton will follow your fingers in real time."
        ),
        "highlight": "Optimal tracking quality is maintained when your entire hand stays inside the frame.",
    },
    {
        "title": "4-Point Perspective Calibration",
        "subtitle": "Mapping Hand Motion to Screen Pixels",
        "icon": "🎯",
        "body": (
            "Every person holds their hand at a different distance and angle. To ensure comfortable "
            "cursor reach across all 4 corners of your screen without reaching off-camera:\n\n"
            "1. Click 'Calibrate' on the Dashboard.\n"
            "2. Point to the 4 targets shown on screen (Top-Left, Top-Right, Bottom-Right, Bottom-Left).\n"
            "3. Hold steady for 1 second per corner while samples are collected.\n\n"
            "A homography matrix maps your natural hand posture to your Windows monitor coordinates."
        ),
        "highlight": "Calibration persists across sessions and is preserved when switching response profiles.",
    },
    {
        "title": "Core Hand Gestures",
        "subtitle": "Intuitive Natural Mouse Actions",
        "icon": "🖱",
        "body": (
            "• Cursor Movement: Raise your Index finger (POINTING) to steer the cursor smoothly.\n"
            "• Left Click: Quick pinch between Thumb and Index fingertip.\n"
            "• Double Click: Two rapid pinches within 350ms.\n"
            "• Right Click: Quick pinch between Thumb and Middle fingertip.\n"
            "• Drag & Drop: Pinch and hold Thumb and Index together for > 0.45s, then move.\n"
            "• Scroll: Raise both Index and Middle fingers together (TWO_FINGER) and move vertically."
        ),
        "highlight": "Virtual mouse control is DISABLED by default on startup until you explicitly enable it.",
    },
    {
        "title": "Safety & Emergency Controls",
        "subtitle": "You Are Always in Complete Control",
        "icon": "🛑",
        "body": (
            "GestureFlow is engineered with strict human-in-the-loop safety invariants:\n\n"
            "• PyAutoGUI Fail-Safe: Flick your physical mouse to any corner of your screen to instantly abort automation.\n"
            "• Emergency Stop: Press the ESC key on your keyboard or click the red 'Emergency Stop' button anytime.\n"
            "• Tracking Loss Auto-Release: If your hand drops out of frame, any held mouse buttons are instantly released.\n"
            "• Physical Mouse Override: Moving your physical mouse takes immediate precedence."
        ),
        "highlight": "Virtual mouse control will never activate automatically without your consent.",
    },
    {
        "title": "You're Ready to Flow!",
        "subtitle": "Desktop Control is at Your Fingertips",
        "icon": "🚀",
        "body": (
            "Setup is complete! Here's how to begin:\n\n"
            "1. Click '▶ Start' on the Dashboard to start camera capture.\n"
            "2. Raise your hand to confirm tracking.\n"
            "3. When ready, click 'Enable Mouse' to activate virtual cursor movement.\n"
            "4. Remember: Press ESC anytime for an instant Emergency Stop."
        ),
        "highlight": "Have fun exploring touchless computing with GestureFlow!",
    },
]


class OnboardingDialog(QDialog):
    """
    7-step onboarding wizard dialog.
    """

    wizard_finished = Signal()

    def __init__(self, settings: Settings, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._current_step = 0

        self.setWindowTitle("GestureFlow — Setup Guide")
        self.resize(680, 520)
        self.setMinimumSize(580, 440)
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {BG_DARK};
                border: 1px solid {BORDER_COLOR};
            }}
        """)

        self._build_ui()

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 20, 24, 20)
        main_layout.setSpacing(16)

        # Top progress bar & indicator
        top_bar = QHBoxLayout()
        self._lbl_step_num = QLabel(f"Step 1 of {len(STEPS)}")
        self._lbl_step_num.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; font-weight: 700;")
        top_bar.addWidget(self._lbl_step_num)
        top_bar.addStretch()

        self._btn_skip = QPushButton("Skip Guide")
        self._btn_skip.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {TEXT_SECONDARY};
                border: none;
                font-size: 11px;
            }}
            QPushButton:hover {{
                color: {TEXT_PRIMARY};
            }}
        """)
        self._btn_skip.clicked.connect(self._on_skip)
        top_bar.addWidget(self._btn_skip)
        main_layout.addLayout(top_bar)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, len(STEPS))
        self._progress_bar.setValue(1)
        self._progress_bar.setFixedHeight(4)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background: {SIDEBAR_BG};
                border: none;
                border-radius: 2px;
            }}
            QProgressBar::chunk {{
                background: {ACCENT_EMERALD};
                border-radius: 2px;
            }}
        """)
        main_layout.addWidget(self._progress_bar)

        # Stacked Step Cards
        self._stack = QStackedWidget()
        for step in STEPS:
            card = self._create_step_card(step)
            self._stack.addWidget(card)
        main_layout.addWidget(self._stack)

        # Bottom navigation row: Back, Next, Finish
        bottom_bar = QHBoxLayout()
        bottom_bar.setContentsMargins(0, 10, 0, 0)

        self._btn_back = QPushButton("◀ Back")
        self._btn_back.setEnabled(False)
        self._btn_back.clicked.connect(self._on_back)
        bottom_bar.addWidget(self._btn_back)

        bottom_bar.addStretch()

        self._btn_next = QPushButton("Next ▶")
        self._btn_next.setObjectName("btn_primary")
        self._btn_next.setStyleSheet(f"""
            QPushButton#btn_primary {{
                background-color: {ACCENT_EMERALD};
                color: #064E3B;
                border: none;
                font-weight: 700;
                padding: 8px 24px;
                border-radius: 6px;
            }}
            QPushButton#btn_primary:hover {{
                background-color: #10B981;
            }}
        """)
        self._btn_next.clicked.connect(self._on_next)
        bottom_bar.addWidget(self._btn_next)

        main_layout.addLayout(bottom_bar)

    def _create_step_card(self, step_data: dict) -> QWidget:
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background-color: {CARD_BG};
                border: 1px solid {BORDER_COLOR};
                border-radius: 10px;
                padding: 16px;
            }}
        """)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Title row
        title_row = QHBoxLayout()
        icon_lbl = QLabel(step_data["icon"])
        icon_lbl.setStyleSheet("font-size: 32px;")
        title_row.addWidget(icon_lbl)

        title_box = QVBoxLayout()
        title_lbl = QLabel(step_data["title"])
        title_lbl.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 18px; font-weight: 800;")
        sub_lbl = QLabel(step_data["subtitle"])
        sub_lbl.setStyleSheet(f"color: {ACCENT_CYAN}; font-size: 12px; font-weight: 600;")
        title_box.addWidget(title_lbl)
        title_box.addWidget(sub_lbl)
        title_row.addLayout(title_box)
        title_row.addStretch()
        layout.addLayout(title_row)

        # Body text
        body_lbl = QLabel(step_data["body"])
        body_lbl.setWordWrap(True)
        body_lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 13px; line-height: 1.5;")
        layout.addWidget(body_lbl)

        layout.addStretch()

        # Highlight callout box
        highlight_box = QFrame()
        highlight_box.setStyleSheet(f"""
            QFrame {{
                background-color: rgba(34, 211, 238, 0.08);
                border-left: 3px solid {ACCENT_CYAN};
                border-radius: 4px;
                padding: 8px 12px;
            }}
        """)
        hl_layout = QHBoxLayout(highlight_box)
        hl_layout.setContentsMargins(8, 4, 8, 4)
        hl_lbl = QLabel(step_data["highlight"])
        hl_lbl.setWordWrap(True)
        hl_lbl.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 12px; font-weight: 500;")
        hl_layout.addWidget(hl_lbl)
        layout.addWidget(highlight_box)

        return card

    def _on_next(self) -> None:
        if self._current_step < len(STEPS) - 1:
            self._current_step += 1
            self._update_step()
        else:
            self._on_finish()

    def _on_back(self) -> None:
        if self._current_step > 0:
            self._current_step -= 1
            self._update_step()

    def _update_step(self) -> None:
        self._stack.setCurrentIndex(self._current_step)
        self._progress_bar.setValue(self._current_step + 1)
        self._lbl_step_num.setText(f"Step {self._current_step + 1} of {len(STEPS)}")
        self._btn_back.setEnabled(self._current_step > 0)

        if self._current_step == len(STEPS) - 1:
            self._btn_next.setText("Finish Setup ✔")
        else:
            self._btn_next.setText("Next ▶")

    def _on_finish(self) -> None:
        self._settings.ui.onboarding_completed = True
        self._settings.save_user_settings()
        self.wizard_finished.emit()
        self.accept()

    def _on_skip(self) -> None:
        self._settings.ui.onboarding_completed = True
        self._settings.save_user_settings()
        self.reject()
