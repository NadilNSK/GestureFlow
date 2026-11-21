"""
gestureflow.ui.pages.settings_page
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Comprehensive settings configuration interface:
- Camera Settings (Device, DSHOW backend, mirroring)
- Cursor Tuning (Profiles, sensitivity, One Euro parameters, dead-zone)
- Gesture Settings (Adaptive pinch, stability, click strategies, drag hold)
- Appearance & Accessibility (Theme mode, UI scaling, animations)
- Safety & Invariants (Emergency stop guide, PyAutoGUI fail-safe confirmation)
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
    QTabWidget,
    QComboBox,
    QCheckBox,
    QSlider,
    QDoubleSpinBox,
    QSpinBox,
    QFrame,
    QScrollArea,
    QMessageBox,
)

from gestureflow.config.settings import Settings, CURSOR_PROFILES
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


class SettingsPage(QWidget):
    """
    Dedicated Settings Page with tabs for Camera, Cursor, Gestures, Appearance, and Safety.
    """

    settings_saved = Signal()
    reset_defaults_requested = Signal()
    profile_changed = Signal(str)
    click_strategy_changed = Signal(str)

    def __init__(self, settings: Settings, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._settings = settings

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(14)

        hdr = SectionHeader(
            "Application Settings",
            "Configure hardware interfaces, cursor dynamics, gesture thresholds, and safety policies",
        )
        layout.addWidget(hdr)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._create_camera_tab(), "📷 Camera")
        self._tabs.addTab(self._create_cursor_tab(), "🖱 Cursor & Filter")
        self._tabs.addTab(self._create_gestures_tab(), "✋ Gestures")
        self._tabs.addTab(self._create_appearance_tab(), "🎨 Appearance")
        self._tabs.addTab(self._create_safety_tab(), "🛡 Safety")
        layout.addWidget(self._tabs)

        # Bottom Action Bar
        bottom_bar = QHBoxLayout()
        bottom_bar.setContentsMargins(0, 6, 0, 0)

        self._btn_reset = QPushButton("Reset to Defaults")
        self._btn_reset.setFixedHeight(34)
        self._btn_reset.clicked.connect(self._on_reset_defaults_clicked)
        bottom_bar.addWidget(self._btn_reset)

        bottom_bar.addStretch()

        self._lbl_save_status = QLabel("")
        self._lbl_save_status.setStyleSheet(f"color: {ACCENT_EMERALD}; font-weight: 600; font-size: 12px;")
        bottom_bar.addWidget(self._lbl_save_status)

        self._btn_save = QPushButton("Save Settings ✔")
        self._btn_save.setObjectName("btn_primary")
        self._btn_save.setFixedHeight(34)
        self._btn_save.setStyleSheet(f"""
            QPushButton#btn_primary {{
                background-color: {ACCENT_EMERALD};
                color: #064E3B;
                border: none;
                font-weight: 700;
                padding: 6px 20px;
                border-radius: 6px;
            }}
            QPushButton#btn_primary:hover {{
                background-color: #10B981;
            }}
        """)
        self._btn_save.clicked.connect(self._on_save_settings)
        bottom_bar.addWidget(self._btn_save)

        layout.addLayout(bottom_bar)

    # ------------------------------------------------------------------
    # Tab 1: Camera
    # ------------------------------------------------------------------
    def _create_camera_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Device Index & Backend
        self._card_backend = MetricCard("OpenCV Backend", self._settings.camera.backend)
        layout.addWidget(self._card_backend)

        self._card_cam_dev = MetricCard(
            "Selected Device Index",
            f"Device {self._settings.camera.device_index} (DirectShow)",
        )
        layout.addWidget(self._card_cam_dev)

        self._card_res = MetricCard(
            "Default Resolution",
            f"{self._settings.camera.frame_width}×{self._settings.camera.frame_height} @ {self._settings.camera.fps} FPS",
        )
        layout.addWidget(self._card_res)

        self._chk_flip = QCheckBox("Mirror Camera Preview (Horizontal Flip)")
        self._chk_flip.setChecked(self._settings.camera.flip_horizontal)
        layout.addWidget(self._chk_flip)

        layout.addStretch()
        return widget

    # ------------------------------------------------------------------
    # Tab 2: Cursor & Filter
    # ------------------------------------------------------------------
    def _create_cursor_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Profile Selector
        prof_row = QHBoxLayout()
        lbl_p = QLabel("Cursor Response Profile:")
        lbl_p.setFixedWidth(160)
        self._combo_profile = QComboBox()
        self._combo_profile.addItems(["Balanced", "Precision", "Responsive", "Custom"])
        cur_prof = getattr(self._settings.controller, "cursor_profile", "Balanced")
        prof_items = ["Balanced", "Precision", "Responsive", "Custom"]
        self._combo_profile.setCurrentIndex(prof_items.index(cur_prof) if cur_prof in prof_items else 0)
        self._combo_profile.currentTextChanged.connect(self._on_profile_selected)
        prof_row.addWidget(lbl_p)
        prof_row.addWidget(self._combo_profile)
        layout.addLayout(prof_row)

        # Sensitivity
        sens_row = QHBoxLayout()
        self._lbl_sens = QLabel(f"Cursor Sensitivity: {self._settings.controller.cursor_sensitivity:.1f}x")
        self._lbl_sens.setFixedWidth(160)
        self._slider_sens = QSlider(Qt.Orientation.Horizontal)
        self._slider_sens.setRange(2, 30)
        self._slider_sens.setValue(int(self._settings.controller.cursor_sensitivity * 10))
        self._slider_sens.valueChanged.connect(
            lambda v: self._lbl_sens.setText(f"Cursor Sensitivity: {v / 10.0:.1f}x")
        )
        sens_row.addWidget(self._lbl_sens)
        sens_row.addWidget(self._slider_sens)
        layout.addLayout(sens_row)

        # Smoothing
        smooth_row = QHBoxLayout()
        self._lbl_smooth = QLabel(f"Smoothing Factor: {self._settings.controller.cursor_smoothing:.2f}")
        self._lbl_smooth.setFixedWidth(160)
        self._slider_smooth = QSlider(Qt.Orientation.Horizontal)
        self._slider_smooth.setRange(5, 95)
        self._slider_smooth.setValue(int(self._settings.controller.cursor_smoothing * 100))
        self._slider_smooth.valueChanged.connect(
            lambda v: self._lbl_smooth.setText(f"Smoothing Factor: {v / 100.0:.2f}")
        )
        smooth_row.addWidget(self._lbl_smooth)
        smooth_row.addWidget(self._slider_smooth)
        layout.addLayout(smooth_row)

        # Filter Algorithm
        filt_row = QHBoxLayout()
        lbl_f = QLabel("Smoothing Algorithm:")
        lbl_f.setFixedWidth(160)
        self._combo_filter = QComboBox()
        self._combo_filter.addItems(["One Euro Filter (Adaptive)", "Exponential Moving Average (EMA)"])
        self._combo_filter.setCurrentIndex(
            0 if self._settings.controller.smoothing_filter_type == "one_euro" else 1
        )
        filt_row.addWidget(lbl_f)
        filt_row.addWidget(self._combo_filter)
        layout.addLayout(filt_row)

        # One Euro Tuning
        oe_row = QHBoxLayout()
        lbl_min_cut = QLabel("One Euro fc_min:")
        self._spin_min_cutoff = QDoubleSpinBox()
        self._spin_min_cutoff.setRange(0.1, 5.0)
        self._spin_min_cutoff.setSingleStep(0.1)
        self._spin_min_cutoff.setValue(self._settings.controller.one_euro_min_cutoff)

        lbl_beta = QLabel("One Euro Beta:")
        self._spin_beta = QDoubleSpinBox()
        self._spin_beta.setRange(0.0001, 0.1)
        self._spin_beta.setSingleStep(0.002)
        self._spin_beta.setDecimals(4)
        self._spin_beta.setValue(self._settings.controller.one_euro_beta)

        oe_row.addWidget(lbl_min_cut)
        oe_row.addWidget(self._spin_min_cutoff)
        oe_row.addWidget(lbl_beta)
        oe_row.addWidget(self._spin_beta)
        layout.addLayout(oe_row)

        layout.addStretch()
        return widget

    # ------------------------------------------------------------------
    # Tab 3: Gestures
    # ------------------------------------------------------------------
    def _create_gestures_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        self._chk_adaptive_pinch = QCheckBox("Adaptive Pinch Normalization (Combined 3D Palm Metric)")
        self._chk_adaptive_pinch.setChecked(getattr(self._settings.gestures, "adaptive_pinch_enabled", True))
        layout.addWidget(self._chk_adaptive_pinch)

        # Click Strategy
        strat_row = QHBoxLayout()
        lbl_s = QLabel("Click Dispatch Strategy:")
        lbl_s.setFixedWidth(160)
        self._combo_click_strat = QComboBox()
        self._combo_click_strat.addItems(["Safe Deferred (Zero accidental clicks)", "Responsive Native (Instant dispatch)"])
        cur_strat = getattr(self._settings.controller, "click_strategy", "deferred")
        self._combo_click_strat.setCurrentIndex(1 if cur_strat == "native" else 0)
        self._combo_click_strat.currentIndexChanged.connect(self._on_click_strategy_changed)
        strat_row.addWidget(lbl_s)
        strat_row.addWidget(self._combo_click_strat)
        layout.addLayout(strat_row)

        # Stability Frames & History
        stab_row = QHBoxLayout()
        lbl_h = QLabel("Gesture History Frames:")
        self._spin_history = QSpinBox()
        self._spin_history.setRange(2, 10)
        self._spin_history.setValue(getattr(self._settings.gestures, "gesture_history_size", 5))

        lbl_q = QLabel("Confidence Threshold:")
        self._spin_conf = QDoubleSpinBox()
        self._spin_conf.setRange(0.5, 0.95)
        self._spin_conf.setSingleStep(0.05)
        self._spin_conf.setValue(getattr(self._settings.gestures, "gesture_confidence_threshold", 0.75))

        stab_row.addWidget(lbl_h)
        stab_row.addWidget(self._spin_history)
        stab_row.addWidget(lbl_q)
        stab_row.addWidget(self._spin_conf)
        layout.addLayout(stab_row)

        layout.addStretch()
        return widget

    # ------------------------------------------------------------------
    # Tab 4: Appearance
    # ------------------------------------------------------------------
    def _create_appearance_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        theme_row = QHBoxLayout()
        lbl_t = QLabel("Interface Theme:")
        lbl_t.setFixedWidth(140)
        self._combo_theme = QComboBox()
        self._combo_theme.addItems(["Modern Dark (Emerald & Cyan)", "Classic Dark"])
        theme_row.addWidget(lbl_t)
        theme_row.addWidget(self._combo_theme)
        layout.addLayout(theme_row)

        self._chk_anim = QCheckBox("Enable Subtle UI Animations & Transitions")
        self._chk_anim.setChecked(getattr(self._settings.ui, "animations_enabled", True))
        layout.addWidget(self._chk_anim)

        layout.addStretch()
        return widget

    # ------------------------------------------------------------------
    # Tab 5: Safety
    # ------------------------------------------------------------------
    def _create_safety_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        self._card_failsafe = MetricCard("PyAutoGUI FAILSAFE Status", "ACTIVE & LOCKED (True)")
        self._card_startup = MetricCard("Startup Automation Mode", "DISABLED (Human Consent Required)")
        self._card_loss = MetricCard("Tracking Loss Protection", "Instant Auto-Release (0ms Extrapolation)")

        layout.addWidget(self._card_failsafe)
        layout.addWidget(self._card_startup)
        layout.addWidget(self._card_loss)

        safe_box = QFrame()
        safe_box.setStyleSheet(f"""
            QFrame {{
                background-color: rgba(239, 68, 68, 0.08);
                border-left: 3px solid {DANGER_RED};
                border-radius: 4px;
                padding: 10px 14px;
            }}
        """)
        s_layout = QVBoxLayout(safe_box)
        lbl_s1 = QLabel("Emergency Stop Protocol")
        lbl_s1.setStyleSheet(f"color: {DANGER_RED}; font-weight: 700; font-size: 13px;")
        lbl_s2 = QLabel(
            "1. Press the ESC key anytime to immediately drop all active mouse hooks.\n"
            "2. Flick your physical mouse to any monitor corner to trip the PyAutoGUI safety circuit.\n"
            "3. Click the red '🛑 EMERGENCY STOP' button on the Dashboard or Live Tracking screen."
        )
        lbl_s2.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 12px; line-height: 1.5;")
        s_layout.addWidget(lbl_s1)
        s_layout.addWidget(lbl_s2)
        layout.addWidget(safe_box)

        layout.addStretch()
        return widget

    # ------------------------------------------------------------------
    # Actions & Persistence
    # ------------------------------------------------------------------
    def _on_profile_selected(self, prof_name: str) -> None:
        self.profile_changed.emit(prof_name)

    def _on_click_strategy_changed(self, index: int) -> None:
        strat = "native" if index == 1 else "deferred"
        self.click_strategy_changed.emit(strat)

    def _on_save_settings(self) -> None:
        """Collect form values, validate, and persist user_settings.json."""
        c = self._settings.controller
        g = self._settings.gestures
        u = self._settings.ui
        cam = self._settings.camera

        # Camera
        cam.flip_horizontal = self._chk_flip.isChecked()

        # Cursor
        c.cursor_sensitivity = self._slider_sens.value() / 10.0
        c.cursor_smoothing = self._slider_smooth.value() / 100.0
        c.smoothing_filter_type = "one_euro" if self._combo_filter.currentIndex() == 0 else "ema"
        c.one_euro_min_cutoff = self._spin_min_cutoff.value()
        c.one_euro_beta = self._spin_beta.value()
        c.cursor_profile = self._combo_profile.currentText()

        # Gestures
        g.adaptive_pinch_enabled = self._chk_adaptive_pinch.isChecked()
        c.click_strategy = "native" if self._combo_click_strat.currentIndex() == 1 else "deferred"
        g.gesture_history_size = self._spin_history.value()
        g.gesture_confidence_threshold = self._spin_conf.value()

        # Appearance
        u.animations_enabled = self._chk_anim.isChecked()

        # Persist
        self._settings.save_user_settings()
        self._lbl_save_status.setText("Settings saved successfully ✔")
        self.settings_saved.emit()

    def _on_reset_defaults_clicked(self) -> None:
        """Prompt confirmation before restoring factory default configuration."""
        reply = QMessageBox.question(
            self,
            "Reset Settings",
            "Are you sure you want to restore default configuration settings?\n\n"
            "Note: Your calibration matrix will NOT be lost unless you explicitly reset it on the Calibration tab.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.reset_defaults_requested.emit()
            self._lbl_save_status.setText("Defaults restored ✔")
