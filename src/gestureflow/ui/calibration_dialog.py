"""
gestureflow.ui.calibration_dialog
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Interactive 4-point calibration dialog for accurate hand-to-screen coordinate mapping.

Displays corner targets (TL, TR, BR, BL) on screen, collects stable pointing samples,
computes perspective/affine transformation matrices, and applies calibration.
"""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QBrush
from PySide6.QtWidgets import (
    QDialog,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
    QMessageBox,
    QFrame,
)

from gestureflow.calibration.calibrator import (
    CALIBRATION_CORNERS,
    CORNER_TL,
    CORNER_TR,
    CORNER_BR,
    CORNER_BL,
    DEFAULT_TARGETS,
    Calibrator,
)
from gestureflow.config.settings import Settings
from gestureflow.gestures.classifier import Gesture

log = logging.getLogger(__name__)

SAMPLES_PER_CORNER = 25


class CalibrationTargetOverlay(QWidget):
    """Widget that renders target indicators at the 4 screen corners."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._active_corner: str = CORNER_TL
        self._progress: float = 0.0  # 0.0 to 1.0
        self._pulse: float = 0.0
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)

    def set_target(self, corner_id: str, progress: float) -> None:
        self._active_corner = corner_id
        self._progress = max(0.0, min(1.0, progress))
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        # Semi-transparent dark backdrop
        painter.fillRect(self.rect(), QColor(15, 15, 20, 220))

        # Draw all 4 targets
        for corner in CALIBRATION_CORNERS:
            norm_x, norm_y = DEFAULT_TARGETS[corner]
            cx = int(norm_x * w)
            cy = int(norm_y * h)

            is_active = (corner == self._active_corner)

            if is_active:
                # Active target: glowing cyan/green with progress arc
                # Outer ring
                painter.setPen(QPen(QColor(0, 229, 255, 180), 2, Qt.PenStyle.DashLine))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(cx - 36, cy - 36, 72, 72)

                # Progress arc
                painter.setPen(QPen(QColor(0, 255, 128), 5))
                span_angle = int(self._progress * 360 * 16)
                painter.drawArc(cx - 36, cy - 36, 72, 72, 90 * 16, -span_angle)

                # Center dot
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(QColor(0, 229, 255)))
                painter.drawEllipse(cx - 10, cy - 10, 20, 20)

                # Label
                painter.setPen(QPen(QColor(255, 255, 255)))
                painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
                txt = f"POINT HERE ({int(self._progress * 100)}%)"
                painter.drawText(cx - 80, cy + 50, 160, 24, Qt.AlignmentFlag.AlignCenter, txt)
            else:
                # Inactive / completed target
                painter.setPen(QPen(QColor(80, 80, 90), 1))
                painter.setBrush(QBrush(QColor(30, 30, 40, 150)))
                painter.drawEllipse(cx - 16, cy - 16, 32, 32)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(QColor(100, 100, 110)))
                painter.drawEllipse(cx - 4, cy - 4, 8, 8)


class CalibrationDialog(QDialog):
    """
    Modal dialog guiding the user through 4-point screen calibration.
    """

    calibration_completed = Signal(object)  # Emits Calibrator

    def __init__(
        self,
        settings: Settings,
        calibrator: Calibrator | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._settings = settings
        self._calibrator = calibrator or Calibrator()
        self._sample_counts: dict[str, int] = {c: 0 for c in CALIBRATION_CORNERS}
        self._is_finished = False

        self.setWindowTitle("GestureFlow — 4-Point Screen Calibration")
        self.resize(900, 600)
        self.setMinimumSize(700, 500)

        self._calibrator.reset_sampling()
        self._build_ui()

    def _build_ui(self) -> None:
        self.setStyleSheet("""
            QDialog { background: #141418; color: #eee; }
            QLabel { color: #ddd; font-family: 'Segoe UI', sans-serif; }
            QPushButton {
                background: #2b2b36; color: #fff; border: 1px solid #444;
                border-radius: 4px; padding: 6px 16px; font-weight: bold;
            }
            QPushButton:hover { background: #3a3a4a; }
            QPushButton#btn_save {
                background: #2e7d32; border: 1px solid #388e3c;
            }
            QPushButton#btn_save:hover { background: #388e3c; }
            QProgressBar {
                border: 1px solid #444; border-radius: 4px;
                background: #1f1f26; text-align: center; color: #fff;
            }
            QProgressBar::chunk { background: #00e5ff; }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Header instructions
        header = QFrame()
        header.setStyleSheet("background: #1c1c24; border-radius: 6px; padding: 6px;")
        h_layout = QVBoxLayout(header)
        h_layout.setContentsMargins(8, 6, 8, 6)

        title = QLabel("🎯 4-Point Screen Calibration")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #00e5ff;")
        h_layout.addWidget(title)

        self._lbl_instruction = QLabel(
            "Instructions: Point your index finger (POINTING gesture) at each flashing corner target. "
            "Hold still until the countdown completes."
        )
        self._lbl_instruction.setWordWrap(True)
        self._lbl_instruction.setStyleSheet("color: #bbb; font-size: 12px;")
        h_layout.addWidget(self._lbl_instruction)

        layout.addWidget(header)

        # Target overlay container
        self._overlay = CalibrationTargetOverlay(self)
        layout.addWidget(self._overlay, stretch=1)

        # Bottom control panel
        bottom_panel = QFrame()
        b_layout = QHBoxLayout(bottom_panel)
        b_layout.setContentsMargins(0, 4, 0, 0)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, SAMPLES_PER_CORNER)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("Target Progress: %v / %m")
        self._progress_bar.setFixedHeight(26)
        b_layout.addWidget(self._progress_bar, stretch=2)

        self._btn_skip = QPushButton("Next Corner ⏭")
        self._btn_skip.clicked.connect(self._on_skip_corner)
        b_layout.addWidget(self._btn_skip)

        self._btn_save = QPushButton("Save Calibration ✔")
        self._btn_save.setObjectName("btn_save")
        self._btn_save.setEnabled(False)
        self._btn_save.clicked.connect(self._on_save_calibration)
        b_layout.addWidget(self._btn_save)

        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.clicked.connect(self.reject)
        b_layout.addWidget(self._btn_cancel)

        layout.addWidget(bottom_panel)

        self._update_target_display()

    def _update_target_display(self) -> None:
        """Update instructions and target overlay for current corner."""
        corner = self._calibrator.current_target_corner
        if not corner or self._is_finished:
            self._lbl_instruction.setText("✅ All 4 corners sampled! Review and click 'Save Calibration'.")
            self._btn_save.setEnabled(True)
            self._btn_skip.setEnabled(False)
            self._overlay.set_target("", 1.0)
            self._progress_bar.setValue(SAMPLES_PER_CORNER)
            return

        names = {
            CORNER_TL: "Top-Left (1/4)",
            CORNER_TR: "Top-Right (2/4)",
            CORNER_BR: "Bottom-Right (3/4)",
            CORNER_BL: "Bottom-Left (4/4)",
        }
        name = names.get(corner, corner)
        self._lbl_instruction.setText(
            f"🎯 Aim at target: <b style='color:#00e5ff;'>{name}</b>. "
            "Point your index finger at the circle and hold still."
        )
        count = self._sample_counts.get(corner, 0)
        self._progress_bar.setValue(count)
        self._overlay.set_target(corner, count / float(SAMPLES_PER_CORNER))

    def process_pipeline_frame(self, pipeline_frame) -> None:
        """
        Feed live camera frame into calibration sampling if POINTING gesture is active.
        """
        if self._is_finished:
            return

        active_hand = getattr(pipeline_frame, "active_hand", None)
        g_res = getattr(pipeline_frame, "gesture_result", None)

        if not active_hand or not g_res:
            return

        # Check for POINTING gesture
        gesture = getattr(g_res, "gesture", None)
        if gesture != Gesture.POINTING:
            return

        landmarks = getattr(active_hand, "landmarks", None)
        if not landmarks or len(landmarks) <= 8:
            return

        lm8 = landmarks[8]
        norm_x = getattr(lm8, "x", None)
        norm_y = getattr(lm8, "y", None)
        if norm_x is None or norm_y is None:
            return

        corner = self._calibrator.current_target_corner
        if not corner:
            return

        count = self._calibrator.add_sample(corner, norm_x, norm_y)
        self._sample_counts[corner] = count
        self._update_target_display()

        if count >= SAMPLES_PER_CORNER:
            has_next = self._calibrator.advance_to_next_corner()
            if not has_next:
                self._finish_sampling()
            else:
                self._update_target_display()

    @Slot()
    def _on_skip_corner(self) -> None:
        """Manually advance to next corner if needed."""
        corner = self._calibrator.current_target_corner
        if corner:
            # If not enough samples, add default center target fallback
            if self._sample_counts[corner] < 5:
                tx, ty = DEFAULT_TARGETS[corner]
                for _ in range(SAMPLES_PER_CORNER):
                    self._calibrator.add_sample(corner, tx, ty)
                self._sample_counts[corner] = SAMPLES_PER_CORNER

        has_next = self._calibrator.advance_to_next_corner()
        if not has_next:
            self._finish_sampling()
        else:
            self._update_target_display()

    def _finish_sampling(self) -> None:
        """Calculate transformation matrices and show completion status."""
        self._is_finished = True
        success = self._calibrator.compute_calibration()
        self._update_target_display()

        if success:
            q = self._calibrator.data.quality_score * 100
            msg = f"Calibration computed successfully!\nQuality score: {q:.0f}%"
            log.info(msg)
        else:
            log.warning("Calibration calculation failed; default bounds used.")

    @Slot()
    def _on_save_calibration(self) -> None:
        """Save calibrated parameters into application settings and accept dialog."""
        if not self._is_finished:
            self._calibrator.compute_calibration()

        calib_dict = self._calibrator.data.to_dict()
        self._settings.controller.calibration_data = calib_dict
        self._settings.save_user_settings()

        log.info("Saved calibration settings to user_settings.json.")
        self.calibration_completed.emit(self._calibrator)
        self.accept()
