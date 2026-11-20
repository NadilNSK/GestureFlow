"""
gestureflow.ui.main_window
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The main application window.

Layout
------
  ┌──────────────────────────────────────────────────────┐
  │  Toolbar: [Start] [Stop] [Settings]   Status bar     │
  ├─────────────────────────┬────────────────────────────┤
  │   Camera preview        │  Side panel                │
  │   (live frame + HUD)    │  · Stats card              │
  │                         │  · Model info              │
  │                         │  · Log tail                │
  └─────────────────────────┴────────────────────────────┘
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QThread, Slot, QTimer, QObject, Signal
from PySide6.QtGui import QImage, QPixmap, QAction, QIcon, QFont, QColor
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QToolBar,
    QStatusBar,
    QSplitter,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QTextEdit,
    QSizePolicy,
    QMessageBox,
    QFrame,
    QSlider,
)

from gestureflow.config.settings import Settings
from gestureflow.tracking.pipeline import PipelineFrame
from gestureflow.ui.workers import PipelineWorker

log = logging.getLogger(__name__)


class _LogSignalEmitter(QObject):
    """Bridge for cross-thread log signal delivery to the GUI thread."""

    message_logged = Signal(str)


class _QtLogHandler(logging.Handler):
    """Logging handler that thread-safely appends records to a QTextEdit widget."""

    def __init__(self, text_edit: QTextEdit) -> None:
        super().__init__()
        self._te = text_edit
        self._emitter = _LogSignalEmitter()
        self._emitter.message_logged.connect(
            self._append_message, Qt.ConnectionType.QueuedConnection
        )
        self.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)s — %(message)s", "%H:%M:%S")
        )

    def _append_message(self, html: str) -> None:
        try:
            self._te.append(html)
            sb = self._te.verticalScrollBar()
            sb.setValue(sb.maximum())
        except RuntimeError:
            pass  # Widget may have been destroyed during shutdown

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            # Colour by level.
            colour = {
                logging.DEBUG: "#888888",
                logging.INFO: "#dddddd",
                logging.WARNING: "#f0c040",
                logging.ERROR: "#ff6060",
                logging.CRITICAL: "#ff40ff",
            }.get(record.levelno, "#dddddd")
            html = f'<span style="color:{colour}; font-family:monospace">{msg}</span>'
            self._emitter.message_logged.emit(html)
        except Exception:
            self.handleError(record)


class MainWindow(QMainWindow):
    """GestureFlow main application window."""

    def __init__(self, settings: Settings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._worker: PipelineWorker | None = None
        self._thread: QThread | None = None
        self._frame_count = 0
        self._log_handler: _QtLogHandler | None = None

        self._build_ui()
        self._apply_dark_theme()
        log.info("MainWindow initialised.")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        s = self._settings.ui
        self.setWindowTitle("GestureFlow — AI Virtual Mouse")
        self.resize(s.window_width, s.window_height)
        self.setMinimumSize(800, 550)

        # ---- Toolbar ----
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)

        self._btn_start = QPushButton("▶  Start")
        self._btn_start.setObjectName("btn_start")
        self._btn_start.setFixedHeight(32)
        self._btn_start.clicked.connect(self._on_start)

        self._btn_stop = QPushButton("■  Stop")
        self._btn_stop.setObjectName("btn_stop")
        self._btn_stop.setFixedHeight(32)
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._on_stop)

        toolbar.addWidget(self._btn_start)
        toolbar.addWidget(self._btn_stop)
        toolbar.addSeparator()

        self._btn_mouse_enable = QPushButton("🖱  Enable Mouse")
        self._btn_mouse_enable.setObjectName("btn_mouse_enable")
        self._btn_mouse_enable.setFixedHeight(32)
        self._btn_mouse_enable.setEnabled(False)
        self._btn_mouse_enable.clicked.connect(self._on_enable_mouse)

        self._btn_mouse_disable = QPushButton("Disable Mouse")
        self._btn_mouse_disable.setObjectName("btn_mouse_disable")
        self._btn_mouse_disable.setFixedHeight(32)
        self._btn_mouse_disable.setEnabled(False)
        self._btn_mouse_disable.clicked.connect(self._on_disable_mouse)

        toolbar.addWidget(self._btn_mouse_enable)
        toolbar.addWidget(self._btn_mouse_disable)
        toolbar.addSeparator()

        self._btn_emergency_stop = QPushButton("🛑 EMERGENCY STOP (Esc)")
        self._btn_emergency_stop.setObjectName("btn_emergency_stop")
        self._btn_emergency_stop.setFixedHeight(32)
        self._btn_emergency_stop.setStyleSheet(
            "QPushButton#btn_emergency_stop { background: #b71c1c; color: white; font-weight: bold; padding: 4px 12px; border-radius: 4px; border: 1px solid #d32f2f; }"
            "QPushButton#btn_emergency_stop:hover { background: #d32f2f; }"
        )
        self._btn_emergency_stop.clicked.connect(self._on_emergency_stop)
        toolbar.addWidget(self._btn_emergency_stop)

        self.addToolBar(toolbar)

        # ---- Status bar ----
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._lbl_status = QLabel("Ready")
        self._status_bar.addWidget(self._lbl_status)

        self._lbl_fps = QLabel("FPS: —")
        self._lbl_fps.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._status_bar.addPermanentWidget(self._lbl_fps)

        # ---- Central widget ----
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # ---- Camera preview ----
        preview_container = QWidget()
        preview_layout = QVBoxLayout(preview_container)
        preview_layout.setContentsMargins(0, 0, 0, 0)

        self._preview_label = QLabel()
        self._preview_label.setObjectName("preview_label")
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._preview_label.setMinimumSize(320, 240)
        self._preview_label.setStyleSheet(
            "QLabel#preview_label { background: #111; border: 1px solid #333; border-radius: 4px; }"
        )
        preview_layout.addWidget(self._preview_label)
        splitter.addWidget(preview_container)

        # ---- Side panel ----
        side_panel = QWidget()
        side_panel.setMinimumWidth(300)
        side_panel.setMaximumWidth(360)
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(8)

        # 1. Recognized Gesture Card
        gesture_group = QGroupBox("Gesture Recognition")
        gesture_layout = QVBoxLayout(gesture_group)
        gesture_layout.setSpacing(4)

        self._lbl_gesture = QLabel("NO GESTURE")
        self._lbl_gesture.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_gesture.setStyleSheet(
            "background: #222; color: #888; font-weight: bold; font-size: 15px; "
            "padding: 8px; border-radius: 4px; border: 1px solid #333;"
        )
        gesture_layout.addWidget(self._lbl_gesture)

        self._lbl_gesture_conf = QLabel("Confidence: —")
        self._lbl_gesture_status = QLabel("Status: Idle")
        gesture_layout.addWidget(self._lbl_gesture_conf)
        gesture_layout.addWidget(self._lbl_gesture_status)
        side_layout.addWidget(gesture_group)

        # 2. Hand Tracking Card
        tracking_group = QGroupBox("Hand Tracking")
        tracking_layout = QVBoxLayout(tracking_group)
        tracking_layout.setSpacing(4)

        self._lbl_hands = QLabel("Hands detected: 0")
        self._lbl_active_hand = QLabel("Active Hand: None")
        self._lbl_track_score = QLabel("Tracking Score: —")
        for lbl in (self._lbl_hands, self._lbl_active_hand, self._lbl_track_score):
            tracking_layout.addWidget(lbl)
        side_layout.addWidget(tracking_group)

        # 3. Finger States Card
        fingers_group = QGroupBox("Finger States")
        fingers_layout = QVBoxLayout(fingers_group)
        fingers_layout.setSpacing(3)

        self._lbl_finger_widgets = {}
        for fname in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            name_lbl = QLabel(f"{fname}:")
            name_lbl.setFixedWidth(60)
            val_lbl = QLabel("FOLDED")
            val_lbl.setStyleSheet("color: #777; font-weight: bold;")
            row.addWidget(name_lbl)
            row.addWidget(val_lbl)
            row.addStretch()
            fingers_layout.addLayout(row)
            self._lbl_finger_widgets[fname] = val_lbl
        side_layout.addWidget(fingers_group)

        # 4. Virtual Mouse Control Card
        mouse_group = QGroupBox("Virtual Mouse Control")
        mouse_layout = QVBoxLayout(mouse_group)
        mouse_layout.setSpacing(5)

        self._lbl_mouse_status = QLabel("MOUSE CONTROL: DISABLED")
        self._lbl_mouse_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_mouse_status.setStyleSheet(
            "background: #222; color: #888; font-weight: bold; font-size: 13px; "
            "padding: 6px; border-radius: 4px; border: 1px solid #333;"
        )
        mouse_layout.addWidget(self._lbl_mouse_status)

        self._lbl_mouse_coords = QLabel("Cursor: (—, —)")
        self._lbl_mouse_coords.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_mouse_coords.setStyleSheet("color: #aaa; font-size: 12px;")
        mouse_layout.addWidget(self._lbl_mouse_coords)

        # Sensitivity slider
        sens_row = QHBoxLayout()
        self._lbl_sens_val = QLabel(f"Sensitivity: {self._settings.controller.cursor_sensitivity:.1f}x")
        self._lbl_sens_val.setFixedWidth(110)
        self._slider_sens = QSlider(Qt.Orientation.Horizontal)
        self._slider_sens.setRange(2, 30)  # 0.2x to 3.0x
        self._slider_sens.setValue(int(self._settings.controller.cursor_sensitivity * 10))
        self._slider_sens.valueChanged.connect(self._on_sensitivity_slider_changed)
        sens_row.addWidget(self._lbl_sens_val)
        sens_row.addWidget(self._slider_sens)
        mouse_layout.addLayout(sens_row)

        # Smoothing slider
        smooth_row = QHBoxLayout()
        self._lbl_smooth_val = QLabel(f"Smoothing: {self._settings.controller.cursor_smoothing:.2f}")
        self._lbl_smooth_val.setFixedWidth(110)
        self._slider_smooth = QSlider(Qt.Orientation.Horizontal)
        self._slider_smooth.setRange(5, 95)  # 0.05 to 0.95
        self._slider_smooth.setValue(int(self._settings.controller.cursor_smoothing * 100))
        self._slider_smooth.valueChanged.connect(self._on_smoothing_slider_changed)
        smooth_row.addWidget(self._lbl_smooth_val)
        smooth_row.addWidget(self._slider_smooth)
        mouse_layout.addLayout(smooth_row)

        # Reset Emergency Stop button
        self._btn_reset_emergency = QPushButton("Clear Emergency Stop")
        self._btn_reset_emergency.setObjectName("btn_reset_emergency")
        self._btn_reset_emergency.setEnabled(False)
        self._btn_reset_emergency.clicked.connect(self._on_reset_emergency_stop)
        mouse_layout.addWidget(self._btn_reset_emergency)

        side_layout.addWidget(mouse_group)

        # 5. Device Info Card
        stats_group = QGroupBox("Device & System")
        stats_layout = QVBoxLayout(stats_group)
        stats_layout.setSpacing(3)
        self._lbl_model = QLabel(f"Model: {Path(self._settings.tracking.model_path).name}")
        self._lbl_resolution = QLabel(
            f"Resolution: {self._settings.camera.frame_width}×{self._settings.camera.frame_height}"
        )
        for lbl in (self._lbl_model, self._lbl_resolution):
            lbl.setWordWrap(True)
            stats_layout.addWidget(lbl)
        side_layout.addWidget(stats_group)

        # 5. Log tail
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self._log_view = QTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.document().setMaximumBlockCount(500)
        self._log_view.setStyleSheet("background:#0d0d0d; color:#ddd; font-family:Consolas,monospace; font-size:10px;")
        log_layout.addWidget(self._log_view)
        log_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        side_layout.addWidget(log_group)

        splitter.addWidget(side_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        # Attach Qt log handler.
        self._log_handler = _QtLogHandler(self._log_view)
        self._log_handler.setLevel(logging.DEBUG)
        logging.getLogger().addHandler(self._log_handler)

        self._show_placeholder()

    # ------------------------------------------------------------------
    # Pipeline control
    # ------------------------------------------------------------------

    @Slot()
    def _on_start(self) -> None:
        if self._thread and self._thread.isRunning():
            return

        log.info("Starting pipeline…")
        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._lbl_status.setText("Starting…")

        is_mock = self._settings.camera.use_mock_camera
        self._btn_mouse_enable.setEnabled(not is_mock)
        if is_mock:
            self._btn_mouse_enable.setToolTip("Virtual mouse control is unavailable in mock camera mode.")
        else:
            self._btn_mouse_enable.setToolTip("Enable virtual mouse cursor control.")
        self._btn_mouse_disable.setEnabled(False)

        self._thread = QThread(self)
        self._worker = PipelineWorker(self._settings)
        self._worker.moveToThread(self._thread)

        # Wire signals.
        self._thread.started.connect(self._worker.start_pipeline)
        self._worker.frame_ready.connect(self._on_frame_ready)
        self._worker.error_occurred.connect(self._on_pipeline_error)
        self._worker.status_changed.connect(self._on_status_changed)

        self._thread.start()

    @Slot()
    def _on_stop(self) -> None:
        log.info("Stopping pipeline…")
        self._btn_stop.setEnabled(False)
        self._btn_mouse_enable.setEnabled(False)
        self._btn_mouse_disable.setEnabled(False)
        if self._worker:
            self._worker.stop_pipeline()
        if self._thread:
            self._thread.quit()
            self._thread.wait(5000)
            self._thread = None
        self._worker = None
        self._btn_start.setEnabled(True)
        self._lbl_status.setText("Stopped")
        self._show_placeholder()
        log.info("Pipeline stopped.")

    @Slot()
    def _on_enable_mouse(self) -> None:
        """Enable virtual mouse cursor control."""
        if self._settings.camera.use_mock_camera:
            QMessageBox.warning(
                self,
                "Safety Guard",
                "Virtual mouse control cannot be enabled while running in mock-camera mode.",
            )
            return

        if self._worker:
            success = self._worker.enable_mouse(True)
            if success:
                self._btn_mouse_enable.setEnabled(False)
                self._btn_mouse_disable.setEnabled(True)
                self._lbl_mouse_status.setText("MOUSE CONTROL: READY (IDLE)")
                self._lbl_mouse_status.setStyleSheet(
                    "background: #0d3840; color: #00e5ff; font-weight: bold; font-size: 13px; "
                    "padding: 6px; border-radius: 4px; border: 1px solid #00b0ff;"
                )
                log.info("Virtual mouse control enabled by user.")
            else:
                log.warning("Could not enable mouse control.")

    @Slot()
    def _on_disable_mouse(self) -> None:
        """Disable virtual mouse cursor control cleanly."""
        if self._worker:
            self._worker.enable_mouse(False)
        is_running = self._thread is not None and self._thread.isRunning()
        self._btn_mouse_enable.setEnabled(is_running and not self._settings.camera.use_mock_camera)
        self._btn_mouse_disable.setEnabled(False)
        self._lbl_mouse_status.setText("MOUSE CONTROL: DISABLED")
        self._lbl_mouse_status.setStyleSheet(
            "background: #222; color: #888; font-weight: bold; font-size: 13px; "
            "padding: 6px; border-radius: 4px; border: 1px solid #333;"
        )
        self._lbl_mouse_coords.setText("Cursor: (—, —)")
        log.info("Virtual mouse control disabled by user.")

    @Slot()
    def _on_emergency_stop(self) -> None:
        """Immediately abort all mouse automation."""
        log.warning("EMERGENCY STOP requested by user (Button or Esc key).")
        if self._worker:
            self._worker.emergency_stop()
        self._btn_mouse_enable.setEnabled(False)
        self._btn_mouse_disable.setEnabled(False)
        self._btn_reset_emergency.setEnabled(True)
        self._lbl_mouse_status.setText("🛑 EMERGENCY STOPPED")
        self._lbl_mouse_status.setStyleSheet(
            "background: #451212; color: #ff5252; font-weight: bold; font-size: 13px; "
            "padding: 6px; border-radius: 4px; border: 1px solid #ff1744;"
        )

    @Slot()
    def _on_reset_emergency_stop(self) -> None:
        """Reset emergency stop state back to clean disabled state."""
        log.info("Resetting emergency stop state.")
        if self._worker:
            self._worker.reset_emergency_stop()
        self._btn_reset_emergency.setEnabled(False)
        is_running = self._thread is not None and self._thread.isRunning()
        self._btn_mouse_enable.setEnabled(is_running and not self._settings.camera.use_mock_camera)
        self._btn_mouse_disable.setEnabled(False)
        self._lbl_mouse_status.setText("MOUSE CONTROL: DISABLED")
        self._lbl_mouse_status.setStyleSheet(
            "background: #222; color: #888; font-weight: bold; font-size: 13px; "
            "padding: 6px; border-radius: 4px; border: 1px solid #333;"
        )

    @Slot(int)
    def _on_sensitivity_slider_changed(self, value: int) -> None:
        sens = value / 10.0
        self._lbl_sens_val.setText(f"Sensitivity: {sens:.1f}x")
        self._settings.controller.cursor_sensitivity = sens
        if self._worker:
            self._worker.set_sensitivity(sens)

    @Slot(int)
    def _on_smoothing_slider_changed(self, value: int) -> None:
        smooth = value / 100.0
        self._lbl_smooth_val.setText(f"Smoothing: {smooth:.2f}")
        self._settings.controller.cursor_smoothing = smooth
        if self._worker:
            self._worker.set_smoothing(smooth)

    # ------------------------------------------------------------------
    # Slots from worker
    # ------------------------------------------------------------------

    @Slot(object)
    def _on_frame_ready(self, pipeline_frame: PipelineFrame) -> None:
        """Render a processed frame on the preview label and update dashboard metrics."""
        self._frame_count += 1
        bgr = pipeline_frame.display_image
        h, w, ch = bgr.shape
        rgb = bgr[:, :, ::-1].copy()  # BGR → RGB, contiguous

        q_img = QImage(rgb.data, w, h, w * ch, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        # Scale to fit the label while preserving aspect ratio.
        label_size = self._preview_label.size()
        scaled = pixmap.scaled(
            label_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        self._preview_label.setPixmap(scaled)

        # 1. Update FPS
        self._lbl_fps.setText(f"FPS: {pipeline_frame.fps:.1f}")

        # 2. Update Hand Tracking metrics
        n_hands = len(pipeline_frame.tracking.hands)
        self._lbl_hands.setText(f"Hands detected: {n_hands}")

        active_hand = pipeline_frame.active_hand
        if active_hand:
            self._lbl_active_hand.setText(f"Active Hand: {active_hand.handedness}")
            self._lbl_track_score.setText(f"Tracking Score: {active_hand.score * 100:.0f}%")
        else:
            self._lbl_active_hand.setText("Active Hand: None")
            self._lbl_track_score.setText("Tracking Score: —")

        # 3. Update Gesture HUD metrics
        g_res = pipeline_frame.gesture_result
        if g_res and active_hand:
            g_name = g_res.label.upper()
            self._lbl_gesture.setText(g_name)
            self._lbl_gesture_conf.setText(f"Confidence: {g_res.confidence * 100:.0f}%")
            status_text = "Stable (Confirmed)" if g_res.is_stable else "Confirming..."
            self._lbl_gesture_status.setText(f"Status: {status_text}")

            # Dynamic badge styling
            g_style = self._get_gesture_badge_style(g_res.gesture)
            self._lbl_gesture.setStyleSheet(g_style)
        else:
            self._lbl_gesture.setText("NO GESTURE")
            self._lbl_gesture.setStyleSheet(
                "background: #222; color: #888; font-weight: bold; font-size: 15px; "
                "padding: 8px; border-radius: 4px; border: 1px solid #333;"
            )
            self._lbl_gesture_conf.setText("Confidence: —")
            self._lbl_gesture_status.setText("Status: Searching...")

        # 4. Update Finger States
        f_states = pipeline_frame.finger_states
        if f_states:
            mapping = {
                "Thumb": f_states.thumb,
                "Index": f_states.index,
                "Middle": f_states.middle,
                "Ring": f_states.ring,
                "Pinky": f_states.pinky,
            }
            for fname, is_ext in mapping.items():
                lbl = self._lbl_finger_widgets.get(fname)
                if lbl:
                    if is_ext:
                        lbl.setText("EXTENDED")
                        lbl.setStyleSheet("color: #00e676; font-weight: bold;")
                    else:
                        lbl.setText("FOLDED")
                        lbl.setStyleSheet("color: #777; font-weight: normal;")
        else:
            for lbl in self._lbl_finger_widgets.values():
                lbl.setText("—")
                lbl.setStyleSheet("color: #555;")

        # 5. Update Virtual Mouse Telemetry
        m_state = getattr(pipeline_frame, "mouse_state", "DISABLED")
        coords = getattr(pipeline_frame, "cursor_coords", None)

        if m_state == "MOVING":
            self._lbl_mouse_status.setText("MOUSE: ACTIVE (MOVING)")
            self._lbl_mouse_status.setStyleSheet(
                "background: #143820; color: #00ff77; font-weight: bold; font-size: 13px; "
                "padding: 6px; border-radius: 4px; border: 1px solid #00e676;"
            )
            if coords:
                self._lbl_mouse_coords.setText(f"Cursor: ({coords[0]}, {coords[1]})")
        elif m_state == "EMERGENCY_STOPPED":
            self._lbl_mouse_status.setText("🛑 EMERGENCY STOPPED")
            self._lbl_mouse_status.setStyleSheet(
                "background: #451212; color: #ff5252; font-weight: bold; font-size: 13px; "
                "padding: 6px; border-radius: 4px; border: 1px solid #ff1744;"
            )
            self._btn_mouse_enable.setEnabled(False)
            self._btn_mouse_disable.setEnabled(False)
            self._btn_reset_emergency.setEnabled(True)
        elif m_state == "IDLE":
            self._lbl_mouse_status.setText("MOUSE: READY (POINT TO MOVE)")
            self._lbl_mouse_status.setStyleSheet(
                "background: #0d3840; color: #00e5ff; font-weight: bold; font-size: 13px; "
                "padding: 6px; border-radius: 4px; border: 1px solid #00b0ff;"
            )
        else:
            if hasattr(self, "_btn_reset_emergency") and not self._btn_reset_emergency.isEnabled():
                self._lbl_mouse_status.setText("MOUSE CONTROL: DISABLED")
                self._lbl_mouse_status.setStyleSheet(
                    "background: #222; color: #888; font-weight: bold; font-size: 13px; "
                    "padding: 6px; border-radius: 4px; border: 1px solid #333;"
                )

    @Slot(str)
    def _on_pipeline_error(self, message: str) -> None:
        log.error("Pipeline error: %s", message)
        self._lbl_status.setText(f"Error: {message[:60]}")
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        if self._worker:
            self._worker.stop_pipeline()
        if self._thread:
            self._thread.quit()
            self._thread.wait(2000)
            self._thread = None
        self._worker = None
        self._show_placeholder()
        QMessageBox.critical(self, "Pipeline Error", message)

    @Slot(str)
    def _on_status_changed(self, status: str) -> None:
        self._lbl_status.setText(status)

    def _get_gesture_badge_style(self, gesture) -> str:
        """Return CSS for gesture badge based on type."""
        from gestureflow.gestures.classifier import Gesture

        if gesture == Gesture.POINTING:
            return "background: #143820; color: #00ff77; font-weight: bold; font-size: 15px; padding: 8px; border-radius: 4px; border: 1px solid #00e676;"
        elif gesture in (Gesture.LEFT_PINCH, Gesture.RIGHT_PINCH):
            return "background: #2a3512; color: #ffee55; font-weight: bold; font-size: 15px; padding: 8px; border-radius: 4px; border: 1px solid #ffd600;"
        elif gesture == Gesture.OPEN_PALM:
            return "background: #301438; color: #ff77ee; font-weight: bold; font-size: 15px; padding: 8px; border-radius: 4px; border: 1px solid #e040fb;"
        elif gesture == Gesture.TWO_FINGER:
            return "background: #122b3b; color: #40c4ff; font-weight: bold; font-size: 15px; padding: 8px; border-radius: 4px; border: 1px solid #00b0ff;"
        elif gesture == Gesture.FIST:
            return "background: #3b2012; color: #ff9100; font-weight: bold; font-size: 15px; padding: 8px; border-radius: 4px; border: 1px solid #ff6d00;"
        return "background: #222; color: #888; font-weight: bold; font-size: 15px; padding: 8px; border-radius: 4px; border: 1px solid #333;"

    def _show_placeholder(self) -> None:
        self._preview_label.clear()
        self._preview_label.setText(
            '<span style="color:#555; font-size:14px;">📷 Click ▶ Start to begin</span>'
        )
        self._preview_label.setTextFormat(Qt.TextFormat.RichText)
        self._lbl_gesture.setText("NO GESTURE")
        self._lbl_gesture.setStyleSheet(
            "background: #222; color: #888; font-weight: bold; font-size: 15px; "
            "padding: 8px; border-radius: 4px; border: 1px solid #333;"
        )
        self._lbl_gesture_conf.setText("Confidence: —")
        self._lbl_gesture_status.setText("Status: Idle")
        self._lbl_hands.setText("Hands detected: 0")
        self._lbl_active_hand.setText("Active Hand: None")
        self._lbl_track_score.setText("Tracking Score: —")
        if hasattr(self, "_lbl_finger_widgets"):
            for lbl in self._lbl_finger_widgets.values():
                lbl.setText("—")
                lbl.setStyleSheet("color: #555;")
        if hasattr(self, "_lbl_mouse_status"):
            if not hasattr(self, "_btn_reset_emergency") or not self._btn_reset_emergency.isEnabled():
                self._lbl_mouse_status.setText("MOUSE CONTROL: DISABLED")
                self._lbl_mouse_status.setStyleSheet(
                    "background: #222; color: #888; font-weight: bold; font-size: 13px; "
                    "padding: 6px; border-radius: 4px; border: 1px solid #333;"
                )
        if hasattr(self, "_lbl_mouse_coords"):
            self._lbl_mouse_coords.setText("Cursor: (—, —)")

    def keyPressEvent(self, event) -> None:
        """Esc key stops automation while the app has keyboard focus."""
        if event.key() == Qt.Key.Key_Escape:
            self._on_emergency_stop()
            event.accept()
        else:
            super().keyPressEvent(event)

    def _apply_dark_theme(self) -> None:
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #1a1a1a; color: #e0e0e0; }
            QToolBar { background: #252525; border: none; padding: 4px 6px; spacing: 6px; }
            QPushButton#btn_start {
                background: #2a7a3a; color: #fff; border: none;
                border-radius: 4px; padding: 4px 14px; font-weight: bold;
            }
            QPushButton#btn_start:hover { background: #33944a; }
            QPushButton#btn_start:disabled { background: #333; color: #666; }
            QPushButton#btn_stop {
                background: #7a2a2a; color: #fff; border: none;
                border-radius: 4px; padding: 4px 14px; font-weight: bold;
            }
            QPushButton#btn_stop:hover { background: #943333; }
            QPushButton#btn_stop:disabled { background: #333; color: #666; }
            QGroupBox {
                border: 1px solid #333; border-radius: 4px;
                margin-top: 8px; padding-top: 6px;
                font-weight: bold; color: #aaa;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; }
            QStatusBar { background: #111; color: #aaa; font-size: 11px; }
            QSplitter::handle { background: #333; }
        """)

    # ------------------------------------------------------------------
    # Window lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        """Ensure the pipeline is stopped before the window closes."""
        log.info("Close event received — stopping pipeline if running.")
        if self._thread and self._thread.isRunning():
            self._on_stop()
        if self._log_handler:
            logging.getLogger().removeHandler(self._log_handler)
            self._log_handler = None
        event.accept()
