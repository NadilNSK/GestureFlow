"""
gestureflow.ui.pages.performance_page
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Real-time diagnostic and performance monitoring dashboard:
- Dual rolling time-series charts (FPS History & Pipeline Latency History)
- High-resolution stage latencies (Inference, Classification, Controller)
- End-to-end pipeline health rating (Optimal, Moderate, High Latency)
- Hardware stream telemetry (Camera index, DSHOW backend, resolution)
"""

from __future__ import annotations

from typing import Optional
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
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
from gestureflow.ui.components.cards import StatusCard, MetricCard, SectionHeader
from gestureflow.ui.components.charts import RollingMetricChart


class PerformancePage(QWidget):
    """
    Dedicated Performance Page for real-time latency and FPS diagnostics.
    """

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

        hdr = SectionHeader(
            "System Performance & Latency Monitor",
            "Real-time instrumentation of computer vision inference, gesture voting, and cursor dispatch",
        )
        layout.addWidget(hdr)

        # --------------------------------------------------------------
        # 1. Primary Metrics Row
        # --------------------------------------------------------------
        cards_row = QHBoxLayout()
        cards_row.setSpacing(10)

        self._card_pipeline_lat = StatusCard("Total Latency", "—", "Software end-to-end", "⏱", ACCENT_CYAN)
        self._card_frame_age = StatusCard("Frame Age", "—", "Capture-to-process", "⏳", ACCENT_CYAN)
        self._card_health = StatusCard("Pipeline Health", "OPTIMAL", "Latency threshold", "💚", ACCENT_EMERALD)
        self._card_fps = StatusCard("Capture / Proc", "— / —", "Target 30 FPS", "⚡", ACCENT_EMERALD)

        cards_row.addWidget(self._card_pipeline_lat)
        cards_row.addWidget(self._card_frame_age)
        cards_row.addWidget(self._card_health)
        cards_row.addWidget(self._card_fps)
        layout.addLayout(cards_row)

        # --------------------------------------------------------------
        # 2. Dual Rolling Charts
        # --------------------------------------------------------------
        charts_row = QHBoxLayout()
        charts_row.setSpacing(14)

        # Chart 1: FPS History (Camera vs Pipeline)
        self._chart_fps = RollingMetricChart(
            title="FPS History (Camera vs Pipeline)",
            series1_name="Cam FPS",
            series2_name="Proc FPS",
            series1_color=ACCENT_CYAN,
            series2_color=ACCENT_EMERALD,
            unit="",
            max_points=60,
            fixed_y_max=45.0,
            min_y_max=35.0,
        )
        charts_row.addWidget(self._chart_fps)

        # Chart 2: Latency History (Pipeline vs Inference)
        self._chart_latency = RollingMetricChart(
            title="Latency History (Pipeline vs Inference)",
            series1_name="Total",
            series2_name="Inference",
            series1_color=ACCENT_CYAN,
            series2_color="#60A5FA",
            unit="ms",
            max_points=60,
            fixed_y_max=None,
            min_y_max=40.0,
        )
        charts_row.addWidget(self._chart_latency)

        layout.addLayout(charts_row)

        # --------------------------------------------------------------
        # 3. Stage Breakdown & Environment Cards
        # --------------------------------------------------------------
        breakdown_row = QHBoxLayout()
        breakdown_row.setSpacing(14)

        # Left: Granular Latency Breakdown
        lat_box = QFrame()
        lat_box.setStyleSheet(f"""
            QFrame {{
                background-color: {CARD_BG};
                border: 1px solid {BORDER_COLOR};
                border-radius: 8px;
                padding: 12px;
            }}
        """)
        lat_layout = QVBoxLayout(lat_box)
        lat_layout.setContentsMargins(12, 10, 12, 10)
        lat_layout.setSpacing(8)

        lbl_lat_title = QLabel("Stage Latency Breakdown (Measured)")
        lbl_lat_title.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 13px; font-weight: 700;")
        lat_layout.addWidget(lbl_lat_title)

        self._metric_infer = MetricCard("1. MediaPipe Inference (GPU/CPU)", "—")
        self._metric_gest = MetricCard("2. Adaptive Gesture Classifier", "—")
        self._metric_ctrl = MetricCard("3. Virtual Mouse Controller", "—")
        self._metric_total = MetricCard("Total Software Processing", "—")

        lat_layout.addWidget(self._metric_infer)
        lat_layout.addWidget(self._metric_gest)
        lat_layout.addWidget(self._metric_ctrl)
        lat_layout.addWidget(self._metric_total)

        breakdown_row.addWidget(lat_box, stretch=1)

        # Right: Hardware & Stream Environment
        env_box = QFrame()
        env_box.setStyleSheet(f"""
            QFrame {{
                background-color: {CARD_BG};
                border: 1px solid {BORDER_COLOR};
                border-radius: 8px;
                padding: 12px;
            }}
        """)
        env_layout = QVBoxLayout(env_box)
        env_layout.setContentsMargins(12, 10, 12, 10)
        env_layout.setSpacing(8)

        lbl_env_title = QLabel("Hardware & Pipeline Configuration")
        lbl_env_title.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 13px; font-weight: 700;")
        env_layout.addWidget(lbl_env_title)

        cam_desc = (
            "Mock Camera (Simulated)"
            if self._settings.camera.use_mock_camera
            else f"Device {self._settings.camera.device_index} (DirectShow)"
        )
        self._metric_cam = MetricCard("Active Capture Device", cam_desc)
        self._metric_res = MetricCard(
            "Frame Resolution",
            f"{self._settings.camera.frame_width}×{self._settings.camera.frame_height} @ {self._settings.camera.fps} FPS",
        )
        self._metric_model = MetricCard("Model Architecture", "MediaPipe Hand Landmarker")
        self._metric_threading = MetricCard("Threading Model", "QThread Worker Pipeline")

        env_layout.addWidget(self._metric_cam)
        env_layout.addWidget(self._metric_res)
        env_layout.addWidget(self._metric_model)
        env_layout.addWidget(self._metric_threading)

        breakdown_row.addWidget(env_box, stretch=1)
        layout.addLayout(breakdown_row)

        # Footnote
        lbl_note = QLabel(
            "* Software latency is instrumented via high-precision time.perf_counter(). "
            "Sensor exposure and OS window manager composition latencies are independent."
        )
        lbl_note.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; font-style: italic;")
        layout.addWidget(lbl_note)
        layout.addStretch()

        scroll.setWidget(container)

        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.addWidget(scroll)

    def update_metrics(
        self,
        camera_fps: float,
        processing_fps: float,
        infer_ms: float,
        gest_ms: float,
        ctrl_ms: float,
        pipeline_ms: float,
        frame_age_ms: float,
    ) -> None:
        """Update telemetry cards and append points to the rolling charts."""
        # Update Status Cards
        self._card_pipeline_lat.set_value(f"{pipeline_ms:.1f}ms", ACCENT_CYAN)
        self._card_frame_age.set_value(f"{frame_age_ms:.1f}ms")
        self._card_fps.set_value(f"{camera_fps:.1f} / {processing_fps:.1f}")

        # Update Health Rating
        if pipeline_ms <= 35.0:
            self._card_health.set_value("OPTIMAL", ACCENT_EMERALD)
            self._card_health.set_subtitle("Smooth (< 35ms)", ACCENT_EMERALD)
        elif pipeline_ms <= 60.0:
            self._card_health.set_value("MODERATE", WARNING_AMBER)
            self._card_health.set_subtitle("Acceptable (< 60ms)", WARNING_AMBER)
        else:
            self._card_health.set_value("HIGH LATENCY", DANGER_RED)
            self._card_health.set_subtitle("Jitter risk (> 60ms)", DANGER_RED)

        # Update Granular Breakdown
        self._metric_infer.set_value(f"{infer_ms:.2f} ms")
        self._metric_gest.set_value(f"{gest_ms:.2f} ms")
        self._metric_ctrl.set_value(f"{ctrl_ms:.2f} ms")
        self._metric_total.set_value(f"{pipeline_ms:.2f} ms", ACCENT_CYAN)

        # Update Charts
        self._chart_fps.add_data_point(camera_fps, processing_fps)
        self._chart_latency.add_data_point(pipeline_ms, infer_ms)
