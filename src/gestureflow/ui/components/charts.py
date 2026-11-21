"""
gestureflow.ui.components.charts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Lightweight, native PySide6 rolling metric charts using QPainter.
Renders smooth time-series graphs for FPS and pipeline latency with:
- Zero external charting dependencies (pure Qt QPainter)
- Bounded history buffers (collections.deque)
- Responsive dark styling and accessible accent colors
- Throttled paint scheduling to guarantee zero pipeline lag
"""

from __future__ import annotations

import collections
import time
from typing import Optional

from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import (
    QPainter,
    QPen,
    QColor,
    QFont,
    QPainterPath,
    QLinearGradient,
)
from PySide6.QtWidgets import QWidget, QSizePolicy

from gestureflow.ui.components.theme import (
    CARD_BG,
    BORDER_COLOR,
    ACCENT_EMERALD,
    ACCENT_CYAN,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_MUTED,
)


class RollingMetricChart(QWidget):
    """
    Lightweight, dual-series rolling time-series chart with bounded buffer.
    """

    def __init__(
        self,
        title: str,
        series1_name: str,
        series2_name: str,
        series1_color: str = ACCENT_EMERALD,
        series2_color: str = ACCENT_CYAN,
        unit: str = "",
        max_points: int = 60,
        fixed_y_max: Optional[float] = None,
        min_y_max: float = 10.0,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._title = title
        self._series1_name = series1_name
        self._series2_name = series2_name
        self._color1 = QColor(series1_color)
        self._color2 = QColor(series2_color)
        self._unit = unit
        self._max_points = max_points
        self._fixed_y_max = fixed_y_max
        self._min_y_max = min_y_max

        self._data1: collections.deque[float] = collections.deque(maxlen=max_points)
        self._data2: collections.deque[float] = collections.deque(maxlen=max_points)

        self._last_val1 = 0.0
        self._last_val2 = 0.0
        self._last_repaint_time = 0.0
        self._min_repaint_interval = 0.060  # ~16 FPS repaint limit for UI responsiveness

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(150)
        self.setMinimumWidth(260)

    def add_data_point(self, val1: float, val2: float) -> None:
        """Append a data sample to both rolling buffers and trigger a throttled update."""
        v1 = max(0.0, float(val1))
        v2 = max(0.0, float(val2))
        self._data1.append(v1)
        self._data2.append(v2)
        self._last_val1 = v1
        self._last_val2 = v2

        now = time.monotonic()
        if (now - self._last_repaint_time) >= self._min_repaint_interval:
            self._last_repaint_time = now
            self.update()

    def clear(self) -> None:
        """Clear all historical data."""
        self._data1.clear()
        self._data2.clear()
        self._last_val1 = 0.0
        self._last_val2 = 0.0
        self.update()

    def paintEvent(self, event) -> None:
        """Render grid, curves, labels, and legend."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        # Background and card frame
        painter.setBrush(QColor(CARD_BG))
        painter.setPen(QPen(QColor(BORDER_COLOR), 1))
        painter.drawRoundedRect(0, 0, w - 1, h - 1, 8, 8)

        # Margins for title, axes, and legend
        left_m = 45
        right_m = 15
        top_m = 32
        bottom_m = 25

        plot_w = max(10, w - left_m - right_m)
        plot_h = max(10, h - top_m - bottom_m)

        # Draw Title & Legend
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.setPen(QColor(TEXT_PRIMARY))
        painter.drawText(12, 22, self._title)

        # Legend (Top Right)
        legend_x = w - 180
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Medium))

        # Series 1 dot & text
        painter.setBrush(self._color1)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(legend_x, 12, 8, 8)
        painter.setPen(QColor(TEXT_SECONDARY))
        txt1 = f"{self._series1_name}: {self._last_val1:.1f}{self._unit}"
        painter.drawText(legend_x + 12, 20, txt1)

        # Series 2 dot & text
        painter.setBrush(self._color2)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(legend_x + 95, 12, 8, 8)
        painter.setPen(QColor(TEXT_SECONDARY))
        txt2 = f"{self._series2_name}: {self._last_val2:.1f}{self._unit}"
        painter.drawText(legend_x + 107, 20, txt2)

        # Determine Y scale
        if self._fixed_y_max is not None:
            max_y = self._fixed_y_max
        else:
            all_vals = list(self._data1) + list(self._data2)
            peak = max(all_vals) if all_vals else 0.0
            max_y = max(self._min_y_max, peak * 1.15)

        # Draw subtle gridlines
        grid_pen = QPen(QColor(BORDER_COLOR), 1, Qt.PenStyle.DashLine)
        axis_pen = QPen(QColor(TEXT_MUTED), 1)
        painter.setFont(QFont("Segoe UI", 8))

        steps = 3
        for i in range(steps + 1):
            y_pos = top_m + plot_h - int((i / steps) * plot_h)
            val_tick = (i / steps) * max_y
            painter.setPen(grid_pen)
            painter.drawLine(left_m, y_pos, left_m + plot_w, y_pos)

            painter.setPen(axis_pen)
            painter.drawText(6, y_pos + 4, f"{val_tick:.0f}")

        # Render curves if we have points
        n = max(len(self._data1), len(self._data2))
        if n < 2:
            painter.setPen(QColor(TEXT_MUTED))
            painter.setFont(QFont("Segoe UI", 9))
            painter.drawText(
                left_m + plot_w // 2 - 50,
                top_m + plot_h // 2,
                "Awaiting data...",
            )
            return

        def _draw_series(data: collections.deque[float], color: QColor):
            if len(data) < 2:
                return
            points: list[QPointF] = []
            dx = plot_w / (self._max_points - 1)
            offset_idx = self._max_points - len(data)

            for i, val in enumerate(data):
                px = left_m + (offset_idx + i) * dx
                clamped_val = min(val, max_y)
                py = top_m + plot_h - (clamped_val / max_y * plot_h)
                points.append(QPointF(px, py))

            path = QPainterPath()
            path.moveTo(points[0])
            for pt in points[1:]:
                path.lineTo(pt)

            # Draw smooth stroke
            pen = QPen(color, 2)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)

            # Subtle filled gradient under curve
            fill_path = QPainterPath(path)
            fill_path.lineTo(points[-1].x(), top_m + plot_h)
            fill_path.lineTo(points[0].x(), top_m + plot_h)
            fill_path.closeSubpath()

            grad = QLinearGradient(0, top_m, 0, top_m + plot_h)
            grad.setColorAt(0.0, QColor(color.red(), color.green(), color.blue(), 40))
            grad.setColorAt(1.0, QColor(color.red(), color.green(), color.blue(), 0))
            painter.fillPath(fill_path, grad)

        # Draw both series
        _draw_series(self._data2, self._color2)
        _draw_series(self._data1, self._color1)
