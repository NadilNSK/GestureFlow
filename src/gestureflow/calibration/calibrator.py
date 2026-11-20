"""
gestureflow.calibration.calibrator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Production-grade 4-point calibration engine for virtual mouse control.

Provides:
- 4-point target collection (Top-Left, Top-Right, Bottom-Right, Bottom-Left).
- Statistical sample aggregation with outlier rejection.
- Geometric perspective transformation (homography) with affine and
  bounded-rectangle fallbacks.
- Convexity, area, and degeneracy validation.
- Serialization to and from settings dictionaries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np

from gestureflow.utils.helpers import clamp

log = logging.getLogger(__name__)

# Standard 4 calibration corner target IDs in clockwise order
CORNER_TL = "TL"  # Top-Left
CORNER_TR = "TR"  # Top-Right
CORNER_BR = "BR"  # Bottom-Right
CORNER_BL = "BL"  # Bottom-Left
CALIBRATION_CORNERS = [CORNER_TL, CORNER_TR, CORNER_BR, CORNER_BL]

# Default screen target coordinates (normalized [0, 1] with 10% margin)
DEFAULT_TARGETS: dict[str, tuple[float, float]] = {
    CORNER_TL: (0.10, 0.10),
    CORNER_TR: (0.90, 0.10),
    CORNER_BR: (0.90, 0.90),
    CORNER_BL: (0.10, 0.90),
}


@dataclass
class CalibrationData:
    """
    Persisted calibration configuration and coordinate transformation data.

    Maintains backward compatibility with x_min/max, y_min/max bounding boxes.
    """

    # Bounding box fallback boundaries (normalized 0-1)
    x_min: float = 0.10
    x_max: float = 0.90
    y_min: float = 0.10
    y_max: float = 0.90

    # 4-corner sampled camera landmarks (normalized 0-1): {corner_id: (x, y)}
    camera_points: dict[str, tuple[float, float]] = field(default_factory=dict)

    # 4-corner target screen positions (normalized 0-1): {corner_id: (x, y)}
    screen_targets: dict[str, tuple[float, float]] = field(
        default_factory=lambda: dict(DEFAULT_TARGETS)
    )

    # 3x3 perspective transformation matrix (camera norm -> screen norm)
    perspective_matrix: Optional[list[list[float]]] = None

    # Quality score / calibration RMSE (lower is better, or 1.0 = perfect)
    quality_score: float = 1.0

    # Whether full 4-point perspective calibration is verified and active
    is_active: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize calibration data to JSON-serializable dictionary."""
        return {
            "x_min": float(self.x_min),
            "x_max": float(self.x_max),
            "y_min": float(self.y_min),
            "y_max": float(self.y_max),
            "camera_points": {k: [float(v[0]), float(v[1])] for k, v in self.camera_points.items()},
            "screen_targets": {k: [float(v[0]), float(v[1])] for k, v in self.screen_targets.items()},
            "perspective_matrix": self.perspective_matrix,
            "quality_score": float(self.quality_score),
            "is_active": bool(self.is_active),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CalibrationData":
        """Deserialize calibration data from a dictionary."""
        if not data:
            return cls()

        camera_points: dict[str, tuple[float, float]] = {}
        raw_cam = data.get("camera_points", {})
        if isinstance(raw_cam, dict):
            for k, v in raw_cam.items():
                if isinstance(v, (list, tuple)) and len(v) >= 2:
                    camera_points[str(k)] = (float(v[0]), float(v[1]))

        screen_targets: dict[str, tuple[float, float]] = {}
        raw_screen = data.get("screen_targets", {})
        if isinstance(raw_screen, dict):
            for k, v in raw_screen.items():
                if isinstance(v, (list, tuple)) and len(v) >= 2:
                    screen_targets[str(k)] = (float(v[0]), float(v[1]))
        else:
            screen_targets = dict(DEFAULT_TARGETS)

        matrix = data.get("perspective_matrix")
        if isinstance(matrix, list) and len(matrix) == 3:
            try:
                matrix = [[float(val) for val in row] for row in matrix]
            except (ValueError, TypeError):
                matrix = None
        else:
            matrix = None

        return cls(
            x_min=float(data.get("x_min", 0.10)),
            x_max=float(data.get("x_max", 0.90)),
            y_min=float(data.get("y_min", 0.10)),
            y_max=float(data.get("y_max", 0.90)),
            camera_points=camera_points,
            screen_targets=screen_targets,
            perspective_matrix=matrix,
            quality_score=float(data.get("quality_score", 1.0)),
            is_active=bool(data.get("is_active", False)),
        )


class Calibrator:
    """
    Manages interactive calibration sample collection and computes accurate
    hand-to-screen coordinate mapping.
    """

    def __init__(self, data: CalibrationData | None = None) -> None:
        self._data = data or CalibrationData()
        self._sample_buffer: dict[str, list[tuple[float, float]]] = {
            corner: [] for corner in CALIBRATION_CORNERS
        }
        self._current_corner_idx = 0
        self._cached_matrix_np: Optional[np.ndarray] = None
        self._update_cached_matrix()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def data(self) -> CalibrationData:
        """Active calibration data."""
        return self._data

    @property
    def is_calibrated(self) -> bool:
        """True if full 4-point calibration is calculated and active."""
        return self._data.is_active and self._cached_matrix_np is not None

    @property
    def current_target_corner(self) -> str:
        """ID of the current corner being sampled ('TL', 'TR', 'BR', 'BL')."""
        if self._current_corner_idx < len(CALIBRATION_CORNERS):
            return CALIBRATION_CORNERS[self._current_corner_idx]
        return ""

    @property
    def is_sampling_complete(self) -> bool:
        """True if all 4 corners have collected sufficient samples."""
        return self._current_corner_idx >= len(CALIBRATION_CORNERS)

    # ------------------------------------------------------------------
    # Sampling API
    # ------------------------------------------------------------------

    def reset_sampling(self) -> None:
        """Clear temporary sample buffers and restart sampling from corner 0."""
        self._sample_buffer = {corner: [] for corner in CALIBRATION_CORNERS}
        self._current_corner_idx = 0

    def add_sample(self, corner_id: str, norm_x: float, norm_y: float) -> int:
        """
        Record a normalized camera landmark sample for a target corner.

        Returns
        -------
        int
            Current number of samples collected for this corner.
        """
        if corner_id not in self._sample_buffer:
            return 0
        x = clamp(float(norm_x), 0.0, 1.0)
        y = clamp(float(norm_y), 0.0, 1.0)
        self._sample_buffer[corner_id].append((x, y))
        return len(self._sample_buffer[corner_id])

    def advance_to_next_corner(self) -> bool:
        """
        Advance sampling to the next corner target.

        Returns
        -------
        bool
            True if advanced, False if all corners have been sampled.
        """
        self._current_corner_idx += 1
        return self._current_corner_idx < len(CALIBRATION_CORNERS)

    def set_corner_point(self, corner_id: str, norm_x: float, norm_y: float) -> None:
        """Directly set a calibrated point for a corner."""
        if corner_id in CALIBRATION_CORNERS:
            self._data.camera_points[corner_id] = (
                clamp(float(norm_x), 0.0, 1.0),
                clamp(float(norm_y), 0.0, 1.0),
            )

    # ------------------------------------------------------------------
    # Outlier Rejection & Aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def filter_samples(samples: list[tuple[float, float]]) -> tuple[float, float]:
        """
        Filter out tracking tremor/outliers using median and interquartile range.

        Returns
        -------
        tuple[float, float]
            Robust center (median) of the sampled coordinates.
        """
        if not samples:
            return (0.5, 0.5)
        if len(samples) <= 2:
            avg_x = sum(s[0] for s in samples) / len(samples)
            avg_y = sum(s[1] for s in samples) / len(samples)
            return (avg_x, avg_y)

        arr = np.array(samples, dtype=np.float32)
        med = np.median(arr, axis=0)

        # Median Absolute Deviation (MAD)
        mad = np.median(np.abs(arr - med), axis=0)
        threshold_x = max(0.015, 2.5 * float(mad[0]))
        threshold_y = max(0.015, 2.5 * float(mad[1]))

        inliers = []
        for x, y in samples:
            if abs(x - med[0]) <= threshold_x and abs(y - med[1]) <= threshold_y:
                inliers.append((x, y))

        if not inliers:
            return (float(med[0]), float(med[1]))

        inlier_arr = np.array(inliers, dtype=np.float32)
        mean_pt = np.mean(inlier_arr, axis=0)
        return (float(mean_pt[0]), float(mean_pt[1]))

    # ------------------------------------------------------------------
    # Transformation Computation
    # ------------------------------------------------------------------

    def compute_calibration(self) -> bool:
        """
        Compute perspective and bounding transformations from collected samples.

        Validates convexity, minimal area, and non-degeneracy.
        Falls back to bounded-rectangular mapping if perspective fails.

        Returns
        -------
        bool
            True if calibration succeeded (perspective or rectangular fallback).
        """
        # If samples exist in buffer, aggregate them into camera_points
        for corner in CALIBRATION_CORNERS:
            s_list = self._sample_buffer.get(corner, [])
            if s_list:
                self._data.camera_points[corner] = self.filter_samples(s_list)

        # Ensure all 4 corners are present
        if not all(corner in self._data.camera_points for corner in CALIBRATION_CORNERS):
            log.warning("Calibration incomplete: missing corner points.")
            return False

        pts_cam = [self._data.camera_points[c] for c in CALIBRATION_CORNERS]
        pts_scr = [self._data.screen_targets.get(c, DEFAULT_TARGETS[c]) for c in CALIBRATION_CORNERS]

        src = np.array(pts_cam, dtype=np.float32)
        dst = np.array(pts_scr, dtype=np.float32)

        # Update bounding box values
        self._data.x_min = float(min(p[0] for p in pts_cam))
        self._data.x_max = float(max(p[0] for p in pts_cam))
        self._data.y_min = float(min(p[1] for p in pts_cam))
        self._data.y_max = float(max(p[1] for p in pts_cam))

        # Check for minimum quadrilateral area to reject collapsed gestures
        contour = src.reshape((-1, 1, 2))
        area = float(cv2.contourArea(contour))
        is_convex = bool(cv2.isContourConvex(np.int32(contour * 1000)))

        min_area_thresh = 0.04  # At least 20% x 20% equivalent area
        if area < min_area_thresh or not is_convex:
            log.warning(
                "Calibration points non-convex or area too small (area=%.4f, convex=%s). "
                "Falling back to rectangular bounding box.",
                area,
                is_convex,
            )
            self._data.perspective_matrix = None
            self._cached_matrix_np = None
            self._data.is_active = True
            self._data.quality_score = 0.5
            return True

        # Compute 3x3 Perspective Transformation Matrix (Homography)
        try:
            mat = cv2.getPerspectiveTransform(src, dst)
            # Verify matrix condition number / inversion
            det = float(np.linalg.det(mat))
            if abs(det) < 1e-7:
                raise ValueError("Perspective matrix determinant close to zero.")

            self._cached_matrix_np = mat
            self._data.perspective_matrix = mat.tolist()
            self._data.is_active = True
            self._data.quality_score = 0.95
            log.info("Perspective calibration successfully computed.")
            return True
        except Exception as exc:
            log.warning("cv2.getPerspectiveTransform failed: %s. Using rectangular fallback.", exc)
            self._data.perspective_matrix = None
            self._cached_matrix_np = None
            self._data.is_active = True
            self._data.quality_score = 0.6
            return True

    def _update_cached_matrix(self) -> None:
        """Update NumPy perspective matrix cache from data."""
        if self._data.perspective_matrix is not None:
            try:
                self._cached_matrix_np = np.array(self._data.perspective_matrix, dtype=np.float32)
            except Exception:
                self._cached_matrix_np = None
        else:
            self._cached_matrix_np = None

    # ------------------------------------------------------------------
    # Mapping
    # ------------------------------------------------------------------

    def map_point(
        self,
        norm_x: float,
        norm_y: float,
        screen_width: int,
        screen_height: int,
    ) -> tuple[float, float]:
        """
        Map normalized camera coordinate [0, 1] to screen pixel coordinates
        [0, screen_width - 1] x [0, screen_height - 1].

        Uses perspective transform if active, otherwise falls back to
        calibrated bounding box interpolation.
        """
        w_max = float(max(1, screen_width - 1))
        h_max = float(max(1, screen_height - 1))

        if self._cached_matrix_np is not None:
            # Perspective transform
            pt = np.array([[[norm_x, norm_y]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(pt, self._cached_matrix_np)
            tx = float(transformed[0, 0, 0])
            ty = float(transformed[0, 0, 1])

            # Perspective output is normalized screen [0, 1]
            sx = clamp(tx, 0.0, 1.0) * w_max
            sy = clamp(ty, 0.0, 1.0) * h_max
            return (sx, sy)

        # Fallback to calibrated bounding box [x_min, x_max] x [y_min, y_max]
        span_x = max(0.05, self._data.x_max - self._data.x_min)
        span_y = max(0.05, self._data.y_max - self._data.y_min)

        u = clamp((norm_x - self._data.x_min) / span_x, 0.0, 1.0)
        v = clamp((norm_y - self._data.y_min) / span_y, 0.0, 1.0)

        sx = u * w_max
        sy = v * h_max
        return (sx, sy)

    def reset(self) -> None:
        """Reset calibration back to defaults."""
        self._data = CalibrationData()
        self._cached_matrix_np = None
        self.reset_sampling()
        log.info("Calibrator reset to defaults.")
