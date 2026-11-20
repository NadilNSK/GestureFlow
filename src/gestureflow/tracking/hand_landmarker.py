"""
gestureflow.tracking.hand_landmarker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Wraps the MediaPipe Tasks **Hand Landmarker** API (not the deprecated
mp.solutions.hands).

References
----------
- https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np

from gestureflow.config.settings import TrackingSettings

log = logging.getLogger(__name__)

# Convenience aliases
_BaseOptions = mp_python.BaseOptions
_HandLandmarker = mp_vision.HandLandmarker
_HandLandmarkerOptions = mp_vision.HandLandmarkerOptions
_RunningMode = mp_vision.RunningMode

# MediaPipe defines 21 hand landmarks (indices 0-20).
NUM_LANDMARKS = 21


@dataclass
class HandLandmarks:
    """
    Normalised landmarks for a single detected hand.

    Each landmark has ``x``, ``y`` ∈ [0, 1] relative to the frame size,
    and ``z`` (depth, relative to the wrist).

    ``landmarks`` and ``world_landmarks`` are lists of mediapipe landmark
    objects; exact types depend on the installed mediapipe version.
    """

    landmarks: list[Any]        # NormalizedLandmark objects
    world_landmarks: list[Any]  # Landmark (world-space) objects
    handedness: str             # "Left" or "Right"
    score: float                # Detection confidence


@dataclass
class TrackingResult:
    """Result returned by ``HandLandmarkerWrapper.detect()``."""

    hands: list[HandLandmarks] = field(default_factory=list)
    timestamp_ms: int = 0

    @property
    def has_hands(self) -> bool:
        return len(self.hands) > 0

    @property
    def primary_hand(self) -> Optional[HandLandmarks]:
        """Return the first detected hand, or None."""
        return self.hands[0] if self.hands else None


class HandLandmarkerWrapper:
    """
    Thin wrapper around the MediaPipe Tasks Hand Landmarker for
    **video-stream** (live camera) mode.

    Parameters
    ----------
    settings:
        ``TrackingSettings`` from the application config.
    model_path:
        Absolute path to the ``hand_landmarker.task`` model file.
    """

    def __init__(self, settings: TrackingSettings, model_path: Path) -> None:
        self._settings = settings
        self._model_path = model_path
        self._landmarker: _HandLandmarker | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Load the model and create the HandLandmarker (IMAGE mode)."""
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Hand Landmarker model not found: {self._model_path}\n"
                "Run  scripts\\download_model.py  to fetch it."
            )

        base_options = _BaseOptions(model_asset_path=str(self._model_path))
        options = _HandLandmarkerOptions(
            base_options=base_options,
            running_mode=_RunningMode.IMAGE,
            num_hands=self._settings.num_hands,
            min_hand_detection_confidence=self._settings.min_detection_confidence,
            min_hand_presence_confidence=self._settings.min_presence_confidence,
            min_tracking_confidence=self._settings.min_tracking_confidence,
        )
        self._landmarker = _HandLandmarker.create_from_options(options)
        log.info("HandLandmarker loaded from %s", self._model_path)

    def close(self) -> None:
        """Release the HandLandmarker resources."""
        if self._landmarker:
            try:
                self._landmarker.close()
            except Exception as exc:
                log.debug("HandLandmarker close exception: %s", exc)
            finally:
                self._landmarker = None
            log.info("HandLandmarker closed.")
            time.sleep(0.05)

    def __enter__(self) -> "HandLandmarkerWrapper":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(self, bgr_frame: np.ndarray, timestamp_ms: int) -> TrackingResult:
        """
        Run hand landmark detection on a single BGR frame.

        Parameters
        ----------
        bgr_frame:
            NumPy array in BGR format (as returned by OpenCV).
        timestamp_ms:
            Frame timestamp in milliseconds.

        Returns
        -------
        TrackingResult
            Contains zero or more detected hands.
        """
        if self._landmarker is None:
            raise RuntimeError("HandLandmarkerWrapper.open() must be called first.")

        # MediaPipe Tasks expects contiguous RGB.
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        try:
            detection = self._landmarker.detect(mp_image)
        except RuntimeError as exc:
            if "cannot schedule new futures after shutdown" in str(exc) or "shutdown" in str(exc):
                log.debug("MediaPipe executor shut down — returning empty result.")
                return TrackingResult(hands=[], timestamp_ms=timestamp_ms)
            raise

        hands: list[HandLandmarks] = []
        for idx, norm_lm_list in enumerate(detection.hand_landmarks):
            handedness_list = (
                detection.handedness[idx] if idx < len(detection.handedness) else []
            )
            world_lm_list = (
                detection.hand_world_landmarks[idx]
                if idx < len(detection.hand_world_landmarks)
                else []
            )

            category = handedness_list[0] if handedness_list else None
            label = (
                (category.display_name or category.category_name or f"Hand {idx + 1}")
                if category
                else f"Hand {idx + 1}"
            )
            score = category.score if category else 1.0
            hand = HandLandmarks(
                landmarks=list(norm_lm_list),
                world_landmarks=list(world_lm_list),
                handedness=label,
                score=score,
            )
            hands.append(hand)

        return TrackingResult(hands=hands, timestamp_ms=timestamp_ms)
