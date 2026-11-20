"""
gestureflow.gestures.classifier
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Phase 2A: Robust rule-based finger-state detection and gesture classification.

Recognises the following core gestures:
  - POINTING: Index finger extended, others folded
  - LEFT_PINCH: Thumb and index fingertips close (touching)
  - RIGHT_PINCH: Thumb and middle fingertips close (touching)
  - TWO_FINGER: Index and middle fingers extended, others folded
  - OPEN_PALM: All five fingers extended
  - FIST: All five fingers folded
  - UNKNOWN: No reliable matching gesture

Features:
  - Full 3D rotation-invariance via joint angles and palm-normalized distance ratios.
  - Hysteresis on pinch enter vs pinch release distances to avoid jitter.
  - Multi-frame confirmation and debouncing for rock-solid stability.
  - Graceful handling of temporary tracking loss.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional, Sequence

import numpy as np

from gestureflow.config.settings import GestureSettings
from gestureflow.tracking.hand_landmarker import HandLandmarks

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and Dataclasses
# ---------------------------------------------------------------------------

class Gesture(Enum):
    """Recognised gesture types."""

    UNKNOWN = auto()
    POINTING = auto()
    LEFT_PINCH = auto()
    RIGHT_PINCH = auto()
    TWO_FINGER = auto()
    OPEN_PALM = auto()
    FIST = auto()

    # Aliases for backward compatibility with Phase 1 tests/code
    NONE = UNKNOWN
    POINT = POINTING
    CLICK = LEFT_PINCH
    RIGHT_CLICK = RIGHT_PINCH


@dataclass
class FingerStates:
    """Extension state of each of the five fingers."""

    thumb: bool = False
    index: bool = False
    middle: bool = False
    ring: bool = False
    pinky: bool = False

    def as_dict(self) -> dict[str, bool]:
        return {
            "Thumb": self.thumb,
            "Index": self.index,
            "Middle": self.middle,
            "Ring": self.ring,
            "Pinky": self.pinky,
        }

    def as_tuple(self) -> tuple[bool, bool, bool, bool, bool]:
        return (self.thumb, self.index, self.middle, self.ring, self.pinky)

    def count_extended(self) -> int:
        return sum(self.as_tuple())

    def summary_str(self) -> str:
        t = "T:+" if self.thumb else "T:-"
        i = "I:+" if self.index else "I:-"
        m = "M:+" if self.middle else "M:-"
        r = "R:+" if self.ring else "R:-"
        p = "P:+" if self.pinky else "P:-"
        return f"{t} {i} {m} {r} {p}"


@dataclass
class GestureResult:
    """Complete output of the gesture classifier."""

    gesture: Gesture = Gesture.UNKNOWN
    confidence: float = 0.0
    label: str = "Unknown"
    finger_states: FingerStates = field(default_factory=FingerStates)
    handedness: str = "Unknown"
    pinch_distance: float = 0.0
    is_stable: bool = False


# ---------------------------------------------------------------------------
# Geometric Helpers (Rotation-Invariant)
# ---------------------------------------------------------------------------

def _to_numpy_coords(landmarks: Any) -> np.ndarray:
    """
    Extract a (21, 3) float32 numpy array from landmarks.

    Accepts:
      - MediaPipe NormalizedLandmark lists / objects
      - numpy ndarray of shape (21, 3) or (21, 2)
      - list of dicts with 'x', 'y', optional 'z'
      - list of tuples (x, y) or (x, y, z)
    """
    if isinstance(landmarks, np.ndarray):
        if landmarks.ndim == 2 and landmarks.shape[0] == 21:
            if landmarks.shape[1] == 3:
                return landmarks.astype(np.float32)
            elif landmarks.shape[1] == 2:
                z = np.zeros((21, 1), dtype=np.float32)
                return np.hstack([landmarks.astype(np.float32), z])

    coords = np.zeros((21, 3), dtype=np.float32)
    for idx, lm in enumerate(landmarks):
        if idx >= 21:
            break
        if hasattr(lm, "x") and hasattr(lm, "y"):
            coords[idx, 0] = lm.x
            coords[idx, 1] = lm.y
            coords[idx, 2] = getattr(lm, "z", 0.0)
        elif isinstance(lm, dict):
            coords[idx, 0] = lm.get("x", 0.0)
            coords[idx, 1] = lm.get("y", 0.0)
            coords[idx, 2] = lm.get("z", 0.0)
        elif isinstance(lm, (list, tuple)):
            coords[idx, 0] = lm[0]
            coords[idx, 1] = lm[1]
            coords[idx, 2] = lm[2] if len(lm) > 2 else 0.0
        else:
            raise ValueError(f"Invalid landmark object at index {idx}: {type(lm)}")

    return coords


def _compute_palm_scale(p: np.ndarray) -> float:
    """
    Compute reference palm scale invariant to distance and hand orientation.

    Distance between Wrist (0) and Middle MCP (9), with fallback to
    Index MCP (5) to Pinky MCP (17).
    """
    d09 = float(np.linalg.norm(p[9] - p[0]))
    if d09 > 1e-4:
        return d09
    d517 = float(np.linalg.norm(p[17] - p[5]))
    return d517 if d517 > 1e-4 else 1.0


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Compute interior angle at joint `b` between vectors `ba` and `bc` in degrees [0, 180].

    180° = fully straight joint.
    < 120° = distinctly bent joint.
    """
    ba = a - b
    bc = c - b
    len_ba = np.linalg.norm(ba)
    len_bc = np.linalg.norm(bc)
    if len_ba < 1e-7 or len_bc < 1e-7:
        return 180.0
    cosine = np.dot(ba, bc) / (len_ba * len_bc)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


# ---------------------------------------------------------------------------
# Finger State Detection
# ---------------------------------------------------------------------------

def detect_finger_states(landmarks: Any, handedness: str = "Right") -> FingerStates:
    """
    Detect whether each of the 5 fingers is extended or folded.

    Uses:
      - Internal joint angles (PIP and DIP angles)
      - Relative Euclidean distance from Wrist and MCP joints
      - Palm-normalized distances

    Avoids relying on fingertip Y coordinates to ensure full 3D rotation invariance.

    Parameters
    ----------
    landmarks:
        21 hand landmarks (MediaPipe objects, array, or list).
    handedness:
        "Right" or "Left" (used for thumb geometry refinement).

    Returns
    -------
    FingerStates
    """
    p = _to_numpy_coords(landmarks)
    palm_scale = _compute_palm_scale(p)

    # 1. Non-thumb finger definitions: (MCP, PIP, DIP, TIP)
    finger_joints = {
        "index": (5, 6, 7, 8),
        "middle": (9, 10, 11, 12),
        "ring": (13, 14, 15, 16),
        "pinky": (17, 18, 19, 20),
    }

    extended_states = {}

    for name, (mcp_i, pip_i, dip_i, tip_i) in finger_joints.items():
        # Joint angles
        ang_pip = joint_angle(p[mcp_i], p[pip_i], p[dip_i])
        ang_dip = joint_angle(p[pip_i], p[dip_i], p[tip_i])

        # Distance from wrist (0)
        d_wrist_tip = float(np.linalg.norm(p[tip_i] - p[0]))
        d_wrist_pip = float(np.linalg.norm(p[pip_i] - p[0]))
        d_wrist_mcp = float(np.linalg.norm(p[mcp_i] - p[0]))

        # Distance from MCP
        d_mcp_tip = float(np.linalg.norm(p[tip_i] - p[mcp_i]))
        d_mcp_pip = float(np.linalg.norm(p[pip_i] - p[mcp_i]))

        # Straightness condition
        is_straight = (ang_pip > 135.0) and (ang_dip > 125.0)

        # Distance extension condition
        is_extended_dist = (d_wrist_tip > d_wrist_pip) and (d_mcp_tip > 1.15 * d_mcp_pip)

        # A finger is extended when both angles and distance metrics agree,
        # or when strongly extended beyond the wrist and MCP.
        is_extended = (is_straight and is_extended_dist) or (
            d_wrist_tip > 1.25 * d_wrist_pip and ang_pip > 120.0
        )
        extended_states[name] = bool(is_extended)

    # 2. Thumb finger detection (0: Wrist, 1: CMC, 2: MCP, 3: IP, 4: TIP)
    ang_thumb_ip = joint_angle(p[2], p[3], p[4])
    ang_thumb_mcp = joint_angle(p[1], p[2], p[3])

    # Distance of thumb tip to Pinky MCP (17) and Index MCP (5)
    d_thumb_pinky = float(np.linalg.norm(p[4] - p[17])) / palm_scale
    d_thumb_index_mcp = float(np.linalg.norm(p[4] - p[5])) / palm_scale
    d_thumb_mcp_tip = float(np.linalg.norm(p[4] - p[2])) / palm_scale

    # When thumb is folded into palm:
    # - It tucks inward close to Pinky MCP (17) or Ring MCP (13)
    # - Or thumb tip is very close to Index MCP (5)
    # When extended (abducted):
    # - Thumb tip is far from Pinky MCP (> 0.52 * palm_scale)
    # - Thumb IP joint is not curled tightly (< 120°)
    # - Thumb tip is extended from its MCP (> 0.35 * palm_scale)
    is_thumb_folded = (
        (d_thumb_pinky < 0.48)
        or (d_thumb_index_mcp < 0.28 and ang_thumb_ip < 140.0)
        or (ang_thumb_ip < 120.0 and d_thumb_mcp_tip < 0.30)
    )
    thumb_extended = not is_thumb_folded and (d_thumb_pinky > 0.50 or d_thumb_mcp_tip > 0.38)
    extended_states["thumb"] = bool(thumb_extended)

    return FingerStates(
        thumb=extended_states["thumb"],
        index=extended_states["index"],
        middle=extended_states["middle"],
        ring=extended_states["ring"],
        pinky=extended_states["pinky"],
    )


def calculate_pinch_distance(landmarks: Any, finger_tip_idx: int = 8) -> float:
    """
    Calculate normalized Euclidean distance between thumb tip (4) and another fingertip.

    finger_tip_idx:
      - 8 for Index fingertip (Left Pinch / Left Click)
      - 12 for Middle fingertip (Right Pinch / Right Click)
    """
    p = _to_numpy_coords(landmarks)
    palm_scale = _compute_palm_scale(p)
    raw_dist = float(np.linalg.norm(p[4] - p[finger_tip_idx]))
    return raw_dist / palm_scale


# ---------------------------------------------------------------------------
# Gesture Classifier
# ---------------------------------------------------------------------------

class GestureClassifier:
    """
    Stateful gesture classifier with debouncing, confidence scoring,
    and hysteresis.

    Parameters
    ----------
    settings:
        Application ``GestureSettings`` instance.
    """

    def __init__(self, settings: GestureSettings | None = None) -> None:
        self._settings = settings or GestureSettings()

        # State tracking for hysteresis & debouncing
        self._confirmed_gesture: Gesture = Gesture.UNKNOWN
        self._current_confidence: float = 0.0
        self._current_finger_states: FingerStates = FingerStates()
        self._current_handedness: str = "Unknown"
        self._current_pinch_distance: float = 0.0

        # Debounce / candidate confirmation counter
        self._candidate_gesture: Gesture = Gesture.UNKNOWN
        self._candidate_count: int = 0

        # Pinch hysteresis flags
        self._left_pinch_active: bool = False
        self._right_pinch_active: bool = False

        # Tracking loss buffer
        self._loss_counter: int = 0

    @property
    def current_gesture(self) -> Gesture:
        return self._confirmed_gesture

    def reset(self) -> None:
        """Reset internal history and hysteresis states."""
        self._confirmed_gesture = Gesture.UNKNOWN
        self._current_confidence = 0.0
        self._current_finger_states = FingerStates()
        self._current_handedness = "Unknown"
        self._current_pinch_distance = 0.0
        self._candidate_gesture = Gesture.UNKNOWN
        self._candidate_count = 0
        self._left_pinch_active = False
        self._right_pinch_active = False
        self._loss_counter = 0

    def classify(self, hand: Optional[HandLandmarks]) -> GestureResult:
        """
        Classify the current hand gesture with temporal stabilization.

        Parameters
        ----------
        hand:
            ``HandLandmarks`` from tracking pipeline, or ``None`` if no hand.

        Returns
        -------
        GestureResult
        """
        if (
            hand is None
            or not hasattr(hand, "landmarks")
            or hand.landmarks is None
            or len(hand.landmarks) == 0
        ):
            return self._handle_tracking_loss()

        self._loss_counter = 0
        p = _to_numpy_coords(hand.landmarks)
        handedness = hand.handedness or "Right"
        self._current_handedness = handedness

        # 1. Detect finger states
        finger_states = detect_finger_states(p, handedness=handedness)
        self._current_finger_states = finger_states

        # 2. Compute normalized pinch distances
        d_index_pinch = calculate_pinch_distance(p, finger_tip_idx=8)
        d_middle_pinch = calculate_pinch_distance(p, finger_tip_idx=12)

        # 3. Apply hysteresis for pinch states
        th_enter = self._settings.pinch_threshold
        th_release = self._settings.pinch_release_threshold

        # Left pinch hysteresis (Thumb + Index)
        if self._left_pinch_active:
            self._left_pinch_active = (d_index_pinch <= th_release) and (d_index_pinch <= d_middle_pinch)
        else:
            self._left_pinch_active = (d_index_pinch <= th_enter) and (d_index_pinch <= d_middle_pinch)

        # Right pinch hysteresis (Thumb + Middle)
        if self._right_pinch_active:
            self._right_pinch_active = (d_middle_pinch <= th_release) and (d_middle_pinch <= d_index_pinch)
        else:
            self._right_pinch_active = (d_middle_pinch <= th_enter) and (d_middle_pinch <= d_index_pinch)

        # 4. Classify raw gesture by strict priority
        raw_gesture, raw_conf, pinch_dist = self._classify_raw(
            finger_states=finger_states,
            d_index_pinch=d_index_pinch,
            d_middle_pinch=d_middle_pinch,
            th_release=th_release,
        )

        self._current_pinch_distance = pinch_dist

        # 5. Temporal debouncing & confirmation frames
        if raw_gesture == self._candidate_gesture:
            self._candidate_count += 1
        else:
            self._candidate_gesture = raw_gesture
            self._candidate_count = 1

        required_frames = max(1, self._settings.confirmation_frames)
        is_stable = self._candidate_count >= required_frames

        if is_stable:
            self._confirmed_gesture = self._candidate_gesture
            self._current_confidence = raw_conf
        else:
            # During transition, decay current confidence slightly
            self._current_confidence = max(0.2, self._current_confidence * 0.9)

        return GestureResult(
            gesture=self._confirmed_gesture,
            confidence=float(self._current_confidence),
            label=self._get_gesture_label(self._confirmed_gesture),
            finger_states=finger_states,
            handedness=handedness,
            pinch_distance=float(self._current_pinch_distance),
            is_stable=is_stable,
        )

    def _classify_raw(
        self,
        finger_states: FingerStates,
        d_index_pinch: float,
        d_middle_pinch: float,
        th_release: float,
    ) -> tuple[Gesture, float, float]:
        """
        Priority-ordered raw gesture evaluation.

        Returns (Gesture, confidence, active_pinch_distance).
        """
        f_thumb = finger_states.thumb
        f_index = finger_states.index
        f_mid = finger_states.middle
        f_ring = finger_states.ring
        f_pinky = finger_states.pinky
        count = finger_states.count_extended()

        # PRIORITY 1: LEFT PINCH (Thumb + Index touching)
        if self._left_pinch_active:
            # Confidence based on closeness
            conf = float(np.clip(1.0 - (d_index_pinch / th_release) * 0.4, 0.6, 1.0))
            return Gesture.LEFT_PINCH, conf, d_index_pinch

        # PRIORITY 2: RIGHT PINCH (Thumb + Middle touching)
        if self._right_pinch_active:
            conf = float(np.clip(1.0 - (d_middle_pinch / th_release) * 0.4, 0.6, 1.0))
            return Gesture.RIGHT_PINCH, conf, d_middle_pinch

        # PRIORITY 3: OPEN PALM (All 5 fingers extended)
        if count == 5 and f_thumb and f_index and f_mid and f_ring and f_pinky:
            return Gesture.OPEN_PALM, 0.95, 0.0

        # PRIORITY 4: FIST (All fingers folded)
        if count == 0 or (count == 1 and f_thumb and not (f_index or f_mid or f_ring or f_pinky)):
            # Thumb can be loosely resting over fingers in a fist
            return Gesture.FIST, 0.90 if count == 0 else 0.80, 0.0

        # PRIORITY 5: TWO FINGER (Index and Middle extended, Ring and Pinky folded)
        if f_index and f_mid and (not f_ring) and (not f_pinky):
            conf = 0.92 if not f_thumb else 0.85
            return Gesture.TWO_FINGER, conf, 0.0

        # PRIORITY 6: POINTING (Index extended, Middle, Ring, Pinky folded)
        if f_index and (not f_mid) and (not f_ring) and (not f_pinky):
            # Thumb can be folded or resting
            conf = 0.94 if not f_thumb else 0.86
            return Gesture.POINTING, conf, 0.0

        # PRIORITY 7: UNKNOWN
        return Gesture.UNKNOWN, 0.20, 0.0

    def _handle_tracking_loss(self) -> GestureResult:
        """Gracefully handle momentary tracking drops without instant reset."""
        self._loss_counter += 1
        max_tol = self._settings.loss_tolerance_frames

        if self._loss_counter <= max_tol and self._confirmed_gesture != Gesture.UNKNOWN:
            # Hold previous gesture with decayed confidence
            self._current_confidence *= 0.75
            return GestureResult(
                gesture=self._confirmed_gesture,
                confidence=float(self._current_confidence),
                label=f"{self._get_gesture_label(self._confirmed_gesture)} (Hold)",
                finger_states=self._current_finger_states,
                handedness=self._current_handedness,
                pinch_distance=float(self._current_pinch_distance),
                is_stable=False,
            )
        else:
            self._confirmed_gesture = Gesture.UNKNOWN
            self._current_confidence = 0.0
            self._left_pinch_active = False
            self._right_pinch_active = False
            return GestureResult(
                gesture=Gesture.UNKNOWN,
                confidence=0.0,
                label="No Hand",
                finger_states=FingerStates(),
                handedness="None",
                pinch_distance=0.0,
                is_stable=False,
            )

    @staticmethod
    def _get_gesture_label(gesture: Gesture) -> str:
        labels = {
            Gesture.UNKNOWN: "Unknown",
            Gesture.POINTING: "Pointing",
            Gesture.LEFT_PINCH: "Left Pinch",
            Gesture.RIGHT_PINCH: "Right Pinch",
            Gesture.TWO_FINGER: "Two Finger",
            Gesture.OPEN_PALM: "Open Palm",
            Gesture.FIST: "Fist",
        }
        return labels.get(gesture, "Unknown")
