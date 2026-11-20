"""
Phase 2A Test Suite — Finger-state detection, gesture classification,
stability, hysteresis, tracking loss, and hand selection.
"""

from __future__ import annotations

import numpy as np
import pytest

from gestureflow.config.settings import GestureSettings, Settings
from gestureflow.gestures.classifier import (
    FingerStates,
    Gesture,
    GestureClassifier,
    GestureResult,
    calculate_pinch_distance,
    detect_finger_states,
    joint_angle,
)
from gestureflow.tracking.hand_landmarker import HandLandmarks, TrackingResult
from gestureflow.tracking.pipeline import TrackingPipeline, PipelineFrame
from gestureflow.camera.capture import CameraFrame


# ---------------------------------------------------------------------------
# Synthetic Hand Fixtures & Generators
# ---------------------------------------------------------------------------

def make_synthetic_hand(
    thumb: bool = True,
    index: bool = True,
    middle: bool = True,
    ring: bool = True,
    pinky: bool = True,
    pinch: str | None = None,
    angle_deg: float = 0.0,
    handedness: str = "Right",
) -> np.ndarray:
    """
    Generate synthetic 21 hand landmarks with precise geometry.

    Can configure extension state of each finger, pinch interaction,
    arbitrary 2D/3D rotation around wrist, and handedness mirroring.
    """
    p = np.zeros((21, 3), dtype=np.float32)
    p[0] = [0.50, 0.80, 0.0]  # Wrist

    # Base MCPs (knuckles across palm)
    p[5] = [0.43, 0.60, 0.0]   # Index MCP
    p[9] = [0.50, 0.58, 0.0]   # Middle MCP
    p[13] = [0.57, 0.60, 0.0]  # Ring MCP
    p[17] = [0.63, 0.64, 0.0]  # Pinky MCP

    # Thumb: 1 (CMC), 2 (MCP), 3 (IP), 4 (TIP)
    p[1] = [0.44, 0.74, 0.0]
    p[2] = [0.38, 0.70, 0.0]
    if thumb:
        p[3] = [0.32, 0.65, 0.0]
        p[4] = [0.26, 0.60, 0.0]  # Extended outward
    else:
        p[3] = [0.42, 0.66, 0.0]
        p[4] = [0.42, 0.72, 0.0]  # Curled inward over palm

    # 4 Fingers: Index, Middle, Ring, Pinky
    mcp_indices = [5, 9, 13, 17]
    ext_flags = [index, middle, ring, pinky]
    for i, ext in enumerate(ext_flags):
        mcp = mcp_indices[i]
        pip = mcp + 1
        dip = mcp + 2
        tip = mcp + 3
        bx = p[mcp][0]

        if ext:
            p[pip] = [bx, 0.50, 0.0]
            p[dip] = [bx, 0.42, 0.0]
            p[tip] = [bx, 0.34, 0.0]  # Straight up, away from wrist
        else:
            p[pip] = [bx, 0.54, 0.0]
            p[dip] = [bx, 0.59, 0.0]
            p[tip] = [bx, 0.62, 0.0]  # Curled down into palm

    # Handle pinch positioning
    if pinch == "index":
        p[4] = p[8] + [0.005, 0.005, 0.0]
    elif pinch == "middle":
        p[4] = p[12] + [0.005, 0.005, 0.0]

    # Mirror for Left hand if requested
    if handedness.lower() == "left":
        p[:, 0] = 1.0 - p[:, 0]

    # Apply 2D rotation around wrist (0)
    if angle_deg != 0.0:
        rad = np.radians(angle_deg)
        c, s = np.cos(rad), np.sin(rad)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        wrist = p[0].copy()
        p = (p - wrist) @ R.T + wrist

    return p


# ---------------------------------------------------------------------------
# Task 2: Finger-State Detection Tests
# ---------------------------------------------------------------------------

class TestFingerStateDetection:
    def test_all_fingers_extended_open_palm(self):
        landmarks = make_synthetic_hand(True, True, True, True, True)
        states = detect_finger_states(landmarks)
        assert states.thumb is True
        assert states.index is True
        assert states.middle is True
        assert states.ring is True
        assert states.pinky is True
        assert states.count_extended() == 5

    def test_all_fingers_folded_fist(self):
        landmarks = make_synthetic_hand(False, False, False, False, False)
        states = detect_finger_states(landmarks)
        assert states.thumb is False
        assert states.index is False
        assert states.middle is False
        assert states.ring is False
        assert states.pinky is False
        assert states.count_extended() == 0

    def test_pointing_index_only(self):
        landmarks = make_synthetic_hand(False, True, False, False, False)
        states = detect_finger_states(landmarks)
        assert states.index is True
        assert states.middle is False
        assert states.ring is False
        assert states.pinky is False

    def test_two_finger_extended(self):
        landmarks = make_synthetic_hand(False, True, True, False, False)
        states = detect_finger_states(landmarks)
        assert states.index is True
        assert states.middle is True
        assert states.ring is False
        assert states.pinky is False

    @pytest.mark.parametrize("angle", [0, 45, 90, 135, 180, 225, 270])
    def test_rotation_invariance_across_360_degrees(self, angle):
        """Finger extension must remain invariant to hand roll/rotation."""
        landmarks = make_synthetic_hand(True, True, True, True, True, angle_deg=angle)
        states = detect_finger_states(landmarks)
        assert states.as_tuple() == (True, True, True, True, True), (
            f"Failed rotation invariance at {angle} degrees"
        )

    def test_left_hand_mirroring(self):
        landmarks = make_synthetic_hand(True, True, True, True, True, handedness="Left")
        states = detect_finger_states(landmarks, handedness="Left")
        assert states.as_tuple() == (True, True, True, True, True)


# ---------------------------------------------------------------------------
# Task 3: Gesture Classification & Priority Tests
# ---------------------------------------------------------------------------

class TestGestureClassification:
    def test_open_palm_classification(self):
        classifier = GestureClassifier()
        hand = HandLandmarks(
            landmarks=make_synthetic_hand(True, True, True, True, True),
            world_landmarks=[],
            handedness="Right",
            score=0.95,
        )
        for _ in range(3):
            res = classifier.classify(hand)
        assert res.gesture == Gesture.OPEN_PALM
        assert res.label == "Open Palm"
        assert res.confidence >= 0.85
        assert res.is_stable is True

    def test_pointing_classification(self):
        classifier = GestureClassifier()
        hand = HandLandmarks(
            landmarks=make_synthetic_hand(False, True, False, False, False),
            world_landmarks=[],
            handedness="Right",
            score=0.92,
        )
        for _ in range(3):
            res = classifier.classify(hand)
        assert res.gesture == Gesture.POINTING
        assert res.label == "Pointing"
        assert res.confidence >= 0.80

    def test_two_finger_classification(self):
        classifier = GestureClassifier()
        hand = HandLandmarks(
            landmarks=make_synthetic_hand(False, True, True, False, False),
            world_landmarks=[],
            handedness="Right",
            score=0.90,
        )
        for _ in range(3):
            res = classifier.classify(hand)
        assert res.gesture == Gesture.TWO_FINGER
        assert res.label == "Two Finger"

    def test_fist_classification(self):
        classifier = GestureClassifier()
        hand = HandLandmarks(
            landmarks=make_synthetic_hand(False, False, False, False, False),
            world_landmarks=[],
            handedness="Right",
            score=0.91,
        )
        for _ in range(3):
            res = classifier.classify(hand)
        assert res.gesture == Gesture.FIST
        assert res.label == "Fist"

    def test_left_pinch_classification(self):
        classifier = GestureClassifier()
        hand = HandLandmarks(
            landmarks=make_synthetic_hand(True, True, False, False, False, pinch="index"),
            world_landmarks=[],
            handedness="Right",
            score=0.94,
        )
        for _ in range(3):
            res = classifier.classify(hand)
        assert res.gesture == Gesture.LEFT_PINCH
        assert res.label == "Left Pinch"

    def test_right_pinch_classification(self):
        classifier = GestureClassifier()
        hand = HandLandmarks(
            landmarks=make_synthetic_hand(True, False, True, False, False, pinch="middle"),
            world_landmarks=[],
            handedness="Right",
            score=0.93,
        )
        for _ in range(3):
            res = classifier.classify(hand)
        assert res.gesture == Gesture.RIGHT_PINCH
        assert res.label == "Right Pinch"

    def test_gesture_priority_pinch_over_pointing(self):
        """When index is extended but thumb is pinching index, pinch MUST take priority."""
        classifier = GestureClassifier()
        hand = HandLandmarks(
            landmarks=make_synthetic_hand(True, True, False, False, False, pinch="index"),
            world_landmarks=[],
            handedness="Right",
            score=0.95,
        )
        for _ in range(3):
            res = classifier.classify(hand)
        assert res.gesture == Gesture.LEFT_PINCH
        assert res.gesture != Gesture.POINTING


# ---------------------------------------------------------------------------
# Task 4: Stability, Hysteresis, Debouncing & Tracking Loss Tests
# ---------------------------------------------------------------------------

class TestGestureStability:
    def test_confirmation_frames_debouncing(self):
        """Candidate gesture must be confirmed for N frames before state switch."""
        settings = GestureSettings(confirmation_frames=3)
        classifier = GestureClassifier(settings=settings)

        hand_palm = HandLandmarks(
            landmarks=make_synthetic_hand(True, True, True, True, True),
            world_landmarks=[],
            handedness="Right",
            score=0.9,
        )
        # Settle on OPEN_PALM
        for _ in range(3):
            classifier.classify(hand_palm)
        assert classifier.current_gesture == Gesture.OPEN_PALM

        hand_fist = HandLandmarks(
            landmarks=make_synthetic_hand(False, False, False, False, False),
            world_landmarks=[],
            handedness="Right",
            score=0.9,
        )
        # Frame 1: candidate is FIST, but confirmed remains OPEN_PALM
        r1 = classifier.classify(hand_fist)
        assert r1.gesture == Gesture.OPEN_PALM
        assert r1.is_stable is False

        # Frame 2: still confirming
        r2 = classifier.classify(hand_fist)
        assert r2.gesture == Gesture.OPEN_PALM
        assert r2.is_stable is False

        # Frame 3: confirmed FIST!
        r3 = classifier.classify(hand_fist)
        assert r3.gesture == Gesture.FIST
        assert r3.is_stable is True

    def test_temporary_tracking_loss_tolerance(self):
        """Loss of tracking for <= 3 frames retains state; >3 frames drops to UNKNOWN."""
        classifier = GestureClassifier(settings=GestureSettings(loss_tolerance_frames=3))

        hand = HandLandmarks(
            landmarks=make_synthetic_hand(False, True, False, False, False),
            world_landmarks=[],
            handedness="Right",
            score=0.9,
        )
        for _ in range(3):
            classifier.classify(hand)
        assert classifier.current_gesture == Gesture.POINTING

        # Drops 1, 2, 3: Hold state
        for _ in range(3):
            res = classifier.classify(None)
            assert res.gesture == Gesture.POINTING
            assert "Hold" in res.label

        # Drop 4: Exceeds tolerance -> resets to UNKNOWN
        res_dropped = classifier.classify(None)
        assert res_dropped.gesture == Gesture.UNKNOWN
        assert res_dropped.label == "No Hand"

    def test_pinch_distance_calculation(self):
        hand = make_synthetic_hand(True, True, False, False, False, pinch="index")
        d = calculate_pinch_distance(hand, finger_tip_idx=8)
        assert d < 0.065

    def test_invalid_landmarks_handled_gracefully(self):
        classifier = GestureClassifier()
        # Empty landmarks
        empty_hand = HandLandmarks(landmarks=[], world_landmarks=[], handedness="Right", score=0.0)
        res = classifier.classify(empty_hand)
        assert res.gesture == Gesture.UNKNOWN


# ---------------------------------------------------------------------------
# Hand Selection (Two Hands) Tests
# ---------------------------------------------------------------------------

class TestTwoHandTracking:
    def test_pipeline_selects_preferred_hand(self, default_settings):
        default_settings.gestures.preferred_handedness = "Right"
        pipeline = TrackingPipeline(default_settings)

        right_hand = HandLandmarks(
            landmarks=make_synthetic_hand(handedness="Right"),
            world_landmarks=[],
            handedness="Right",
            score=0.85,
        )
        left_hand = HandLandmarks(
            landmarks=make_synthetic_hand(handedness="Left"),
            world_landmarks=[],
            handedness="Left",
            score=0.92,
        )

        # Right hand should be selected based on preference even if Left has slightly higher score
        selected = pipeline._select_primary_hand([left_hand, right_hand])
        assert selected is right_hand

    def test_pipeline_fallback_to_higher_confidence(self, default_settings):
        default_settings.gestures.preferred_handedness = "Right"
        pipeline = TrackingPipeline(default_settings)

        left1 = HandLandmarks(
            landmarks=make_synthetic_hand(handedness="Left"),
            world_landmarks=[],
            handedness="Left",
            score=0.70,
        )
        left2 = HandLandmarks(
            landmarks=make_synthetic_hand(handedness="Left"),
            world_landmarks=[],
            handedness="Left",
            score=0.95,
        )

        # Neither matches Right; select higher confidence Left hand
        selected = pipeline._select_primary_hand([left1, left2])
        assert selected is left2
