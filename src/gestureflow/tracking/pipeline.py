"""
gestureflow.tracking.pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Combines camera capture, MediaPipe Hand Landmarker, and gesture classification
into a single threaded pipeline that runs off the UI thread.

Emits ``PipelineFrame`` objects via Qt signals so the UI can render live preview
images, hand landmarks, gesture states, and HUD metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from gestureflow.camera.capture import CameraFrame
from gestureflow.config.settings import Settings
from gestureflow.controller.mouse_controller import MouseController, MouseState
from gestureflow.gestures.classifier import (
    FingerStates,
    Gesture,
    GestureClassifier,
    GestureResult,
)
from gestureflow.tracking.hand_landmarker import (
    HandLandmarks,
    HandLandmarkerWrapper,
    TrackingResult,
)

log = logging.getLogger(__name__)

# Standard 21-landmark skeletal connectivity
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # Index
    (0, 9), (9, 10), (10, 11), (11, 12),      # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),    # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),    # Pinky
    (5, 9), (9, 13), (13, 17),                 # Palm knuckles
]


@dataclass
class PipelineFrame:
    """
    Processed frame ready for display and UI dispatch.

    Contains annotated BGR image, tracking metadata, gesture states, and
    virtual mouse coordinates.
    """

    display_image: np.ndarray   # BGR with visual overlays
    tracking: TrackingResult
    fps: float
    frame_index: int
    gesture_result: Optional[GestureResult] = None
    finger_states: Optional[FingerStates] = None
    active_hand: Optional[HandLandmarks] = None
    cursor_coords: Optional[tuple[int, int]] = None
    mouse_state: str = "DISABLED"
    is_precision_mode: bool = False
    is_calibrated: bool = False
    is_dragging: bool = False
    mouse_action: str = "NONE"
    left_click_count: int = 0
    right_click_count: int = 0
    double_click_count: int = 0
    scroll_direction: str = "IDLE"

    # Phase 3 Telemetry & Performance Instrumentation
    inference_time_ms: float = 0.0
    classification_time_ms: float = 0.0
    controller_time_ms: float = 0.0
    pipeline_latency_ms: float = 0.0
    frame_age_ms: float = 0.0
    processing_fps: float = 0.0
    camera_fps: float = 0.0
    tracking_quality: str = "EXCELLENT"
    cursor_profile: str = "Balanced"
    click_strategy: str = "deferred"

    # Phase 4.1 Stability & Diagnostic Telemetry
    tracking_state: str = "HAND LOST"
    detection_ratio: float = 1.0
    mediapipe_no_hands_count: int = 0
    continuity_rejected_count: int = 0
    tracking_lost_count: int = 0
    reacquired_count: int = 0


class TrackingPipeline:
    """
    Stateful tracking & gesture processing pipeline.

    Parameters
    ----------
    settings:
        Application ``Settings`` instance.
    """

    def __init__(self, settings: Settings) -> None:
        import collections

        self._settings = settings
        self._landmarker = HandLandmarkerWrapper(
            settings=settings.tracking,
            model_path=settings.model_path,
        )
        self._classifier = GestureClassifier(settings=settings.gestures)
        self._mouse_controller = MouseController(
            settings=settings.controller,
            mock_mode=settings.camera.use_mock_camera,
        )
        self._prev_active_wrist: Optional[tuple[float, float]] = None
        self._fps_times: list[float] = []

        # Phase 4.1 State Machine & Diagnostic Telemetry
        self._tracking_state: str = "HAND LOST"
        self._consecutive_lost_frames: int = 0
        self._mediapipe_no_hands_count: int = 0
        self._continuity_rejected_count: int = 0
        self._tracking_lost_count: int = 0
        self._reacquired_count: int = 0
        self._detection_history: collections.deque[bool] = collections.deque(maxlen=30)

    @property
    def mouse_controller(self) -> MouseController:
        """Access the pipeline's virtual mouse controller."""
        return self._mouse_controller

    @property
    def tracking_state(self) -> str:
        """Current hand tracking state machine status."""
        return self._tracking_state

    @property
    def mediapipe_no_hands_count(self) -> int:
        return self._mediapipe_no_hands_count

    @property
    def continuity_rejected_count(self) -> int:
        return self._continuity_rejected_count

    @property
    def tracking_lost_count(self) -> int:
        return self._tracking_lost_count

    @property
    def reacquired_count(self) -> int:
        return self._reacquired_count

    @property
    def consecutive_lost_frames(self) -> int:
        return self._consecutive_lost_frames

    @property
    def detection_ratio(self) -> float:
        if not self._detection_history:
            return 0.0
        return sum(self._detection_history) / len(self._detection_history)

    def open(self) -> None:
        """Load the MediaPipe model and reset classifier and mouse state."""
        self._landmarker.open()
        self._classifier.reset()
        self._mouse_controller.reset_smoothing()
        self._prev_active_wrist = None
        self._fps_times.clear()
        self._tracking_state = "HAND LOST"
        self._consecutive_lost_frames = 0
        self._mediapipe_no_hands_count = 0
        self._continuity_rejected_count = 0
        self._tracking_lost_count = 0
        self._reacquired_count = 0
        self._detection_history.clear()

    def close(self) -> None:
        """Release the MediaPipe model and disable mouse control."""
        self._landmarker.close()
        self._classifier.reset()
        self._mouse_controller.disable()
        self._prev_active_wrist = None
        self._fps_times.clear()
        self._tracking_state = "HAND LOST"
        self._consecutive_lost_frames = 0
        self._detection_history.clear()

    def __enter__(self) -> "TrackingPipeline":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def process(self, camera_frame: CameraFrame) -> PipelineFrame:
        """
        Process a single camera frame through tracking, gesture classification,
        and virtual mouse control.

        Parameters
        ----------
        camera_frame:
            Raw frame from ``CameraCapture``.

        Returns
        -------
        PipelineFrame
        """
        import time

        t_process_start = time.perf_counter()
        capture_perf = getattr(camera_frame, "capture_perf_time", 0.0)
        frame_age_ms = (t_process_start - capture_perf) * 1000.0 if capture_perf > 0.0 else 0.0

        bgr = camera_frame.image.copy()

        # 1. MediaPipe Tasks Hand Landmarker detection
        t_infer_start = time.perf_counter()
        tracking = self._landmarker.detect(bgr, camera_frame.timestamp_ms)
        inference_time_ms = (time.perf_counter() - t_infer_start) * 1000.0

        if not tracking.has_hands:
            self._mediapipe_no_hands_count += 1
            self._detection_history.append(False)
        else:
            self._detection_history.append(True)

        # 2. Hand selection (up to 2 hands) with spatial continuity
        t_gest_start = time.perf_counter()
        active_hand = self._select_primary_hand(tracking.hands)

        # 3. Gesture classification & stabilization
        gesture_result = self._classifier.classify(active_hand)
        classification_time_ms = (time.perf_counter() - t_gest_start) * 1000.0
        finger_states = gesture_result.finger_states if active_hand else None

        # 4. Virtual mouse controller dispatch
        t_ctrl_start = time.perf_counter()
        cursor_coords: Optional[tuple[int, int]] = None
        mouse_state = self._mouse_controller.state.value
        if self._mouse_controller.is_enabled and not self._settings.camera.use_mock_camera:
            try:
                ts_sec = float(camera_frame.timestamp_ms) / 1000.0 if camera_frame.timestamp_ms else None
                cursor_coords = self._mouse_controller.process_hand(
                    active_hand, gesture_result, timestamp=ts_sec
                )
                mouse_state = self._mouse_controller.state.value
            except Exception as exc:
                mouse_state = self._mouse_controller.state.value
                log.error("Virtual mouse controller processing error: %s", exc)
                raise
        controller_time_ms = (time.perf_counter() - t_ctrl_start) * 1000.0

        # Total pipeline latency from camera frame capture to completion
        pipeline_latency_ms = (
            (time.perf_counter() - capture_perf) * 1000.0
            if capture_perf > 0.0
            else (time.perf_counter() - t_process_start) * 1000.0
        )

        # Rolling processing FPS
        now_mono = time.monotonic()
        self._fps_times.append(now_mono)
        if len(self._fps_times) > 30:
            self._fps_times.pop(0)
        processing_fps = (
            (len(self._fps_times) - 1) / (self._fps_times[-1] - self._fps_times[0])
            if len(self._fps_times) > 1
            else camera_frame.fps
        )

        # Tracking quality determination
        if not active_hand or not tracking.has_hands:
            if self._tracking_state == "TRACKING UNSTABLE":
                tracking_quality = "UNSTABLE"
            elif self._tracking_state == "REACQUIRING":
                tracking_quality = "REACQUIRING"
            else:
                tracking_quality = "LOST"
        elif active_hand.score >= 0.85:
            tracking_quality = "EXCELLENT"
        elif active_hand.score >= 0.65:
            tracking_quality = "GOOD"
        else:
            tracking_quality = "DEGRADED"

        # 5. Draw active tracking region box
        bgr = self._draw_tracking_region(bgr, self._mouse_controller.is_enabled)

        # 6. Draw visual annotations & HUD overlay
        if self._settings.ui.show_landmarks and tracking.has_hands:
            bgr = self._draw_hands(bgr, tracking.hands, active_hand, gesture_result, cursor_coords)

        if self._settings.ui.show_gesture_label:
            bgr = self._draw_hud(bgr, gesture_result, active_hand, len(tracking.hands), mouse_state, cursor_coords, tracking_quality)

        if self._settings.ui.show_fps:
            self._draw_fps(bgr, camera_frame.fps)

        return PipelineFrame(
            display_image=bgr,
            tracking=tracking,
            fps=camera_frame.fps,
            frame_index=camera_frame.frame_index,
            gesture_result=gesture_result,
            finger_states=finger_states,
            active_hand=active_hand,
            cursor_coords=cursor_coords,
            mouse_state=mouse_state,
            is_precision_mode=self._mouse_controller.is_precision_mode,
            is_calibrated=self._mouse_controller.calibrator.is_calibrated,
            is_dragging=self._mouse_controller.is_dragging,
            mouse_action=self._mouse_controller.last_action,
            left_click_count=self._mouse_controller.left_click_count,
            right_click_count=self._mouse_controller.right_click_count,
            double_click_count=self._mouse_controller.double_click_count,
            scroll_direction=self._mouse_controller.scroll_direction,
            inference_time_ms=inference_time_ms,
            classification_time_ms=classification_time_ms,
            controller_time_ms=controller_time_ms,
            pipeline_latency_ms=pipeline_latency_ms,
            frame_age_ms=frame_age_ms,
            processing_fps=processing_fps,
            camera_fps=camera_frame.fps,
            tracking_quality=tracking_quality,
            cursor_profile=self._mouse_controller.cursor_profile,
            click_strategy=self._mouse_controller.click_strategy,
            tracking_state=self._tracking_state,
            detection_ratio=self.detection_ratio,
            mediapipe_no_hands_count=self._mediapipe_no_hands_count,
            continuity_rejected_count=self._continuity_rejected_count,
            tracking_lost_count=self._tracking_lost_count,
            reacquired_count=self._reacquired_count,
        )

    # ------------------------------------------------------------------
    # Hand Selection Logic
    # ------------------------------------------------------------------

    def _select_primary_hand(self, hands: list[HandLandmarks]) -> Optional[HandLandmarks]:
        """
        Consistently select the primary control hand when 1 or 2 hands are visible.

        Prioritises spatial continuity with the previously active hand, then
        preferred handedness ("Right" by default), and detection score.
        """
        if not hands:
            self._consecutive_lost_frames += 1
            if self._consecutive_lost_frames == 1:
                self._tracking_state = "TRACKING UNSTABLE"
            elif self._consecutive_lost_frames == 2:
                self._tracking_state = "REACQUIRING"
            else:
                if self._tracking_state != "HAND LOST":
                    self._tracking_lost_count += 1
                self._tracking_state = "HAND LOST"
                self._prev_active_wrist = None
                if self._mouse_controller.is_dragging:
                    self._mouse_controller.release_held_buttons()
            return None

        # When hands are detected:
        if self._consecutive_lost_frames > 0:
            self._reacquired_count += 1
            self._consecutive_lost_frames = 0
        self._tracking_state = "HAND DETECTED"

        def _get_wrist_xy(hand: HandLandmarks) -> Optional[tuple[float, float]]:
            if hand.landmarks is None or len(hand.landmarks) == 0:
                return None
            lm0 = hand.landmarks[0]
            if hasattr(lm0, "x") and hasattr(lm0, "y"):
                return float(lm0.x), float(lm0.y)
            return float(lm0[0]), float(lm0[1])

        if len(hands) == 1:
            chosen = hands[0]
            self._prev_active_wrist = _get_wrist_xy(chosen)
            return chosen

        # When 2 hands are detected:
        pref = self._settings.gestures.preferred_handedness.lower()

        # If we were already tracking a hand, prioritize spatial continuity
        if self._prev_active_wrist is not None:
            px, py = self._prev_active_wrist

            def _dist_sq(h: HandLandmarks) -> float:
                w = _get_wrist_xy(h)
                if w is None:
                    return 999.0
                return (w[0] - px) ** 2 + (w[1] - py) ** 2

            closest_hand = min(hands, key=_dist_sq)
            if _dist_sq(closest_hand) < 0.125:  # within ~35% normalized distance
                self._prev_active_wrist = _get_wrist_xy(closest_hand)
                return closest_hand
            else:
                self._continuity_rejected_count += 1

        # Otherwise check if one matches user preference
        pref_matches = [h for h in hands if h.handedness.lower() == pref]
        if len(pref_matches) == 1:
            chosen = pref_matches[0]
        else:
            # Fallback to higher detection confidence
            chosen = max(hands, key=lambda h: h.score)

        self._prev_active_wrist = _get_wrist_xy(chosen)
        return chosen

    # ------------------------------------------------------------------
    # Visual Drawing & HUD
    # ------------------------------------------------------------------

    def _draw_tracking_region(self, bgr: np.ndarray, is_mouse_enabled: bool) -> np.ndarray:
        """Render active tracking region rectangle or calibrated polygon on camera preview frame."""
        h, w = bgr.shape[:2]
        calib = self._mouse_controller.calibrator
        is_prec = self._mouse_controller.is_precision_mode

        if calib.is_calibrated and calib.data.camera_points:
            pts = []
            for corner in ["TL", "TR", "BR", "BL"]:
                if corner in calib.data.camera_points:
                    cx, cy = calib.data.camera_points[corner]
                    pts.append((int(cx * w), int(cy * h)))
            if len(pts) == 4:
                poly = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                poly_color = (0, 255, 255) if is_prec else ((0, 220, 255) if is_mouse_enabled else (80, 80, 80))
                cv2.polylines(bgr, [poly], isClosed=True, color=poly_color, thickness=2, lineType=cv2.LINE_AA)
                cv2.putText(
                    bgr,
                    "CALIBRATED ZONE" + (" [PRECISION]" if is_prec else ""),
                    (pts[0][0] + 5, max(20, pts[0][1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    poly_color,
                    1,
                    cv2.LINE_AA,
                )
                return bgr

        m = self._settings.controller.tracking_region_margin
        rx1 = int(w * m)
        ry1 = int(h * m)
        rx2 = int(w * (1.0 - m))
        ry2 = int(h * (1.0 - m))

        if is_prec:
            box_color = (0, 255, 255)  # Yellow in precision mode
        elif is_mouse_enabled:
            box_color = (255, 220, 0)  # Cyan
        else:
            box_color = (70, 70, 70)   # Muted gray

        cv2.rectangle(bgr, (rx1, ry1), (rx2, ry2), box_color, 1, cv2.LINE_AA)

        # Draw corner accents
        clen = 16
        cv2.line(bgr, (rx1, ry1), (rx1 + clen, ry1), box_color, 2, cv2.LINE_AA)
        cv2.line(bgr, (rx1, ry1), (rx1, ry1 + clen), box_color, 2, cv2.LINE_AA)
        cv2.line(bgr, (rx2, ry1), (rx2 - clen, ry1), box_color, 2, cv2.LINE_AA)
        cv2.line(bgr, (rx2, ry1), (rx2, ry1 + clen), box_color, 2, cv2.LINE_AA)
        cv2.line(bgr, (rx1, ry2), (rx1 + clen, ry2), box_color, 2, cv2.LINE_AA)
        cv2.line(bgr, (rx1, ry2), (rx1, ry2 - clen), box_color, 2, cv2.LINE_AA)
        cv2.line(bgr, (rx2, ry2), (rx2 - clen, ry2), box_color, 2, cv2.LINE_AA)
        cv2.line(bgr, (rx2, ry2), (rx2, ry2 - clen), box_color, 2, cv2.LINE_AA)

        lbl = "TRACKING REGION [MOUSE ACTIVE]" if is_mouse_enabled else "TRACKING REGION [MOUSE OFF]"
        ty = ry1 - 8 if ry1 > 20 else ry1 + 18
        cv2.putText(bgr, lbl, (rx1 + 6, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.40, box_color, 1, cv2.LINE_AA)
        return bgr

    def _draw_hands(
        self,
        bgr: np.ndarray,
        hands: list[HandLandmarks],
        active_hand: Optional[HandLandmarks],
        gesture_result: GestureResult,
        cursor_coords: Optional[tuple[int, int]] = None,
    ) -> np.ndarray:
        """Draw landmarks, connections, wrist tags, pinch lines, and cursor reticle."""
        h, w = bgr.shape[:2]

        for hand in hands:
            is_active = (hand is active_hand)
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand.landmarks]

            # Primary vs secondary color schemes
            if is_active:
                conn_color = (0, 225, 120)    # Vibrant green
                dot_color = (255, 255, 255)    # White
                ring_color = (0, 180, 70)      # Dark green
            else:
                conn_color = (180, 140, 80)    # Subdued slate / orange
                dot_color = (200, 200, 200)
                ring_color = (120, 100, 50)

            # Draw skeletal lines
            for a, b in HAND_CONNECTIONS:
                if 0 <= a < len(pts) and 0 <= b < len(pts):
                    cv2.line(bgr, pts[a], pts[b], conn_color, 2, cv2.LINE_AA)

            # Draw landmark dots
            for pt in pts:
                cv2.circle(bgr, pt, 4, dot_color, -1, cv2.LINE_AA)
                cv2.circle(bgr, pt, 4, ring_color, 1, cv2.LINE_AA)

            # Draw wrist label with handedness and confidence
            if pts:
                wrist_pt = pts[0]
                tag = f"{hand.handedness} ({hand.score * 100:.0f}%)"
                if is_active:
                    tag += " [ACTIVE]"
                cv2.putText(
                    bgr,
                    tag,
                    (max(10, wrist_pt[0] - 40), min(h - 10, wrist_pt[1] + 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    conn_color,
                    1,
                    cv2.LINE_AA,
                )

            # Highlight active pinch interactions on the primary hand
            if is_active and gesture_result.gesture in (Gesture.LEFT_PINCH, Gesture.RIGHT_PINCH):
                thumb_tip = pts[4]
                pinch_tip = pts[8] if gesture_result.gesture == Gesture.LEFT_PINCH else pts[12]
                pinch_color = (255, 220, 0) if gesture_result.gesture == Gesture.LEFT_PINCH else (0, 180, 255)

                # Draw connecting line and halos
                cv2.line(bgr, thumb_tip, pinch_tip, pinch_color, 3, cv2.LINE_AA)
                cv2.circle(bgr, thumb_tip, 8, pinch_color, 2, cv2.LINE_AA)
                cv2.circle(bgr, pinch_tip, 8, pinch_color, 2, cv2.LINE_AA)

            # Highlight index fingertip target reticle when virtual mouse pointing is active
            if is_active and gesture_result.gesture == Gesture.POINTING and len(pts) > 8:
                tip = pts[8]
                reticle_color = (0, 255, 255)  # Bright cyan
                cv2.circle(bgr, tip, 14, reticle_color, 2, cv2.LINE_AA)
                cv2.circle(bgr, tip, 3, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.line(bgr, (tip[0] - 18, tip[1]), (tip[0] + 18, tip[1]), reticle_color, 1, cv2.LINE_AA)
                cv2.line(bgr, (tip[0], tip[1] - 18), (tip[0], tip[1] + 18), reticle_color, 1, cv2.LINE_AA)

        return bgr

    def _draw_hud(
        self,
        bgr: np.ndarray,
        gesture: GestureResult,
        active_hand: Optional[HandLandmarks],
        num_hands: int,
        mouse_state: str = "DISABLED",
        cursor_coords: Optional[tuple[int, int]] = None,
        tracking_quality: str = "EXCELLENT",
    ) -> np.ndarray:
        """Render upper-left HUD card with gesture, finger states, hand status, and mouse telemetry."""
        h, w = bgr.shape[:2]

        # HUD semi-transparent background box
        hud_w, hud_h = 320, 126
        overlay = bgr.copy()
        cv2.rectangle(overlay, (10, 10), (10 + hud_w, 10 + hud_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.75, bgr, 0.25, 0, bgr)
        cv2.rectangle(bgr, (10, 10), (10 + hud_w, 10 + hud_h), (50, 50, 50), 1)

        # 1. Gesture Name Badge
        g_name = gesture.label.upper()
        conf_pct = int(gesture.confidence * 100)
        g_text = f"GESTURE: {g_name} ({conf_pct}%)"

        if gesture.gesture == Gesture.POINTING:
            g_col = (0, 255, 150)    # Green
        elif gesture.gesture in (Gesture.LEFT_PINCH, Gesture.RIGHT_PINCH):
            g_col = (255, 220, 0)    # Cyan / Yellow
        elif gesture.gesture == Gesture.OPEN_PALM:
            g_col = (255, 120, 220)  # Magenta
        elif gesture.gesture == Gesture.TWO_FINGER:
            g_col = (100, 200, 255)  # Sky blue
        elif gesture.gesture == Gesture.FIST:
            g_col = (0, 160, 255)    # Orange
        else:
            g_col = (160, 160, 160)  # Gray

        cv2.putText(bgr, g_text, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.58, g_col, 2, cv2.LINE_AA)

        # 2. Hand tracking status
        if active_hand:
            hand_str = f"Hand: {active_hand.handedness} ({num_hands} in view) • {tracking_quality}"
            h_col = (0, 255, 180) if tracking_quality in ("EXCELLENT", "GOOD") else (0, 220, 255)
        elif self._tracking_state == "TRACKING UNSTABLE":
            hand_str = "Hand: TRACKING UNSTABLE (1/2)"
            h_col = (0, 200, 255)
        elif self._tracking_state == "REACQUIRING":
            hand_str = "Hand: REACQUIRING (2/2)"
            h_col = (0, 165, 255)
        else:
            hand_str = "Hand: Searching..."
            h_col = (130, 130, 130)

        cv2.putText(bgr, hand_str, (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.48, h_col, 1, cv2.LINE_AA)

        # 3. Finger States indicator string
        f_str = gesture.finger_states.summary_str()
        cv2.putText(bgr, f"Fingers: {f_str}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 180), 1, cv2.LINE_AA)

        # 4. Virtual Mouse Control Telemetry
        if mouse_state == "MOVING":
            m_col = (0, 255, 120)
            coord_str = f"({cursor_coords[0]}, {cursor_coords[1]})" if cursor_coords else "—"
            m_text = f"Mouse: MOVING {coord_str}"
        elif mouse_state == "EMERGENCY_STOPPED":
            m_col = (50, 50, 255)
            m_text = "Mouse: EMERGENCY STOPPED"
        elif mouse_state == "IDLE":
            m_col = (255, 220, 0)
            m_text = "Mouse: READY (Idle)"
        else:
            m_col = (140, 140, 140)
            m_text = "Mouse: DISABLED"

        cv2.putText(bgr, m_text, (20, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.48, m_col, 1, cv2.LINE_AA)

        return bgr

    @staticmethod
    def _draw_fps(bgr: np.ndarray, fps: float) -> None:
        """Render FPS meter in top-right corner."""
        w = bgr.shape[1]
        text = f"FPS: {fps:.1f}"
        cv2.putText(
            bgr,
            text,
            (w - 120, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 230, 100),
            2,
            cv2.LINE_AA,
        )
