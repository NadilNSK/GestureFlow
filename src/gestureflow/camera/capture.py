"""
gestureflow.camera.capture
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Thread-safe webcam capture using OpenCV.

The ``CameraCapture`` class runs in a **background thread** and publishes
frames via a callback so the UI thread is never blocked.

Usage
-----
    cap = CameraCapture(settings.camera)
    cap.start(on_frame=my_callback, on_error=my_error_handler)
    ...
    cap.stop()
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from gestureflow.config.settings import CameraSettings

log = logging.getLogger(__name__)


@dataclass
class CameraFrame:
    """A single captured frame with metadata."""

    image: np.ndarray          # BGR, shape (H, W, 3)
    timestamp_ms: int          # Milliseconds since capture start
    frame_index: int           # 0-based frame counter
    fps: float                 # Measured FPS at the moment of capture


@dataclass
class DiscoveredCamera:
    """Metadata for a detected video capture device."""

    index: int
    name: str
    is_available: bool
    backend: str = "DSHOW"


def discover_available_cameras(
    max_indices: int = 5,
    backend: str = "DSHOW",
) -> list[DiscoveredCamera]:
    """
    Scan for connected video capture devices on Windows using cv2.CAP_DSHOW.

    Parameters
    ----------
    max_indices:
        Number of device indices to probe (0 through max_indices - 1).
    backend:
        Backend name ("DSHOW", "MSMF", or "DEFAULT").

    Returns
    -------
    list[DiscoveredCamera]
        List of probed devices with status and descriptions.
    """
    b_upper = (backend or "DSHOW").upper()
    backend_flag = cv2.CAP_DSHOW if b_upper in ("DSHOW", "CAP_DSHOW") else cv2.CAP_ANY

    discovered: list[DiscoveredCamera] = []

    for idx in range(max_indices):
        cap = None
        try:
            cap = cv2.VideoCapture(idx, backend_flag)
            opened = cap.isOpened()
            can_read = False
            if opened:
                ret, _ = cap.read()
                can_read = bool(ret)

            if opened and can_read:
                if idx == 1:
                    label = f"Camera 1 — Primary Webcam ({backend})"
                elif idx == 2:
                    label = f"Camera 2 — Virtual / Secondary ({backend})"
                else:
                    label = f"Camera {idx} — Ready ({backend})"
                discovered.append(
                    DiscoveredCamera(
                        index=idx,
                        name=label,
                        is_available=True,
                        backend=backend,
                    )
                )
            elif opened:
                label = f"Camera {idx} — Open, No Frames ({backend})"
                discovered.append(
                    DiscoveredCamera(
                        index=idx,
                        name=label,
                        is_available=True,
                        backend=backend,
                    )
                )
            else:
                label = f"Camera {idx} (Not Available)"
                discovered.append(
                    DiscoveredCamera(
                        index=idx,
                        name=label,
                        is_available=False,
                        backend=backend,
                    )
                )
        except Exception as exc:
            log.debug("Probing camera %d error: %s", idx, exc)
            discovered.append(
                DiscoveredCamera(
                    index=idx,
                    name=f"Camera {idx} (Error)",
                    is_available=False,
                    backend=backend,
                )
            )
        finally:
            if cap is not None:
                cap.release()

    return discovered


FrameCallback = Callable[[CameraFrame], None]
ErrorCallback = Callable[[Exception], None]


class CameraCapture:
    """
    Non-blocking webcam capture on a daemon thread.

    Parameters
    ----------
    settings:
        ``CameraSettings`` dataclass instance.
    """

    def __init__(self, settings: CameraSettings) -> None:
        self._settings = settings
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._cap: cv2.VideoCapture | None = None
        self._on_frame: FrameCallback | None = None
        self._on_error: ErrorCallback | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(
        self,
        on_frame: FrameCallback,
        on_error: ErrorCallback | None = None,
    ) -> None:
        """
        Open the camera and begin capturing frames in a background thread.

        Parameters
        ----------
        on_frame:
            Called with a ``CameraFrame`` for each captured frame.
            Runs on the capture thread — keep it fast or hand off to a queue.
        on_error:
            Called if an unrecoverable camera error occurs.
        """
        if self._thread and self._thread.is_alive():
            log.warning("CameraCapture.start() called while already running.")
            return

        self._on_frame = on_frame
        self._on_error = on_error
        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._capture_loop,
            name="CameraCapture",
            daemon=True,
        )
        self._thread.start()
        log.info("Camera capture thread started (device=%d).", self._settings.device_index)

    def stop(self, timeout: float = 3.0) -> None:
        """Signal the capture thread to stop and wait for it to exit."""
        log.info("Stopping camera capture…")
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                log.warning("Camera thread did not exit within %.1f s.", timeout)
        self._thread = None
        log.info("Camera capture stopped.")

    @property
    def is_running(self) -> bool:
        """True if the capture thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Internal — capture loop (runs on the daemon thread)
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        """Main loop: open → read frames → close."""
        if self._settings.use_mock_camera:
            self._mock_capture_loop()
            return

        try:
            self._cap = self._open_camera()
        except RuntimeError as exc:
            log.error("Failed to open camera: %s", exc)
            if self._on_error:
                self._on_error(exc)
            return

        frame_index = 0
        consecutive_failures = 0
        t_start = time.monotonic()
        fps_times: list[float] = []

        try:
            while not self._stop_event.is_set():
                ok, bgr = self._cap.read()
                if not ok:
                    consecutive_failures += 1
                    if consecutive_failures >= 30:
                        err = RuntimeError(
                            "Webcam disconnected or stream lost after multiple read failures."
                        )
                        log.error(str(err))
                        if self._on_error:
                            self._on_error(err)
                        break
                    log.warning("Camera read returned False — retrying… (%d/30)", consecutive_failures)
                    time.sleep(0.05)
                    continue

                consecutive_failures = 0

                # Horizontal flip (mirror mode).
                if self._settings.flip_horizontal:
                    bgr = cv2.flip(bgr, 1)

                # FPS calculation (rolling window of 30 frames).
                now = time.monotonic()
                fps_times.append(now)
                if len(fps_times) > 30:
                    fps_times.pop(0)
                fps = (
                    (len(fps_times) - 1) / (fps_times[-1] - fps_times[0])
                    if len(fps_times) > 1
                    else 0.0
                )

                timestamp_ms = int((now - t_start) * 1000)

                frame = CameraFrame(
                    image=bgr,
                    timestamp_ms=timestamp_ms,
                    frame_index=frame_index,
                    fps=fps,
                )
                frame_index += 1

                if self._on_frame:
                    try:
                        self._on_frame(frame)
                    except Exception as cb_exc:
                        log.exception("on_frame callback raised: %s", cb_exc)

        except Exception as exc:
            log.exception("Unexpected error in capture loop: %s", exc)
            if self._on_error:
                self._on_error(exc)
        finally:
            if self._cap:
                self._cap.release()
                self._cap = None
            log.debug("Camera device released.")

    def _mock_capture_loop(self) -> None:
        """Synthetic frame generation loop for headless or testing environments."""
        log.info("Running mock camera capture loop.")
        w, h = self._settings.frame_width, self._settings.frame_height
        target_delay = 1.0 / max(1, self._settings.fps)
        frame_index = 0
        t_start = time.monotonic()
        fps_times: list[float] = []

        try:
            while not self._stop_event.is_set():
                now = time.monotonic()
                fps_times.append(now)
                if len(fps_times) > 30:
                    fps_times.pop(0)
                fps = (
                    (len(fps_times) - 1) / (fps_times[-1] - fps_times[0])
                    if len(fps_times) > 1
                    else float(self._settings.fps)
                )

                # Generate clean synthetic frame
                bgr = np.zeros((h, w, 3), dtype=np.uint8)
                # Subtle dark grid background
                bgr[::40, :, :] = (20, 20, 20)
                bgr[:, ::40, :] = (20, 20, 20)
                cv2.putText(
                    bgr,
                    f"MOCK CAMERA: Frame #{frame_index}",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (100, 200, 100),
                    2,
                )

                timestamp_ms = int((now - t_start) * 1000)
                frame = CameraFrame(
                    image=bgr,
                    timestamp_ms=timestamp_ms,
                    frame_index=frame_index,
                    fps=fps,
                )
                frame_index += 1

                if self._on_frame:
                    self._on_frame(frame)

                time.sleep(target_delay)
        finally:
            log.info("Mock camera capture loop ended.")

    def _open_camera(self) -> cv2.VideoCapture:
        """Open and configure the VideoCapture device using DirectShow or configured backend."""
        idx = self._settings.device_index
        backend_name = (getattr(self._settings, "backend", "DSHOW") or "DSHOW").upper()

        if backend_name in ("DSHOW", "CAP_DSHOW"):
            backend_flag = cv2.CAP_DSHOW
        elif backend_name in ("MSMF", "CAP_MSMF"):
            backend_flag = cv2.CAP_MSMF
        else:
            backend_flag = cv2.CAP_ANY

        log.info("Opening camera device=%d with backend=%s", idx, backend_name)
        cap = cv2.VideoCapture(idx, backend_flag)
        if not cap.isOpened() and backend_flag != cv2.CAP_ANY:
            log.warning(
                "Cannot open camera %d with backend %s; attempting default backend fallback.",
                idx,
                backend_name,
            )
            cap = cv2.VideoCapture(idx)

        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at device index {idx} with backend {backend_name}. "
                "Check that the webcam is connected, not disabled in Windows Settings, "
                "and not in use by another application (e.g. Teams, Zoom, or browser)."
            )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._settings.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._settings.frame_height)
        cap.set(cv2.CAP_PROP_FPS, self._settings.fps)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
        log.info(
            "Camera opened: device=%d  backend=%s  resolution=%dx%d  fps=%d",
            idx,
            backend_name,
            actual_w,
            actual_h,
            actual_fps,
        )
        return cap
