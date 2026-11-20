"""
gestureflow.ui.workers
~~~~~~~~~~~~~~~~~~~~~~~
QThread-based worker that runs the camera + tracking pipeline and emits
processed frames as Qt signals so the UI thread can update widgets safely.
"""

from __future__ import annotations

import logging
import traceback

import numpy as np
from PySide6.QtCore import QThread, Signal, QObject

from gestureflow.camera.capture import CameraCapture, CameraFrame
from gestureflow.tracking.pipeline import TrackingPipeline, PipelineFrame
from gestureflow.config.settings import Settings

log = logging.getLogger(__name__)


class PipelineWorker(QObject):
    """
    Worker object moved to a ``QThread`` that owns the camera + pipeline.

    Signals
    -------
    frame_ready(PipelineFrame):
        Emitted for every processed frame.  Connect to UI slots.
    error_occurred(str):
        Emitted when an unrecoverable error happens.
    status_changed(str):
        Emitted with human-readable status messages.
    """

    frame_ready: Signal = Signal(object)        # PipelineFrame
    error_occurred: Signal = Signal(str)        # error message
    status_changed: Signal = Signal(str)        # status text

    def __init__(self, settings: Settings, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._camera = CameraCapture(settings.camera)
        self._pipeline: TrackingPipeline | None = None
        self._running = False

    # ------------------------------------------------------------------
    # Slots (called from the owning QThread)
    # ------------------------------------------------------------------

    def start_pipeline(self) -> None:
        """Open the pipeline and start the camera. Called in the worker thread."""
        log.info("PipelineWorker: starting pipeline…")
        self.status_changed.emit("Loading hand landmark model…")
        try:
            self._pipeline = TrackingPipeline(self._settings)
            self._pipeline.open()
        except FileNotFoundError as exc:
            msg = str(exc)
            log.error(msg)
            self.error_occurred.emit(msg)
            return
        except Exception as exc:
            msg = f"Failed to load MediaPipe model: {exc}"
            log.exception(msg)
            self.error_occurred.emit(msg)
            return

        self._running = True
        self.status_changed.emit("Camera starting…")
        self._camera.start(on_frame=self._on_frame, on_error=self._on_camera_error)
        self.status_changed.emit("Running")

    @property
    def mouse_controller(self):
        """Access the underlying MouseController if pipeline is open."""
        return self._pipeline.mouse_controller if self._pipeline else None

    def enable_mouse(self, enabled: bool) -> bool:
        """Enable or disable gesture mouse control."""
        if self._pipeline:
            if enabled:
                return self._pipeline.mouse_controller.enable()
            else:
                self._pipeline.mouse_controller.disable()
                return True
        return False

    def emergency_stop(self) -> None:
        """Trigger emergency stop on virtual mouse controller."""
        if self._pipeline:
            self._pipeline.mouse_controller.emergency_stop()

    def reset_emergency_stop(self) -> None:
        """Reset emergency stop state."""
        if self._pipeline:
            self._pipeline.mouse_controller.reset_emergency_stop()

    def set_sensitivity(self, val: float) -> None:
        """Set mouse sensitivity."""
        if self._pipeline:
            self._pipeline.mouse_controller.set_sensitivity(val)

    def set_smoothing(self, val: float) -> None:
        """Set mouse smoothing factor."""
        if self._pipeline:
            self._pipeline.mouse_controller.set_smoothing(val)

    def set_precision_mode(self, enabled: bool) -> None:
        """Enable or disable precision aiming mode."""
        if self._pipeline:
            self._pipeline.mouse_controller.set_precision_mode(enabled)

    def set_filter_type(self, filter_type: str) -> None:
        """Set smoothing filter algorithm ('one_euro' or 'ema')."""
        if self._pipeline:
            self._pipeline.mouse_controller.set_filter_type(filter_type)

    def set_one_euro_params(
        self,
        min_cutoff: float | None = None,
        beta: float | None = None,
        d_cutoff: float | None = None,
    ) -> None:
        """Update One Euro filter parameters."""
        if self._pipeline:
            self._pipeline.mouse_controller.set_one_euro_params(min_cutoff, beta, d_cutoff)

    def set_calibrator(self, calibrator) -> None:
        """Assign calibrated calibrator to mouse controller."""
        if self._pipeline:
            self._pipeline.mouse_controller.set_calibrator(calibrator)

    def set_action_enabled(self, action_name: str, enabled: bool) -> None:
        """Enable or disable an individual mouse gesture action."""
        if not self._pipeline:
            return
        c_settings = self._pipeline.mouse_controller.settings
        act = action_name.lower().strip()
        if act == "left_click":
            c_settings.left_click_enabled = bool(enabled)
        elif act == "right_click":
            c_settings.right_click_enabled = bool(enabled)
        elif act == "double_click":
            c_settings.double_click_enabled = bool(enabled)
        elif act == "drag":
            c_settings.drag_enabled = bool(enabled)
        elif act == "scroll":
            c_settings.scroll_enabled = bool(enabled)

    def set_drag_hold_duration(self, val: float) -> None:
        """Update required hold duration before drag activates."""
        if self._pipeline:
            self._pipeline.mouse_controller.settings.drag_hold_duration = max(0.15, min(2.0, float(val)))

    def set_double_click_interval(self, val: float) -> None:
        """Update maximum interval for double-click pinches."""
        if self._pipeline:
            self._pipeline.mouse_controller.settings.double_click_interval = max(0.10, min(1.5, float(val)))

    def set_scroll_sensitivity(self, val: float) -> None:
        """Update vertical scroll sensitivity."""
        if self._pipeline:
            self._pipeline.mouse_controller.settings.scroll_sensitivity = max(0.1, min(10.0, float(val)))

    def stop_pipeline(self) -> None:
        """Stop the camera and release model resources. Called in the worker thread."""
        log.info("PipelineWorker: stopping pipeline…")
        self._running = False
        if self._pipeline:
            self._pipeline.mouse_controller.disable()
        self._camera.stop()
        if self._pipeline:
            self._pipeline.close()
            self._pipeline = None
        self.status_changed.emit("Stopped")
        log.info("PipelineWorker: pipeline stopped.")

    # ------------------------------------------------------------------
    # Internal callbacks
    # ------------------------------------------------------------------

    def _on_frame(self, camera_frame: CameraFrame) -> None:
        """Called by CameraCapture on its capture thread."""
        if not self._running or self._pipeline is None:
            return
        try:
            pipeline_frame = self._pipeline.process(camera_frame)
            self.frame_ready.emit(pipeline_frame)
        except Exception as exc:
            # Check for PyAutoGUI FailSafeException
            is_failsafe = False
            try:
                import pyautogui
                failsafe_cls = getattr(pyautogui, "FailSafeException", None)
                if failsafe_cls is not None and isinstance(exc, failsafe_cls):
                    is_failsafe = True
            except ImportError:
                pass

            if is_failsafe:
                msg = "FAILSAFE TRIGGERED: Mouse moved to screen corner! Emergency stop activated."
                log.critical(msg)
                self.error_occurred.emit(msg)
                self.status_changed.emit("EMERGENCY STOP")
            elif "cannot schedule new futures after shutdown" in str(exc) or not self._running:
                log.debug("Frame processing ignored during shutdown.")
            else:
                log.error("Pipeline processing error: %s\n%s", exc, traceback.format_exc())

    def _on_camera_error(self, exc: Exception) -> None:
        """Called by CameraCapture when an unrecoverable error occurs."""
        self._running = False
        msg = f"Camera error: {exc}"
        log.error(msg)
        self.error_occurred.emit(msg)
