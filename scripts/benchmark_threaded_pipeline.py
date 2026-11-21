"""
Benchmark script for Task 3: Threaded Pipeline Architecture Benchmark.
Measures:
CameraCapture (thread 1) -> Latest-Frame Buffer -> PipelineProcessingThread (thread 2)
Records:
- Captured FPS
- Processed FPS
- Dropped frames count
- MediaPipe inference duration
- Frame age (latency from capture completion to processing start)
- Capture-to-controller latency estimate
"""

from __future__ import annotations

import collections
import logging
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
from gestureflow.config.settings import Settings
from gestureflow.camera.capture import CameraCapture, CameraFrame
from gestureflow.tracking.pipeline import TrackingPipeline, PipelineFrame

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ThreadedBenchmark")


def benchmark_threaded_pipeline(
    backend: str = "DSHOW",
    device_index: int = 0,
    duration_sec: float = 5.0,
) -> dict:
    print(f"\n=======================================================")
    print(f"  THREADED PIPELINE BENCHMARK: Backend={backend}, Device={device_index}")
    print(f"  Duration: {duration_sec} seconds")
    print(f"=======================================================")

    settings = Settings.load(defaults_path=PROJECT_ROOT / "default_settings.json")
    settings.camera.device_index = device_index
    settings.camera.backend = backend
    settings.camera.use_mock_camera = False

    # Pipeline
    pipeline = TrackingPipeline(settings)
    pipeline.open()

    # Threading synchronizers
    frame_lock = threading.Lock()
    new_frame_event = threading.Event()
    latest_frame: CameraFrame | None = None
    is_running = True

    # Telemetry
    captured_frames: list[CameraFrame] = []
    processed_frames: list[PipelineFrame] = []
    dropped_frames_count = 0
    capture_intervals = []
    t_prev_capture = None

    # 1. Camera Frame Callback (Runs on CameraCapture thread)
    def on_camera_frame(cam_frame: CameraFrame) -> None:
        nonlocal latest_frame, dropped_frames_count, t_prev_capture
        now = time.perf_counter()
        if t_prev_capture is not None:
            capture_intervals.append(now - t_prev_capture)
        t_prev_capture = now

        captured_frames.append(cam_frame)
        with frame_lock:
            if latest_frame is not None:
                dropped_frames_count += 1
            latest_frame = cam_frame
        new_frame_event.set()

    # 2. Pipeline Processing Loop (Runs on dedicated worker thread)
    process_thread_times = []
    inference_durations = []
    frame_ages = []

    def processing_loop() -> None:
        nonlocal latest_frame
        while is_running:
            if not new_frame_event.wait(timeout=0.05):
                continue
            new_frame_event.clear()
            with frame_lock:
                frame_to_proc = latest_frame
                latest_frame = None

            if frame_to_proc is None or not is_running:
                continue

            t0 = time.perf_counter()
            p_frame = pipeline.process(frame_to_proc)
            t1 = time.perf_counter()

            process_thread_times.append((t1 - t0) * 1000.0)
            inference_durations.append(p_frame.inference_time_ms)
            frame_ages.append(p_frame.frame_age_ms)
            processed_frames.append(p_frame)

    proc_thread = threading.Thread(target=processing_loop, name="ProcWorker", daemon=True)
    proc_thread.start()

    camera = CameraCapture(settings.camera)
    camera.start(on_frame=on_camera_frame)

    t_start = time.perf_counter()
    time.sleep(duration_sec)
    t_end = time.perf_counter()

    is_running = False
    new_frame_event.set()
    camera.stop()
    proc_thread.join(timeout=2.0)
    pipeline.close()

    total_time = t_end - t_start
    n_captured = len(captured_frames)
    n_processed = len(processed_frames)
    captured_fps = n_captured / total_time if total_time > 0 else 0.0
    processed_fps = n_processed / total_time if total_time > 0 else 0.0

    avg_infer = float(np.mean(inference_durations)) if inference_durations else 0.0
    p95_infer = float(np.percentile(inference_durations, 95)) if inference_durations else 0.0
    avg_proc = float(np.mean(process_thread_times)) if process_thread_times else 0.0
    avg_age = float(np.mean(frame_ages)) if frame_ages else 0.0
    avg_interval_ms = float(np.mean(capture_intervals) * 1000.0) if capture_intervals else 0.0

    print(f"\n  [RESULTS: {backend}]")
    print(f"  Duration:                  {total_time:.2f} s")
    print(f"  Camera Captured Frames:    {n_captured} ({captured_fps:.2f} FPS)")
    print(f"  Pipeline Processed Frames: {n_processed} ({processed_fps:.2f} FPS)")
    print(f"  Dropped Frames (Buffer):   {dropped_frames_count} ({(dropped_frames_count / max(1, n_captured) * 100):.1f}%)")
    print(f"  Average Capture Interval:  {avg_interval_ms:.2f} ms")
    print(f"  MediaPipe Inference (avg): {avg_infer:.2f} ms (p95: {p95_infer:.2f} ms)")
    print(f"  Total Process Time (avg):  {avg_proc:.2f} ms")
    print(f"  Average Frame Age:         {avg_age:.2f} ms")
    print(f"  Est. Capture-to-Ctrl Lat:  {avg_age + avg_proc:.2f} ms")

    return {
        "backend": backend,
        "captured_fps": captured_fps,
        "processed_fps": processed_fps,
        "n_captured": n_captured,
        "n_processed": n_processed,
        "dropped_frames": dropped_frames_count,
        "avg_infer_ms": avg_infer,
        "avg_proc_ms": avg_proc,
        "avg_age_ms": avg_age,
        "latency_ms": avg_age + avg_proc,
    }


def main():
    res_dshow = benchmark_threaded_pipeline(backend="DSHOW", device_index=0, duration_sec=5.0)
    time.sleep(1.0)
    res_msmf = benchmark_threaded_pipeline(backend="MSMF", device_index=0, duration_sec=5.0)

    print("\n" + "=" * 70)
    print("  THREADED PIPELINE COMPARISON: DSHOW vs MSMF")
    print("=" * 70)
    print(f"{'Metric':<30} | {'DSHOW':<18} | {'MSMF':<18}")
    print("-" * 70)
    print(f"{'Captured FPS':<30} | {res_dshow['captured_fps']:>14.2f} FPS | {res_msmf['captured_fps']:>14.2f} FPS")
    print(f"{'Processed FPS':<30} | {res_dshow['processed_fps']:>14.2f} FPS | {res_msmf['processed_fps']:>14.2f} FPS")
    print(f"{'Dropped Frames':<30} | {res_dshow['dropped_frames']:>14d}     | {res_msmf['dropped_frames']:>14d}    ")
    print(f"{'MediaPipe Inference':<30} | {res_dshow['avg_infer_ms']:>14.2f} ms  | {res_msmf['avg_infer_ms']:>14.2f} ms ")
    print(f"{'Pipeline Process Time':<30} | {res_dshow['avg_proc_ms']:>14.2f} ms  | {res_msmf['avg_proc_ms']:>14.2f} ms ")
    print(f"{'Frame Age at Processing':<30} | {res_dshow['avg_age_ms']:>14.2f} ms  | {res_msmf['avg_age_ms']:>14.2f} ms ")
    print(f"{'End-to-End Latency':<30} | {res_dshow['latency_ms']:>14.2f} ms  | {res_msmf['latency_ms']:>14.2f} ms ")
    print("=" * 70)


if __name__ == "__main__":
    main()
