"""
scripts.benchmark_phase4_1
~~~~~~~~~~~~~~~~~~~~~~~~~~
Comprehensive Evidence-Based Performance Benchmark for Phase 4.1 & Phase 4.2.

Provides 3 distinct, independent benchmark modes:
1. Camera-Only Hardware Benchmark:
   Measures pure hardware frame delivery with NO MediaPipe, NO PySide6, NO controller.
   Records: Delivered FPS, cap.read() avg & p95 duration, frame interval & jitter,
   requested vs actual resolution, and requested vs actual FOURCC.

2. Sequential Capture + Pipeline Benchmark:
   Measures blocking single-thread loop (cap.read() followed immediately by pipeline.process()).
   Identifies throughput degradation caused by single-thread lockstep execution.

3. Actual Threaded Application Benchmark:
   Measures the real application decoupled architecture:
   CameraCapture (thread 1) -> 1-slot buffer -> PipelineProcessingThread (thread 2)
   Records: Captured FPS, Processed FPS, Dropped Frames count, MediaPipe Inference latency,
   true post-capture Frame Age, and end-to-end latency estimate.

Usage:
    python scripts/benchmark_phase4_1.py [--mode {all,camera,sequential,threaded}]
                                         [--device DEVICE]
                                         [--backend {DSHOW,MSMF}]
                                         [--frames FRAMES]
"""

from __future__ import annotations

import argparse
import collections
import logging
import sys
import threading
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cv2
import numpy as np

from gestureflow.config.settings import Settings
from gestureflow.camera.capture import CameraCapture, CameraFrame
from gestureflow.tracking.pipeline import TrackingPipeline, PipelineFrame

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("Benchmark")


def fourcc_to_str(fourcc_val: float) -> str:
    val = int(fourcc_val)
    if val <= 0:
        return "None"
    chars = [chr((val >> (8 * i)) & 0xFF) for i in range(4)]
    s = "".join(chars).strip()
    return s if s and all(c.isprintable() for c in s) else f"0x{val:08X}"


# ===========================================================================
# 1. CAMERA-ONLY HARDWARE BENCHMARK
# ===========================================================================

def benchmark_camera_only(
    device_index: int = 0,
    backend_str: str = "DSHOW",
    width: int = 640,
    height: int = 480,
    target_fps: int = 30,
    fourcc_str: str = "MJPG",
    target_frames: int = 60,
) -> dict:
    backend_flag = cv2.CAP_DSHOW if backend_str.upper() == "DSHOW" else cv2.CAP_MSMF
    print(f"\n[1/3] Camera-Only Hardware Benchmark: {backend_str} (Device {device_index})")
    print(f"      Target: {width}x{height} @ {target_fps}fps, FOURCC={fourcc_str}")

    cap = cv2.VideoCapture(device_index, backend_flag)
    if not cap.isOpened():
        print(f"[-] ERROR: Failed to open device {device_index} with backend {backend_str}")
        return {"status": "FAIL", "error": f"Open failure on {backend_str}"}

    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, target_fps)
        if backend_str.upper() == "DSHOW":
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc_str))
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        reported_fps = cap.get(cv2.CAP_PROP_FPS)
        actual_fourcc = fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC))

        # Warmup
        for _ in range(8):
            cap.read()

        read_durations = []
        frame_timestamps = []

        t_start = time.perf_counter()
        for _ in range(target_frames):
            t0 = time.perf_counter()
            ret, frame = cap.read()
            t1 = time.perf_counter()
            if not ret or frame is None:
                continue
            read_durations.append((t1 - t0) * 1000.0)
            frame_timestamps.append(t1)
        t_end = time.perf_counter()

        total_time = t_end - t_start
        delivered_fps = len(frame_timestamps) / total_time if total_time > 0 else 0.0
        avg_read_ms = float(np.mean(read_durations)) if read_durations else 0.0
        p95_read_ms = float(np.percentile(read_durations, 95)) if read_durations else 0.0

        intervals = np.diff(frame_timestamps) * 1000.0 if len(frame_timestamps) > 1 else [0.0]
        avg_interval = float(np.mean(intervals))
        std_interval = float(np.std(intervals))

        print(f"      Delivered FPS:     {delivered_fps:.2f} FPS ({len(frame_timestamps)} frames)")
        print(f"      Driver Reported:   {actual_w}x{actual_h} @ {reported_fps:.1f} FPS, FOURCC={actual_fourcc}")
        print(f"      cap.read() time:   avg={avg_read_ms:.2f}ms, p95={p95_read_ms:.2f}ms")
        print(f"      Frame Interval:    avg={avg_interval:.2f}ms, jitter std={std_interval:.2f}ms")

        return {
            "status": "PASS",
            "backend": backend_str,
            "delivered_fps": delivered_fps,
            "reported_fps": reported_fps,
            "actual_res": f"{actual_w}x{actual_h}",
            "actual_fourcc": actual_fourcc,
            "avg_read_ms": avg_read_ms,
            "p95_read_ms": p95_read_ms,
            "avg_interval_ms": avg_interval,
            "std_interval_ms": std_interval,
        }
    finally:
        cap.release()
        time.sleep(0.4)


# ===========================================================================
# 2. SEQUENTIAL CAPTURE + PIPELINE BENCHMARK (Single Thread Loop)
# ===========================================================================

def benchmark_sequential_loop(
    device_index: int = 0,
    backend_str: str = "DSHOW",
    target_frames: int = 60,
) -> dict:
    print(f"\n[2/3] Sequential Loop Benchmark (Capture + Inference in single thread): {backend_str}")
    settings = Settings.load(defaults_path=PROJECT_ROOT / "default_settings.json")
    settings.camera.device_index = device_index
    settings.camera.backend = backend_str
    settings.camera.use_mock_camera = False

    pipeline = TrackingPipeline(settings)
    pipeline.open()

    backend_flag = cv2.CAP_DSHOW if backend_str.upper() == "DSHOW" else cv2.CAP_MSMF
    cap = cv2.VideoCapture(device_index, backend_flag)
    if not cap.isOpened():
        pipeline.close()
        return {"status": "FAIL", "error": f"Camera open failed on {backend_str}"}

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if backend_str.upper() == "DSHOW":
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    # Warmup
    for _ in range(5):
        cap.read()

    frame_times = []
    infer_times = []
    proc_times = []
    frame_ages = []
    hands_detected = 0

    t_start = time.perf_counter()
    prev_time = t_start

    try:
        for idx in range(target_frames):
            ret, frame = cap.read()
            t_cap_end = time.perf_counter()
            if not ret or frame is None:
                continue

            frame_times.append(t_cap_end - prev_time)
            prev_time = t_cap_end

            # Valid capture_perf_time is recorded POST-read
            cam_frame = CameraFrame(
                image=frame,
                timestamp_ms=int((t_cap_end - t_start) * 1000),
                frame_index=idx,
                fps=30.0,
                capture_perf_time=t_cap_end,
            )

            t0 = time.perf_counter()
            p_frame = pipeline.process(cam_frame)
            t1 = time.perf_counter()

            proc_times.append((t1 - t0) * 1000.0)
            infer_times.append(p_frame.inference_time_ms)
            frame_ages.append(p_frame.frame_age_ms)
            if p_frame.tracking.has_hands:
                hands_detected += 1

    finally:
        cap.release()
        pipeline.close()
        time.sleep(0.4)

    total_time = time.perf_counter() - t_start
    seq_fps = len(frame_times) / total_time if total_time > 0 else 0.0
    avg_infer = float(np.mean(infer_times)) if infer_times else 0.0
    avg_proc = float(np.mean(proc_times)) if proc_times else 0.0
    avg_age = float(np.mean(frame_ages)) if frame_ages else 0.0

    print(f"      Sequential FPS:    {seq_fps:.2f} FPS ({len(frame_times)} frames in {total_time:.2f}s)")
    print(f"      Avg Inference:     {avg_infer:.2f} ms")
    print(f"      Full Process Time: {avg_proc:.2f} ms")
    print(f"      Post-Capture Age:  {avg_age:.2f} ms")
    print(f"      Hands Detected:    {hands_detected}/{len(frame_times)}")

    return {
        "status": "PASS",
        "backend": backend_str,
        "sequential_fps": seq_fps,
        "avg_infer_ms": avg_infer,
        "avg_proc_ms": avg_proc,
        "avg_age_ms": avg_age,
        "hands_detected": hands_detected,
    }


# ===========================================================================
# 3. THREADED APPLICATION ARCHITECTURE BENCHMARK
# ===========================================================================

def benchmark_threaded_pipeline(
    device_index: int = 0,
    backend_str: str = "DSHOW",
    duration_sec: float = 5.0,
) -> dict:
    print(f"\n[3/3] Threaded Application Architecture Benchmark: {backend_str}")
    print(f"      CameraCapture thread -> 1-slot buffer -> PipelineWorker thread")

    settings = Settings.load(defaults_path=PROJECT_ROOT / "default_settings.json")
    settings.camera.device_index = device_index
    settings.camera.backend = backend_str
    settings.camera.use_mock_camera = False

    pipeline = TrackingPipeline(settings)
    pipeline.open()

    frame_lock = threading.Lock()
    new_frame_event = threading.Event()
    latest_frame = [None]
    running = [True]

    captured = [0]
    processed = [0]
    dropped = [0]
    infer_times = []
    frame_ages = []
    hands_detected = [0]

    def on_frame(cam_frame: CameraFrame) -> None:
        captured[0] += 1
        with frame_lock:
            if latest_frame[0] is not None:
                dropped[0] += 1
            latest_frame[0] = cam_frame
        new_frame_event.set()

    def proc_loop() -> None:
        while running[0]:
            if not new_frame_event.wait(timeout=0.05):
                continue
            new_frame_event.clear()
            with frame_lock:
                f = latest_frame[0]
                latest_frame[0] = None

            if f is not None and running[0]:
                p = pipeline.process(f)
                processed[0] += 1
                infer_times.append(p.inference_time_ms)
                frame_ages.append(p.frame_age_ms)
                if p.tracking.has_hands:
                    hands_detected[0] += 1

    t_worker = threading.Thread(target=proc_loop, daemon=True)
    t_worker.start()

    camera = CameraCapture(settings.camera)
    camera.start(on_frame=on_frame)

    # Warmup until streaming starts
    for _ in range(60):
        if captured[0] >= 5:
            break
        time.sleep(0.1)

    # Reset counters for clean timing window
    captured[0] = 0
    processed[0] = 0
    dropped[0] = 0
    infer_times.clear()
    frame_ages.clear()
    hands_detected[0] = 0

    t_start = time.perf_counter()
    time.sleep(duration_sec)
    t_end = time.perf_counter()

    running[0] = False
    new_frame_event.set()
    camera.stop()
    t_worker.join(timeout=2.0)
    pipeline.close()
    time.sleep(0.4)

    elapsed = t_end - t_start
    c_fps = captured[0] / elapsed if elapsed > 0 else 0.0
    p_fps = processed[0] / elapsed if elapsed > 0 else 0.0
    avg_inf = float(np.mean(infer_times)) if infer_times else 0.0
    avg_age = float(np.mean(frame_ages)) if frame_ages else 0.0

    print(f"      Captured FPS:      {c_fps:.2f} FPS ({captured[0]} frames in {elapsed:.2f}s)")
    print(f"      Processed FPS:     {p_fps:.2f} FPS ({processed[0]} frames)")
    print(f"      Dropped in Buffer: {dropped[0]} frames ({(dropped[0] / max(1, captured[0]) * 100):.1f}%)")
    print(f"      Avg Inference:     {avg_inf:.2f} ms")
    print(f"      Avg Frame Age:     {avg_age:.2f} ms")
    print(f"      Hands Detected:    {hands_detected[0]}/{processed[0]}")

    return {
        "status": "PASS",
        "backend": backend_str,
        "captured_fps": c_fps,
        "processed_fps": p_fps,
        "dropped_frames": dropped[0],
        "avg_infer_ms": avg_inf,
        "avg_age_ms": avg_age,
        "hands_detected": hands_detected[0],
    }


def main():
    parser = argparse.ArgumentParser(description="GestureFlow Phase 4.2 Diagnostic Benchmark")
    parser.add_argument("--mode", choices=["all", "camera", "sequential", "threaded"], default="all")
    parser.add_argument("--device", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--backend", choices=["DSHOW", "MSMF"], default="DSHOW", help="Camera backend")
    parser.add_argument("--frames", type=int, default=60, help="Frames to capture (default: 60)")
    args = parser.parse_args()

    print("=" * 75)
    print("  GESTUREFLOW PHASE 4.2: EVIDENCE-BASED FPS BOTTLENECK DIAGNOSIS")
    print("=" * 75)

    res_cam = None
    res_seq = None
    res_thr = None

    if args.mode in ("all", "camera"):
        res_cam = benchmark_camera_only(device_index=args.device, backend_str=args.backend, target_frames=args.frames)

    if args.mode in ("all", "sequential"):
        res_seq = benchmark_sequential_loop(device_index=args.device, backend_str=args.backend, target_frames=args.frames)

    if args.mode in ("all", "threaded"):
        res_thr = benchmark_threaded_pipeline(device_index=args.device, backend_str=args.backend, duration_sec=5.0)

    print("\n" + "=" * 75)
    print("  DIAGNOSTIC BENCHMARK SUMMARY")
    print("=" * 75)
    if res_cam:
        print(f"  [Camera-Only Hardware]  Delivered: {res_cam['delivered_fps']:.2f} FPS | Read Avg: {res_cam['avg_read_ms']:.2f} ms | FOURCC: {res_cam['actual_fourcc']}")
    if res_seq:
        print(f"  [Sequential Loop]       Throughput: {res_seq['sequential_fps']:.2f} FPS | Infer: {res_seq['avg_infer_ms']:.2f} ms | Frame Age: {res_seq['avg_age_ms']:.2f} ms")
    if res_thr:
        print(f"  [Threaded App Pipeline] Captured: {res_thr['captured_fps']:.2f} FPS | Processed: {res_thr['processed_fps']:.2f} FPS | Buffer Dropped: {res_thr['dropped_frames']}")
    print("=" * 75)


if __name__ == "__main__":
    main()
