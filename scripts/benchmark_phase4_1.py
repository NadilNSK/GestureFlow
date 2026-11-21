"""
scripts.benchmark_phase4_1
~~~~~~~~~~~~~~~~~~~~~~~~~~
Physical Camera 0 & MediaPipe Performance Benchmark for Phase 4.1.

Measures:
1. Camera 0 DirectShow connection and hardware capture FPS
2. Decoupled pipeline frame processing rate and frame age
3. MediaPipe HandLandmarker (VIDEO mode) inference latency
4. Tracking stability, detection ratio, and jitter metrics

Usage:
    python scripts/benchmark_phase4_1.py [--frames 150] [--device 0]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cv2
import numpy as np

from gestureflow.config.settings import Settings
from gestureflow.camera.capture import CameraCapture, CameraFrame
from gestureflow.tracking.pipeline import TrackingPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("Benchmark")


def run_benchmark(device_index: int = 0, target_frames: int = 150) -> dict:
    print("\n" + "=" * 70)
    print("   GestureFlow Phase 4.1 — Physical Benchmark & Stability Audit")
    print("=" * 70)
    print(f"Target Device: Camera {device_index} (DirectShow)")
    print(f"Sample Count:  {target_frames} frames\n")

    settings = Settings.load(defaults_path=PROJECT_ROOT / "default_settings.json")
    settings.camera.device_index = device_index
    settings.camera.backend = "DSHOW"
    settings.camera.use_mock_camera = False

    # 1. Probe Hardware
    print("[1/4] Probing Camera 0 with cv2.CAP_DSHOW...")
    cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[-] ERROR: Unable to open Camera {device_index} with DirectShow.")
        return {"status": "FAIL", "error": "Camera open failure"}

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"      Camera OK: Resolution {actual_w}x{actual_h}, Driver Reported FPS: {actual_fps}")

    # Warmup read
    for _ in range(5):
        cap.read()
    cap.release()

    # 2. Pipeline Initialization
    print("\n[2/4] Initializing TrackingPipeline (MediaPipe HandLandmarker in VIDEO mode)...")
    pipeline = TrackingPipeline(settings)
    pipeline.open()

    # 3. Capture & Processing Loop
    print(f"\n[3/4] Running live benchmark on Camera {device_index} for {target_frames} frames...")
    print("      (Hold hand in camera view to benchmark tracking stability)")

    cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    frame_times = []
    infer_times = []
    proc_times = []
    frame_ages = []
    hands_detected_count = 0

    t_start = time.perf_counter()
    prev_time = t_start

    try:
        for idx in range(target_frames):
            t_cap_start = time.perf_counter()
            ret, frame = cap.read()
            t_cap_end = time.perf_counter()

            if not ret or frame is None:
                continue

            frame_times.append(t_cap_end - prev_time)
            prev_time = t_cap_end

            cam_frame = CameraFrame(
                image=frame,
                timestamp_ms=int((t_cap_end - t_start) * 1000),
                frame_index=idx,
                fps=30.0,
                capture_perf_time=t_cap_start,
            )

            t_proc_start = time.perf_counter()
            pipeline_frame = pipeline.process(cam_frame)
            t_proc_end = time.perf_counter()

            proc_times.append((t_proc_end - t_proc_start) * 1000.0)
            infer_times.append(pipeline_frame.inference_time_ms)
            frame_ages.append(pipeline_frame.frame_age_ms)

            if pipeline_frame.tracking.has_hands:
                hands_detected_count += 1

            if (idx + 1) % 30 == 0:
                cur_fps = len(frame_times) / sum(frame_times) if frame_times else 0
                avg_infer = np.mean(infer_times[-30:])
                print(f"      Frame {idx + 1:3d}/{target_frames} | Live FPS: {cur_fps:.1f} | Avg Infer: {avg_infer:.1f}ms | State: {pipeline_frame.tracking_state}")

    finally:
        cap.release()
        pipeline.close()

    total_duration = time.perf_counter() - t_start
    avg_fps = len(frame_times) / total_duration if total_duration > 0 else 0
    avg_infer = float(np.mean(infer_times)) if infer_times else 0.0
    p95_infer = float(np.percentile(infer_times, 95)) if infer_times else 0.0
    avg_proc = float(np.mean(proc_times)) if proc_times else 0.0
    avg_age = float(np.mean(frame_ages)) if frame_ages else 0.0
    det_rate = (hands_detected_count / len(frame_times) * 100.0) if frame_times else 0.0

    print("\n" + "=" * 70)
    print("   BENCHMARK RESULTS SUMMARY")
    print("=" * 70)
    print(f"   Total Frames Captured:     {len(frame_times)} / {target_frames}")
    print(f"   Effective Camera FPS:      {avg_fps:.2f} FPS")
    print(f"   MediaPipe Inference (avg): {avg_infer:.2f} ms")
    print(f"   MediaPipe Inference (p95): {p95_infer:.2f} ms")
    print(f"   Full Pipeline Time (avg):  {avg_proc:.2f} ms")
    print(f"   Average Frame Age:         {avg_age:.2f} ms")
    print(f"   Hand Detection Frames:     {hands_detected_count} ({det_rate:.1f}%)")
    print("=" * 70)

    results = {
        "status": "PASS",
        "frames_captured": len(frame_times),
        "effective_fps": avg_fps,
        "avg_inference_ms": avg_infer,
        "p95_inference_ms": p95_infer,
        "avg_pipeline_ms": avg_proc,
        "avg_frame_age_ms": avg_age,
        "detection_rate_pct": det_rate,
    }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GestureFlow Phase 4.1 Benchmark")
    parser.add_argument("--device", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--frames", type=int, default=120, help="Frames to capture (default: 120)")
    args = parser.parse_args()

    results = run_benchmark(device_index=args.device, target_frames=args.frames)
    if results.get("status") == "PASS":
        print("[+] Benchmark completed successfully.")
        sys.exit(0)
    else:
        print(f"[-] Benchmark failed: {results.get('error')}")
        sys.exit(1)
