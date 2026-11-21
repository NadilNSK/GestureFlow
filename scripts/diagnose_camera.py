"""
Diagnostic script for Task 2: Independent Camera Hardware Benchmarks.
Tests camera-only capture with no MediaPipe, PySide6, or mouse controller.
"""

from __future__ import annotations

import time
import cv2
import numpy as np


def fourcc_to_str(fourcc_val: float) -> str:
    val = int(fourcc_val)
    chars = [chr((val >> (8 * i)) & 0xFF) for i in range(4)]
    return "".join(chars)


def benchmark_config(
    name: str,
    device_index: int,
    backend_flag: int,
    fourcc_str: str | None,
    width: int,
    height: int,
    fps: int,
    num_frames: int = 60,
) -> dict:
    print(f"\n--- Testing: {name} ---")
    cap = cv2.VideoCapture(device_index, backend_flag)
    if not cap.isOpened():
        print(f"[-] Failed to open camera with backend {backend_flag}")
        return {"name": name, "supported": False, "error": "Failed to open"}

    try:
        if fourcc_str:
            f_code = cv2.VideoWriter_fourcc(*fourcc_str)
            cap.set(cv2.CAP_PROP_FOURCC, f_code)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        reported_fps = cap.get(cv2.CAP_PROP_FPS)
        actual_fourcc_val = cap.get(cv2.CAP_PROP_FOURCC)
        actual_fourcc = fourcc_to_str(actual_fourcc_val)
        auto_exp = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        exp = cap.get(cv2.CAP_PROP_EXPOSURE)

        print(f"    Requested: {width}x{height} @ {fps}fps, FOURCC={fourcc_str}")
        print(f"    Driver Reported: {actual_w}x{actual_h} @ {reported_fps:.1f}fps, Actual FOURCC={actual_fourcc!r} (val={int(actual_fourcc_val)})")
        print(f"    Auto Exposure: {auto_exp}, Exposure: {exp}")

        # Warmup (10 frames)
        for _ in range(10):
            ret, _ = cap.read()
            if not ret:
                break

        # Benchmarking loop
        read_durations = []
        frame_timestamps = []

        t_bench_start = time.perf_counter()
        for i in range(num_frames):
            t0 = time.perf_counter()
            ret, frame = cap.read()
            t1 = time.perf_counter()
            if not ret or frame is None:
                continue
            read_durations.append((t1 - t0) * 1000.0)
            frame_timestamps.append(t1)

        t_bench_end = time.perf_counter()
        total_time = t_bench_end - t_bench_start

        if len(frame_timestamps) < 2:
            return {"name": name, "supported": False, "error": "Insufficient frames read"}

        delivered_fps = len(frame_timestamps) / total_time
        avg_read_ms = float(np.mean(read_durations))
        p95_read_ms = float(np.percentile(read_durations, 95))
        intervals = np.diff(frame_timestamps) * 1000.0
        avg_interval_ms = float(np.mean(intervals))
        std_interval_ms = float(np.std(intervals))

        print(f"    Delivered FPS:     {delivered_fps:.2f} FPS")
        print(f"    cap.read() avg:    {avg_read_ms:.2f} ms")
        print(f"    cap.read() p95:    {p95_read_ms:.2f} ms")
        print(f"    Frame Interval:    {avg_interval_ms:.2f} ms (jitter std: {std_interval_ms:.2f} ms)")

        return {
            "name": name,
            "supported": True,
            "requested_fourcc": fourcc_str,
            "actual_fourcc": actual_fourcc,
            "requested_res": f"{width}x{height}",
            "actual_res": f"{actual_w}x{actual_h}",
            "requested_fps": fps,
            "reported_fps": reported_fps,
            "delivered_fps": delivered_fps,
            "avg_read_ms": avg_read_ms,
            "p95_read_ms": p95_read_ms,
            "avg_interval_ms": avg_interval_ms,
            "std_interval_ms": std_interval_ms,
            "auto_exposure": auto_exp,
            "exposure": exp,
        }
    finally:
        cap.release()
        time.sleep(0.5)  # Allow hardware driver to release


def main():
    print("=" * 70)
    print("  TASK 2: INDEPENDENT CAMERA-ONLY HARDWARE BENCHMARKS (DEVICE 0)")
    print("=" * 70)

    configs = [
        ("DSHOW + MJPG + 640x480 + 30 FPS", 0, cv2.CAP_DSHOW, "MJPG", 640, 480, 30),
        ("DSHOW + YUY2 + 640x480 + 30 FPS", 0, cv2.CAP_DSHOW, "YUY2", 640, 480, 30),
        ("DSHOW + MJPG + 320x240 + 30 FPS", 0, cv2.CAP_DSHOW, "MJPG", 320, 240, 30),
        ("MSMF  + MJPG + 640x480 + 30 FPS", 0, cv2.CAP_MSMF,  "MJPG", 640, 480, 30),
        ("MSMF  + YUY2 + 640x480 + 30 FPS", 0, cv2.CAP_MSMF,  "YUY2", 640, 480, 30),
    ]

    results = []
    for cfg in configs:
        res = benchmark_config(*cfg)
        results.append(res)

    print("\n" + "=" * 70)
    print("  SUMMARY TABLE OF INDEPENDENT CAMERA MEASUREMENTS")
    print("=" * 70)
    print(f"{'Configuration':<34} | {'Delivered':<9} | {'Read Avg':<8} | {'Read p95':<8} | {'Interval':<8} | {'Actual FOURCC':<10}")
    print("-" * 90)
    for r in results:
        if r.get("supported"):
            print(f"{r['name']:<34} | {r['delivered_fps']:>6.2f}fps | {r['avg_read_ms']:>6.2f}ms | {r['p95_read_ms']:>6.2f}ms | {r['avg_interval_ms']:>6.2f}ms | {r['actual_fourcc']:<10}")
        else:
            print(f"{r['name']:<34} | {'FAILED':<9} | {'-':<8} | {'-':<8} | {'-':<8} | {r.get('error', 'Error'):<10}")
    print("=" * 70)


if __name__ == "__main__":
    main()
