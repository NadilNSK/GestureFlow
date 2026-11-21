import collections
import ctypes
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gestureflow.config.settings import Settings
from gestureflow.camera.capture import CameraCapture
from gestureflow.tracking.pipeline import TrackingPipeline


def run_q_test(max_q: int):
    settings = Settings.load()
    settings.camera.backend = "MSMF"
    pipeline = TrackingPipeline(settings)
    pipeline.open()

    frame_lock = threading.Lock()
    new_frame_event = threading.Event()
    q = collections.deque(maxlen=max_q)
    running = [True]

    captured = [0]
    processed = [0]
    dropped = [0]
    infer_times = []
    frame_ages = []

    def on_frame(cam_frame):
        captured[0] += 1
        with frame_lock:
            if len(q) == max_q:
                dropped[0] += 1
            q.append(cam_frame)
        new_frame_event.set()

    def proc_loop():
        while running[0]:
            if not new_frame_event.wait(timeout=0.02):
                continue
            with frame_lock:
                if not q:
                    new_frame_event.clear()
                    continue
                f = q.pop()  # Latest frame
                q.clear()
                new_frame_event.clear()

            if f is not None and running[0]:
                p = pipeline.process(f)
                processed[0] += 1
                infer_times.append(p.inference_time_ms)
                frame_ages.append(p.frame_age_ms)

    t_worker = threading.Thread(target=proc_loop, daemon=True)
    t_worker.start()

    camera = CameraCapture(settings.camera)
    camera.start(on_frame=on_frame)

    # Warmup
    for _ in range(50):
        if captured[0] >= 5:
            break
        time.sleep(0.1)

    captured[0] = 0
    processed[0] = 0
    dropped[0] = 0
    infer_times.clear()
    frame_ages.clear()

    t_start = time.perf_counter()
    time.sleep(5.0)
    t_end = time.perf_counter()

    running[0] = False
    new_frame_event.set()
    camera.stop()
    t_worker.join(timeout=2.0)
    pipeline.close()

    elapsed = t_end - t_start
    print(f"=== max_q={max_q} ===")
    print(f"  Captured FPS:  {captured[0] / elapsed:.2f} FPS ({captured[0]} frames)")
    print(f"  Processed FPS: {processed[0] / elapsed:.2f} FPS ({processed[0]} frames)")
    print(f"  Dropped:       {dropped[0]} frames")
    print(f"  Avg Inference: {sum(infer_times)/len(infer_times):.2f} ms")
    print(f"  Avg Frame Age: {sum(frame_ages)/len(frame_ages):.2f} ms")


if __name__ == "__main__":
    ctypes.windll.winmm.timeBeginPeriod(1)
    try:
        run_q_test(1)
        time.sleep(1.0)
        run_q_test(2)
    finally:
        ctypes.windll.winmm.timeEndPeriod(1)
