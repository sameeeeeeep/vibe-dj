"""Crowd-energy sensing from a webcam.

Energy is derived from inter-frame motion (how much the crowd is moving) and
self-calibrates to the room via an adaptive scale, so the output is a stable
0..1 "vibe" signal. If OpenCV or a camera is unavailable it falls back to a
simulated signal so the rest of the system still runs.
"""

from __future__ import annotations

import math
import threading
import time


class CrowdSensor:
    def __init__(self, simulate: bool = False, camera: int = 0, smoothing: float = 0.1):
        self.simulate = simulate
        self.camera = camera
        self.smoothing = smoothing      # EMA factor for the published energy
        self._energy = 0.5
        self._scale = 1e-6              # adaptive normaliser for raw motion
        self._jpeg: bytes | None = None  # latest cam frame, JPEG-encoded (camera mode)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.mode = "simulated" if simulate else "camera"

    @property
    def energy(self) -> float:
        with self._lock:
            return self._energy

    @property
    def last_jpeg(self) -> bytes | None:
        """Most recent webcam frame as JPEG bytes, or None (simulated/no cam)."""
        with self._lock:
            return self._jpeg

    def _publish(self, raw_motion: float) -> None:
        # Adaptive scale: chase peaks quickly, decay slowly to recalibrate.
        self._scale = max(raw_motion, self._scale * 0.999)
        norm = min(1.0, raw_motion / self._scale) if self._scale > 0 else 0.0
        with self._lock:
            self._energy += self.smoothing * (norm - self._energy)

    def start(self) -> "CrowdSensor":
        target = self._sim_loop if self.simulate else self._camera_loop
        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()
        return self

    def _camera_loop(self) -> None:
        try:
            import cv2
            import numpy as np
        except Exception:
            self.simulate, self.mode = True, "simulated (opencv missing)"
            return self._sim_loop()

        cap = cv2.VideoCapture(self.camera)
        if not cap.isOpened():
            self.simulate, self.mode = True, "simulated (no camera)"
            return self._sim_loop()

        prev = None
        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            small = cv2.resize(frame, (160, 120))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype("float32")
            if prev is not None:
                motion = float(np.abs(gray - prev).mean())
                self._publish(motion)
            prev = gray
            ok_enc, buf = cv2.imencode(".jpg", small)
            if ok_enc:
                jpeg = buf.tobytes()
                with self._lock:
                    self._jpeg = jpeg
            time.sleep(0.03)
        cap.release()

    def _sim_loop(self) -> None:
        import random
        t = 0.0
        target = 0.5
        while not self._stop.is_set():
            if random.random() < 0.02:
                target = random.uniform(0.2, 0.95)
            base = 0.5 + 0.35 * math.sin(t / 12.0)
            val = 0.85 * base + 0.15 * target + random.uniform(-0.05, 0.05)
            with self._lock:
                self._energy = min(1.0, max(0.0, val))
            t += 0.25
            time.sleep(0.25)

    def stop(self) -> None:
        self._stop.set()
