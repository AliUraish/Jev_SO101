from __future__ import annotations

from dataclasses import dataclass
import threading
import time

from .models import Config


@dataclass(frozen=True)
class Frame:
    jpeg: bytes
    captured_at: float


class Cameras:
    """One capture worker per USB camera; retain only the latest frame from each."""

    def __init__(self, config: Config):
        self.config = config
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.frames: dict[str, Frame] = {}
        self.errors: dict[str, str] = {}
        self.threads = []

    def start(self):
        for name, camera in self.config.cameras.items():
            thread = threading.Thread(target=self._capture, args=(name, camera), daemon=True, name=f"camera-{name}")
            thread.start()
            self.threads.append(thread)

    def _capture(self, name, camera):
        import cv2
        cap = cv2.VideoCapture(camera.source)
        try:
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open {name} camera")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera.height)
            cap.set(cv2.CAP_PROP_FPS, camera.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            while not self.stop.is_set():
                # Timestamp before the read: a slow/blocking read must not make an old frame look fresh.
                captured_at = time.monotonic()
                ok, image = cap.read()
                if not ok:
                    raise RuntimeError(f"Lost {name} camera")
                ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok:
                    raise RuntimeError(f"Cannot encode {name} frame")
                with self.lock:
                    self.frames[name] = Frame(encoded.tobytes(), captured_at)
        except Exception as exc:
            with self.lock:
                self.errors[name] = type(exc).__name__
                self.frames.pop(name, None)
        finally:
            cap.release()

    def snapshot(self) -> dict[str, Frame] | None:
        with self.lock:
            frames = dict(self.frames)
        if set(frames) != {"overhead", "wrist"}:
            return None
        times = [f.captured_at for f in frames.values()]
        now = time.monotonic()
        if now - min(times) > self.config.camera_max_age_s:
            return None
        if max(times) - min(times) > self.config.camera_max_skew_s:
            return None
        return frames

    def close(self):
        self.stop.set()
        for thread in self.threads:
            thread.join(timeout=1)
