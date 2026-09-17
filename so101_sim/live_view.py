"""Loopback-only camera viewer. HTTP and JPEG work never runs on the control thread."""
from __future__ import annotations

import json
import queue
import secrets
import socketserver
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .jev_arm_adapter import encode_jpeg


class LoopbackServer(ThreadingHTTPServer):
    def server_bind(self):
        # HTTPServer normally reverse-resolves the address, which can stall macOS startup.
        socketserver.TCPServer.server_bind(self)
        self.server_name = "localhost"
        self.server_port = self.server_address[1]


class LiveView:
    def __init__(self, sim):
        self.sim = sim
        self.updates = queue.SimpleQueue()
        self.stop_requested = threading.Event()
        self.closed = threading.Event()
        self._shutdown = threading.Event()
        self._lock = threading.Lock()
        self._started = time.monotonic()
        self._frames = {}
        self._state = {"stage": "approach", "reason": "Waiting for a fresh observation", "active": False,
                       "finished": False, "success": False, "elapsed_s": 0, "frame_id": 0}
        self._token = secrets.token_urlsafe(24)
        self._server = None
        self._capture_thread = None
        self._http_thread = None

    def start(self, open_browser=True):
        view = self
        page = Path(__file__).with_name("live_view.html").read_text().replace("__TOKEN__", self._token).encode()

        class Handler(BaseHTTPRequestHandler):
            def reply(self, code, content, content_type):
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                try:
                    self.wfile.write(content)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_GET(self):
                path = urlsplit(self.path).path
                if path == "/":
                    return self.reply(200, page, "text/html; charset=utf-8")
                if path == "/status":
                    return self.reply(200, json.dumps(view.snapshot()).encode(), "application/json")
                if path in ("/overhead.jpg", "/wrist.jpg"):
                    with view._lock:
                        frame = view._frames.get(path[1:-4])
                    return self.reply(200 if frame else 503, frame or b"Waiting for camera", "image/jpeg")
                self.reply(404, b"Not found", "text/plain")

            def do_POST(self):
                if not secrets.compare_digest(self.headers.get("X-View-Token", ""), view._token):
                    return self.reply(403, b"Forbidden", "text/plain")
                if self.path not in ("/stop", "/close"):
                    return self.reply(404, b"Not found", "text/plain")
                view.stop_requested.set()
                if self.path == "/close":
                    view.closed.set()
                self.reply(200, b"{}", "application/json")

            def log_message(self, *_):
                pass

        self._server = LoopbackServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self._server.server_port}/"
        self._http_thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
        self._capture_thread = threading.Thread(target=self._capture_loop, name="sim-live-view", daemon=True)
        self._http_thread.start()
        self._capture_thread.start()
        print(f"Live simulation: {self.url}", flush=True)
        if open_browser:
            try:
                webbrowser.open(self.url)
            except webbrowser.Error:
                print("Open the live simulation URL in your browser.", flush=True)

    def snapshot(self):
        with self._lock:
            state = dict(self._state)
        if not state["finished"]:
            state["elapsed_s"] = time.monotonic() - self._started
        return state

    def _capture_loop(self):
        previous_capture = None
        while not self._shutdown.is_set():
            update = None
            while True:
                try:
                    update = self.updates.get_nowait()
                except queue.Empty:
                    break
            if update:
                with self._lock:
                    if not self._state["finished"]:
                        self._state.update(update)
            try:
                frames = self.sim.get_camera_frames()
                captured = min((f.captured_at for f in frames.values()), default=None)
                if len(frames) == 2 and captured != previous_capture:
                    encoded = {name: encode_jpeg(f.rgb) for name, f in frames.items()}
                    with self._lock:
                        self._frames = encoded
                        self._state["frame_id"] += 1
                    previous_capture = captured
            except Exception as exc:
                with self._lock:
                    self._state["camera_error"] = type(exc).__name__
            self._shutdown.wait(0.1)

    def finish(self, success, error=None):
        self._shutdown.set()
        if self._capture_thread:
            self._capture_thread.join(timeout=2)
        # Capture the settled final view before the simulator is closed.
        frames = self.sim.render_now()
        encoded = {name: encode_jpeg(f.rgb) for name, f in frames.items()}
        with self._lock:
            self._frames = encoded
            self._state.update(finished=True, success=bool(success), active=False,
                               elapsed_s=time.monotonic() - self._started,
                               reason="Block placed successfully" if success else (error or "Verification failed"))
            self._state["frame_id"] += 1

    def wait_closed(self):
        print("Final view stays open. Click Close viewer or press Ctrl+C here to exit.", flush=True)
        try:
            while not self.closed.wait(0.2):
                pass
        except KeyboardInterrupt:
            pass

    def close(self):
        self._shutdown.set()
        if self._capture_thread:
            self._capture_thread.join(timeout=2)
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        if self._http_thread:
            self._http_thread.join(timeout=2)
