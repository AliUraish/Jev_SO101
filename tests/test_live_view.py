import time
from types import SimpleNamespace

import httpx
import numpy as np

from so101_sim.live_view import LiveView


def test_live_frames_status_stop_and_final_view():
    class Sim:
        color = 0

        def get_camera_frames(self):
            return {name: SimpleNamespace(rgb=np.full((12, 16, 3), self.color, dtype=np.uint8),
                                          captured_at=time.monotonic()) for name in ("overhead", "wrist")}

        def render_now(self):
            return self.get_camera_frames()

    sim = Sim()
    view = LiveView(sim)
    view.start(open_browser=False)
    try:
        with httpx.Client(base_url=view.url, trust_env=False) as client:
            assert client.get("/").status_code == 200
            assert client.get("/../../.env").status_code == 404
            deadline = time.monotonic() + 2
            while client.get("/status").json()["frame_id"] == 0 and time.monotonic() < deadline:
                time.sleep(0.01)
            first = client.get("/overhead.jpg")
            assert first.status_code == 200 and first.content.startswith(b"\xff\xd8")
            sim.color = 255
            view.updates.put({"stage": "lift", "reason": "Executing lift", "active": True})
            deadline = time.monotonic() + 2
            while client.get("/status").json()["stage"] != "lift" and time.monotonic() < deadline:
                time.sleep(0.01)
            assert client.get("/status").json()["active"]
            assert client.get("/overhead.jpg").content != first.content
            assert client.post("/stop").status_code == 403
            assert not view.stop_requested.is_set()
            headers = {"X-View-Token": view._token}
            assert client.post("/stop", headers=headers).status_code == 200
            assert view.stop_requested.is_set()
            view.finish(False, "Local stop requested")
            state = client.get("/status").json()
            assert state["finished"] and not state["success"] and not state["active"]
            assert state["reason"] == "Local stop requested"
            assert client.get("/wrist.jpg").status_code == 200
            assert client.post("/close", headers=headers).status_code == 200
            assert view.closed.is_set()
    finally:
        view.close()
