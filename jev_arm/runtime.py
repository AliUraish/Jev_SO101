from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path

from .cameras import Cameras
from .clients import AstraVision, JevJudge
from .controller import Controller
from .models import Config, Decision
from .robot import LeRobotArm


class DecisionWorker:
    def __init__(self, config, cameras, vision, judge):
        self.config, self.cameras, self.vision, self.judge = config, cameras, vision, judge
        self.lock = threading.Lock()
        self.state = None
        self.results = queue.Queue(maxsize=1)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name="astra-jev-decisions")

    def publish(self, state):
        with self.lock:
            self.state = state

    def _run(self):
        while not self.stop.wait(self.config.decision_interval_s):
            if not self.results.empty():
                continue
            with self.lock:
                state = self.state
            if not state or state["active"] or state["fault"] or state["done"]:
                continue
            frames = self.cameras.snapshot()
            if not frames:
                continue
            captured_at = min(frame.captured_at for frame in frames.values())
            if captured_at < state["idle_since"]:
                continue
            start = time.monotonic()
            try:
                scene = self.vision.describe({name: f.jpeg for name, f in frames.items()}, state)
                vision_elapsed = time.monotonic() - start
                # Avoid spending a Jev call on evidence already too old to execute.
                if time.monotonic() - captured_at > self.config.decision_max_age_s:
                    result = {"event": "stale_vision", "revision": state["revision"], "vision_s": vision_elapsed}
                else:
                    answers = self.judge.decide(scene, state)
                    result = {
                        "event": "decision",
                        "decision": Decision(revision=state["revision"], captured_at=captured_at, scene=scene, answers=answers),
                        "robot": state,
                        "vision_s": vision_elapsed,
                        "total_s": time.monotonic() - start,
                    }
            except Exception as exc:
                # Do not log response bodies, image bytes, API keys, or exception request headers.
                result = {"event": "model_error", "revision": state["revision"], "error_type": type(exc).__name__}
            if not self.stop.is_set():
                try:
                    self.results.put_nowait(result)
                except queue.Full:
                    pass

    def close(self):
        self.stop.set()
        self.thread.join(timeout=1)
        # Only close clients after their owner is finished; an in-flight daemon exits with process.
        if not self.thread.is_alive():
            self.vision.close()
            self.judge.close()


def wait_for_cameras(cameras, timeout_s=10.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        frames = cameras.snapshot()
        if frames:
            return frames
        if cameras.errors:
            raise RuntimeError(f"Camera capture failed: {cameras.errors}")
        time.sleep(0.05)
    raise RuntimeError("Both cameras must deliver fresh frames; check indexes and camera permissions")


def run_live(config: Config, log_path: Path):
    config.validate_motion(hardware=True)
    return _run(config, log_path, LeRobotArm(config), Cameras(config))


def run_simulation(config: Config, log_path: Path, robot, cameras, *, status_queue=None, stop_event=None):
    """Use injected simulation adapters while preserving the simulation-only flag."""
    if not config.simulation_only:
        raise ValueError("Simulation runner requires simulation_only=true")
    config.validate_motion()
    if not config.overhead_reference or not config.overhead_reference.is_file():
        raise ValueError("Simulation requires a reference image matching the taught layout")
    return _run(config, log_path, robot, cameras, status_queue=status_queue, stop_event=stop_event)


def _run(config: Config, log_path: Path, robot, cameras, *, status_queue=None, stop_event=None):
    if config.stop_file.exists():
        raise ValueError(f"Local stop is active: {config.stop_file}")
    worker = None
    connected = False
    try:
        cameras.start()
        wait_for_cameras(cameras)
        vision = AstraVision(config)
        try:
            judge = JevJudge(config)
        except Exception:
            vision.close()
            raise
        worker = DecisionWorker(config, cameras, vision, judge)
        robot.connect()
        connected = True
        start = time.monotonic()
        positions = robot.read()
        controller = Controller(config, positions, start)
        worker.publish(controller.state(positions))
        worker.thread.start()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        previous_reason = None
        with log_path.open("a") as log:
            while not controller.done and not controller.fault:
                tick_start = time.monotonic()
                positions = robot.read()
                now = time.monotonic()
                frames_ok = cameras.snapshot() is not None
                stop_requested = config.stop_file.exists() or (stop_event is not None and stop_event.is_set())
                if not stop_requested and now - start > config.max_session_s:
                    controller.abort(
                        f"Session timed out after {config.max_session_s:g}s at {controller.expected_skill}; "
                        f"last status: {controller.reason}", positions
                    )
                if now - tick_start > config.max_tick_gap_s:
                    controller.abort("Robot read exceeded controller deadline", positions)
                # Evaluate local stop/timing/camera state BEFORE accepting a queued model decision.
                target = controller.tick(positions, now, frames_ok, stop_requested)
                try:
                    result = worker.results.get_nowait()
                except queue.Empty:
                    result = None
                if result:
                    event = dict(result)
                    if result["event"] == "decision":
                        event["decision"] = result["decision"].model_dump()
                        if frames_ok and not controller.fault:
                            controller.accept(result["decision"], positions, time.monotonic())
                    elif result["event"] == "model_error" and result["revision"] == controller.revision:
                        controller.abort("Model request failed: " + result["error_type"], positions)
                    elif result["event"] == "stale_vision" and result["revision"] == controller.revision and not controller.fault:
                        controller.reason = (
                            f"Holding: Astra took {result['vision_s']:.1f}s; observation exceeded "
                            f"the {config.decision_max_age_s:g}s freshness limit"
                        )
                    log.write(json.dumps(event) + "\n")
                    log.flush()
                if controller.fault:
                    robot.hold()
                else:
                    robot.send(target)
                worker.publish(controller.state(positions))
                if controller.reason != previous_reason:
                    if status_queue is not None:
                        status_queue.put_nowait({"stage": controller.expected_skill, "reason": controller.reason,
                                                "active": controller.active})
                    print(controller.reason, flush=True)
                    log.write(json.dumps({"event": "controller", "reason": controller.reason, "stage": controller.expected_skill, "at": time.monotonic()}) + "\n")
                    log.flush()
                    previous_reason = controller.reason
                time.sleep(max(0, 1 / config.control_hz - (time.monotonic() - tick_start)))
        if controller.fault:
            raise RuntimeError(controller.fault)
    finally:
        # Freeze hardware BEFORE waiting on camera/model shutdown.
        try:
            if connected:
                try:
                    robot.hold()
                finally:
                    robot.close()
        finally:
            try:
                if worker:
                    if worker.thread.ident is not None:
                        worker.close()
                    else:
                        worker.vision.close()
                        worker.judge.close()
            finally:
                cameras.close()
