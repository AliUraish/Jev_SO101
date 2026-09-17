from __future__ import annotations

from .models import Config, Decision, JOINTS, PRIMITIVES, STAGES


class Controller:
    """Deterministic state machine; called only by the thread owning the robot bus."""

    def __init__(self, config: Config, positions: dict[str, float], now: float):
        config.validate_motion()
        config.check_pose(positions)
        if any(abs(positions[j] - config.start_pose[j]) > config.joint_limits[j].arrival_tolerance for j in JOINTS):
            raise ValueError("Arm must already be at the recorded start pose; automatic homing is disabled")
        self.config = config
        self.index = 0
        self.revision = 0
        self.idle_since = now
        self.last_tick = now
        self.last_capture = -1.0
        self.last_command = dict(positions)
        self.active = False
        self.fault: str | None = None
        self.done = False
        self.reason = "Waiting for a fresh observation"
        self.confirmations = 0
        self.arrival_ticks = 0

    @property
    def expected_skill(self):
        return STAGES[self.index]

    def state(self, positions: dict[str, float]) -> dict:
        return {
            "revision": self.revision,
            "expected_skill": self.expected_skill,
            "next_primitive": PRIMITIVES[self.expected_skill],
            "active": self.active,
            "idle_since": self.idle_since,
            "completed_skills": list(STAGES[:self.index]),
            "joints": dict(positions),
            "units": "arm degrees; gripper 0–100",
            "fault": self.fault,
            "done": self.done,
        }

    def abort(self, reason: str, positions: dict[str, float]):
        if not self.fault:
            self.fault = reason
            self.reason = reason
            self.active = False
            self.revision += 1
            self.last_command = dict(positions)

    def _precondition(self, decision: Decision) -> str | None:
        scene = decision.scene
        if scene.hazards or scene.human_in_workspace == "yes":
            return "hazard"
        if scene.human_in_workspace != "no" or scene.uncertainties:
            return "Visual evidence is uncertain"
        if scene.glass_upright != "yes" or scene.glass_at_reference != "yes":
            return "Glass must be upright and at its taught location"
        stage = self.expected_skill
        if stage in ("approach", "grasp"):
            if scene.block_at_reference != "yes" or scene.block_in_gripper != "no":
                return "Block must be visible at the taught pickup location"
        if stage in ("lift", "place", "release") and scene.block_in_gripper != "yes":
            return "Need visual confirmation that the gripper holds the block"
        if stage in ("approach", "grasp", "place", "release"):
            if scene.glass_opening_clear != "yes" or scene.block_fits_opening != "yes":
                return "Glass opening must be clear and large enough for the block"
        if stage == "release" and scene.gripper_above_glass != "yes":
            return "Release requires gripper centered above the glass opening"
        return None

    def accept(self, decision: Decision, positions: dict[str, float], now: float):
        if self.active or self.fault or self.done:
            return
        if decision.revision != self.revision:
            self.reason = "Discarded a decision from another controller stage"
            return
        age = now - decision.captured_at
        if age < 0 or age > self.config.decision_max_age_s or decision.captured_at < self.idle_since:
            self.reason = "Discarded a stale observation"
            self.confirmations = 0
            return
        if decision.captured_at <= self.last_capture:
            self.reason = "Discarded a duplicate observation"
            return
        self.last_capture = decision.captured_at
        answers = decision.answers
        precondition = self._precondition(decision)
        if answers.unsafe.noul >= self.config.max_unsafe_probability or precondition == "hazard":
            detail = "; ".join(decision.scene.hazards) or "Jev rejected the next motion"
            self.abort(f"Unsafe decision (p={answers.unsafe.noul:.2f}): {detail}", positions)
            return
        if precondition:
            self.reason = precondition
            self.confirmations = 0
            return
        task = answers.task
        if task.choice != self.expected_skill or task.confidence < self.config.min_confidence or task.probabilities[task.choice] < self.config.min_task_probability:
            self.reason = "Holding: next skill not confidently supported"
            self.confirmations = 0
            return
        if self.expected_skill == "verify":
            scene = decision.scene
            if answers.done.noul >= self.config.min_done_probability and scene.block_inside_glass == "yes" and scene.block_in_gripper == "no":
                self.confirmations += 1
            else:
                self.confirmations = 0
            self.done = self.confirmations >= 2
            self.reason = "Goal visually confirmed" if self.done else "Waiting for two independent visual confirmations"
            return
        self.active = True
        self.active_since = now
        self.waypoint_index = 0
        self._start_segment(positions, now)
        self.reason = f"Executing {self.expected_skill}"

    def _start_segment(self, positions: dict[str, float], now: float):
        waypoint = self.config.skills[self.expected_skill][self.waypoint_index]
        self.segment_origin = dict(positions)
        self.segment_target = waypoint.joints
        self.segment_start = now
        # Smoothstep has a peak derivative of 1.5. Stretch time to enforce configured speeds.
        speed_duration = max(1.5 * abs(waypoint.joints[j] - positions[j]) / self.config.joint_limits[j].speed_per_s for j in JOINTS)
        self.segment_duration = max(waypoint.duration_s, speed_duration)
        self.arrival_ticks = 0

    def tick(self, positions: dict[str, float], now: float, cameras_ok: bool = True, stop_requested: bool = False) -> dict[str, float]:
        self.config.check_pose(positions)
        gap = now - self.last_tick
        self.last_tick = now
        if stop_requested:
            self.abort("Local stop requested", positions)
        if self.active and (gap < 0 or gap > self.config.max_tick_gap_s):
            self.abort("Controller missed its timing deadline", positions)
        if self.active and not cameras_ok:
            self.abort("Camera stream lost or stale", positions)
        if self.fault:
            return dict(self.last_command)
        if any(abs(positions[j] - self.last_command[j]) > self.config.joint_limits[j].tracking_tolerance for j in JOINTS):
            self.abort("Joint tracking error exceeded configured tolerance", positions)
        if not self.active:
            return dict(self.last_command)
        if now - self.active_since > self.config.command_timeout_s:
            self.abort("Primitive exceeded command deadline", positions)
            return dict(self.last_command)
        elapsed = now - self.segment_start
        if elapsed > self.segment_duration + self.config.settle_timeout_s:
            self.abort("Arm failed to reach waypoint", positions)
            return dict(self.last_command)
        t = min(1.0, max(0.0, elapsed / self.segment_duration))
        blend = t * t * (3.0 - 2.0 * t)
        target = {j: self.segment_origin[j] + blend * (self.segment_target[j] - self.segment_origin[j]) for j in JOINTS}
        self.config.check_pose(target)
        # Also cap per-tick target changes; timing jitter must never create a large jump.
        dt = max(0.0, min(gap, 1.0 / self.config.control_hz))
        for j in JOINTS:
            bound = self.config.joint_limits[j].speed_per_s * dt
            target[j] = max(self.last_command[j] - bound, min(self.last_command[j] + bound, target[j]))
        self.last_command = target
        reached = t == 1.0 and all(abs(positions[j] - self.segment_target[j]) <= self.config.joint_limits[j].arrival_tolerance for j in JOINTS)
        self.arrival_ticks = self.arrival_ticks + 1 if reached else 0
        if self.arrival_ticks >= 3:
            self.waypoint_index += 1
            if self.waypoint_index < len(self.config.skills[self.expected_skill]):
                self._start_segment(positions, now)
            else:
                self.active = False
                self.index += 1
                self.revision += 1
                self.idle_since = now
                self.reason = f"Waiting for {self.expected_skill} decision"
        return dict(self.last_command)
