# SO-101 block-into-cup simulation (MuJoCo)

A runnable simulation of the SO-101 arm on a wooden tabletop with one blue block and one red party cup,
matching the reference photo of the physical setup. It supplies two camera images and joint measurements,
accepts joint commands in the controller's units, keeps physics running in wall-clock time while external
models are thinking, and reports evaluator-only ground truth.

```
so101_sim/
  assets/so101/        robot MJCF + STL meshes (google-deepmind/mujoco_menagerie `robotstudio_so101`, Apache-2.0)
  assets/generated/    wood.png, cup_outer.stl, cup_inner.stl  (procedural; rebuild with `python -m so101_sim.build_assets`)
  scene.py             SceneConfig (all geometry in meters) and MJCF generation
  sim.py               SO101Sim: reset / read / send / step / cameras / ground truth / close
  units.py             joint-unit conversion (MuJoCo radians <-> degrees, gripper 0-100)
  ik.py                damped least-squares IK on the tool center point (used to plan waypoints)
  task.py              IK-planned skill waypoints, scripted executor, evaluator, jev_arm config export
  jev_arm_adapter.py   SimArm / SimCameras with the same call shapes as jev_arm.robot / jev_arm.cameras
  examples/            minimal.py, scripted_pick_place.py, run_jev_arm_in_sim.py
  media/pick_place.mp4 recording of the scripted run (overhead | wrist)
tests/test_sim.py      unit mapping, deterministic reset, RGB frames, real-time physics, scripted task, jev_arm controller in sim
```

## Install and launch

Tested on this machine: MacBook with Apple M4 (10 cores), macOS 26.5.2, no discrete/CUDA GPU. MuJoCo physics runs on
the CPU (about 3% of one core at real time); rendering uses MuJoCo's offscreen CGL context on the Apple GPU (about
40 ms per 640x480 camera pair, 15 Hz stream). The simulator runs in-process, so there is nothing to connect to over the
network. On Linux, set `MUJOCO_GL=egl` for headless rendering. Python 3.11, MuJoCo 3.13.0, numpy 2.2.6, opencv 4.12.

```sh
# from the project directory, using the existing .venv
uv pip install --python .venv/bin/python "mujoco>=3.3,<4" numpy        # or: uv pip install -e ".[sim]"
.venv/bin/python -m so101_sim.examples.minimal                          # reset, read both images, move a joint
.venv/bin/python -m so101_sim.examples.scripted_pick_place              # real-time scripted pick-and-place (~25 s)
.venv/bin/python -m so101_sim.examples.scripted_pick_place --fast --video so101_sim/media/pick_place.mp4 \
    --export-config config.sim.json --reference captures/sim/overhead_reference.jpg
.venv/bin/python -m pytest -q tests/test_sim.py
```

`--export-config` writes a `config.local.json`-compatible file with the start pose, joint limits, and the six
planned skills (approach, grasp, lift, place, release, retract) in controller units, satisfying the jev_arm
validation rules (approach keeps the start opening, closure is a stationary gripper-only waypoint, release only
opens the gripper at the place pose, retract keeps the released opening).

## 1. Robot model

Source: `robotstudio_so101` from mujoco_menagerie (MJCF derived from TheRobotStudio's `so101_new_calib.xml`), with
these edits, marked `so101_sim` in `assets/so101/so101.xml`:
the menagerie wrist camera is renamed `wrist` and given a 640x480 / 60 deg pinhole definition, and a `tcp` site is
added to the gripper body (tool center point between the jaw pads at a 25 mm opening).

* Fixed base at the world origin; the base bottom sits on the table top (z = 0).
* Six hinge joints named exactly as the controller expects. Ranges from the MJCF:

| Joint | MJCF range (deg) | Controller units | Sim conversion |
| --- | --- | --- | --- |
| shoulder_pan | -110 .. +110 | degrees | `deg = degrees(rad)`; + turns the arm toward -y (clockwise from above) |
| shoulder_lift | -100 .. +100 | degrees | `deg = degrees(rad)`; + lowers the arm toward the table |
| elbow_flex | -96.8 .. +96.8 | degrees | `deg = degrees(rad)`; + lowers the forearm |
| wrist_flex | -95 .. +95 | degrees | `deg = degrees(rad)`; + pitches the gripper down |
| wrist_roll | -157.2 .. +157.2 | degrees | `deg = degrees(rad)`; + is right-hand about the gripper axis |
| gripper (moving jaw hinge) | -10 .. +100 | 0-100 opening | `pct = 100 * (rad - (-0.174533)) / (1.7453292 - (-0.174533))` |

Zero for every arm joint is the MJCF zero: arm fully extended horizontally along +x, TCP at (0.39, 0, 0.25) m.
Sign is +1 and zero offset is 0 for all arm joints (`units.ARM_CONVERSION`, one place to change). This mapping
is for the simulated arm only; the physical LeRobot calibration stays separate and is not used here.

Gripper 0-100 is a normalized hinge position, not millimeters. Measured fingertip gap (spheres at the jaw tips):

| gripper % | 0 | 9 | 18 | 27 | 36 | 45 | 55 | 64 | 73 | 82 | 91 | 100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gap (mm) | 4 | 16 | 30 | 44 | 57 | 70 | 82 | 94 | 105 | 116 | 125 | 133 |

* Actuators: position servos per joint (menagerie `sts3215` class, kp 998, kv 2.73, force limit +/-2.94 N m,
  derived for an STS3215 at P-gain 16). The gripper's force limit is lowered to 0.5 N m
  (`SceneConfig.gripper_force_limit_nm`): at the stall torque the jaw squeezes the 25 mm block with 39 N and the
  block leaves the jaws at up to 1 m/s on release; at 0.5 N m the squeeze is 6.7 N and the release is clean.
  Commanding the gripper to 8% on the block stalls the jaw near 16%, so the controller's gripper arrival
  tolerance is 12 units in the exported config.
* Block contacts use a 10 ms time constant with priority over the pads and cup (`SceneConfig.block_solref`).
  At the 4 ms minimum for the 2 ms timestep, corner impacts inside the cup rebounded at 80%+ and knocked the
  cup over; 10 ms gives a 10% rebound.
* Collision: primitive boxes for the arm links; the gripper uses the menagerie hand-tuned set (boxes, tip spheres,
  small convex meshes) with `condim=6`, friction 1.0 and soft `solref` for grasping. Link masses and full inertia
  tensors come from the CAD-derived MJCF.

## 2. Tabletop environment (meters, world frame at the robot base, +z up, table top at z = 0)

| Object | Geometry | Start position (center) | Mass |
| --- | --- | --- | --- |
| Table | box 0.80 x 0.70 x 0.04, top at z = 0, wood texture, friction 0.9 | (0.25, 0, -0.02) | static |
| Block | cube 0.025, blue, friction 0.9, condim 4 | (0.22, 0.06, 0.0125) | 0.015 kg |
| Cup | tapered 9 oz party cup: height 0.095, outer radius 0.040 at rim / 0.028 at base, red outside, white inside | base at (0.22, -0.09, 0) | 0.030 kg |
| Distractor block (optional, `distractor_block=True`) | same as block | (0.30, 0.11, 0.0125) | 0.015 kg |
| Arm start pose | pan 0, lift -35, elbow 60, wrist_flex 10, roll 0 (deg), gripper 22 | TCP at (0.29, 0, 0.08) | |

The cup collision is a ring of 24 thin tilted boxes (3 mm) plus a floor (a box by default, `cup_floor_shape`):
a real opening and hollow interior; the block (25 mm) passes a 74 mm opening and the interior radius is 35.7 mm
1 cm below the rim, where it is released.
The visual cup is two thin meshes (outer red, inner off-white); set `cup_transparent_alpha=0.35` in `SceneConfig`
for a translucent glass look instead. The block is grasped and released purely through contact physics.

Lighting: one shadow-casting warm key light above the operator side (the bright pool on the table in the photo),
a dim cool fill, dark back wall and gradient sky. The baseline has no cable across the workspace. Set
`show_cable=True` to restore the cosmetic cable as a visual hazard stress case (it has no collision).
Object positions are fixed by default; `block_jitter` / `cup_jitter` (m) plus the reset
seed add reproducible jitter when you want it.

## 3. Cameras

| Camera | Mount | Resolution | FOV / intrinsics | Pose |
| --- | --- | --- | --- | --- |
| overhead | fixed, steep view into cup interior | 640x480 | fovy 50 deg; fx = fy = 514.7 px, cx 320, cy 240 | position (0.35, 0, 0.65) m, looking at (0.18, 0, 0) m |
| wrist | body `camera_mount`, child of `gripper` (moves with wrist roll) | 640x480 | fovy 60 deg; fx = fy = 415.7 px, cx 320, cy 240 | pos (0, 0.055, -0.045) m, euler (-0.57, 0, 0) rad relative to the gripper body |

Frames are `numpy uint8 (H, W, 3)` in **RGB** order (convert with `cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)` for OpenCV
and JPEG encoding; `jev_arm_adapter.encode_jpeg` does this). Each `Frame` carries `sim_time` (simulation seconds)
and `captured_at` (`time.monotonic()` when physics state is copied), plus `intrinsics`. Both views share one
state snapshot and capture timestamp. GPU rendering runs outside the physics lock so it cannot block joint
reads or commands. The render thread keeps only the newest frame
per camera, at 15 Hz by default (`render_hz`). The overhead reference image of the taught layout is written by
`scripted_pick_place.py --reference PATH`.

## 4. Interface (`so101_sim.SO101Sim`)

| Operation | Purpose |
| --- | --- |
| `SO101Sim(scene_config=None, realtime=True, render_hz=15)` | Build model; start render thread; start real-time physics if `realtime` |
| `reset(seed=None, scene_config=None) -> joints` | Reproducible start scene (objects re-placed, arm at start pose, 0.3 s settle). New `scene_config` rebuilds geometry |
| `read_joint_positions() -> {joint: value}` | All six joints in controller units |
| `read_joint_velocities()` | deg/s and %/s |
| `send_joint_targets({joint: value}) -> applied` | Position targets in controller units (clipped to actuator range); missing joints keep their last target |
| `get_camera_frames() -> {"overhead": Frame, "wrist": Frame}` | Newest frame per camera |
| `step(dt)` | Advance physics synchronously (paused/debug mode) |
| `start()` / `pause()` / `running` | Real-time physics on/off. While running, `step` is unnecessary |
| `get_ground_truth() -> dict` | Evaluator-only state (below) |
| `close()` | Stop threads |
| `sim_time`, `stats` | Simulation clock; steps, physics CPU time, render timing, worst catch-up drop |

Real-time mode advances physics from wall-clock time in a background thread; nothing blocks while Astra or Jev
respond. If the process stalls for more than 0.25 s (`max_catchup_s`) the clock is dropped instead of spiraling,
and `stats["behind_s_max"]` records it. Timestep 0.002 s, `implicitfast`, elliptic cones.

Minimal example (`examples/minimal.py`): construct, `reset(seed=0)`, print both frames' shapes and timestamps,
command `shoulder_pan = -30`, watch it move while sleeping, print ground truth.

## 5. Ground truth for the evaluator (never send to the models)

`get_ground_truth()["flags"]`:

| Flag | Meaning |
| --- | --- |
| `block_lifted` | block center > 1 cm above rest height and not touching the table |
| `block_held` | block in contact with both the fixed jaw and the moving jaw |
| `block_inside_cup` | block within the cup interior frustum (cup frame) between floor and rim, cup upright |
| `block_settled` | block linear speed < 1 cm/s and angular speed < 0.2 rad/s |
| `cup_tipped` | cup tilt > 30 deg or fallen |
| `block_on_floor` | block left the table |
| `arm_touching_table`, `arm_touching_cup` | any arm/jaw geom in contact with the table / cup |
| `success` | `block_inside_cup and block_settled and not block_held and not cup_tipped` |

Also included: block and cup poses (position, quaternion w-x-y-z), velocities, cup tilt angle, TCP position and
axes, the block's position in the cup frame, and the full contact list with normal forces. `task.evaluate(sim)`
waits for motion to stop and returns the success verdict, which is the completion criterion (not the last waypoint).

## 6. Scripted validation and connecting the pipeline

`task.plan_skills()` computes the six-stage path from the scene geometry with IK. Design points, each found by
measurement in this simulator (see `TaskParameters` for the numbers):

* Top-down grasp with the jaws closing tangentially (wrist roll ~90 deg). The open jaw faces sit at -13.6 mm
  (fixed) and +23.9 mm (moving) from the tool point, so the approach is offset 5 mm to center the block.
* Transport at 12.5 cm with a 20 deg radial tilt: a pure top-down pose at that height is outside the arm's reach.
* Entry into the cup at 12 deg, release at 10 deg with the block 1 cm below the rim, gripper opened only to 22%,
  then a straight lift-out at the same tilt. The open gripper spans -24..+34 mm, so the cup-side targets are
  offset -5 mm to center it in the 35.7 mm interior radius.
* A high clearing pose (16 cm, 35 deg) before the home pose: the joint-space path straight from above the cup to
  home sweeps the fixed jaw through the rim.

Executed with the controller-style smoothstep ramp (`run_skills(ramp=True)`), the sequence succeeds in 16/16
perturbed trials (0-0.5 s timing jitter, half with 3 mm block jitter) with no cup tipping. `tests/test_sim.py`
runs it through contact physics and also drives the simulator with the unmodified
`jev_arm.controller.Controller` using the exported waypoints and the demo's synthetic decisions.

To run the real pipeline (Astra + Jev + controller) against the simulator without editing pipeline code:

```sh
.venv/bin/python -m so101_sim.examples.scripted_pick_place --fast --export-config config.sim.json --reference captures/sim/overhead_reference.jpg
.venv/bin/python -m so101_sim.examples.run_jev_arm_in_sim --config config.sim.json     # needs both API keys
```

The real-API command opens a live browser viewer automatically. It shows overhead and wrist camera
feeds, the current motion/decision stage, elapsed time, and a Stop simulation button. JPEG encoding
and HTTP serving run in background threads; the viewer cannot issue joint commands. Stop requests
are handled by the controller on its next tick. Closing the browser tab alone does not stop the run;
use the Stop button or Ctrl+C in the terminal.

After the run, the final camera view stays available until you click Close viewer or press Ctrl+C.
The terminal prints the local viewer URL if your browser does not open automatically. Add `--headless`
to run with terminal output and saved final images only.

That script calls the shared runtime with explicit `SimArm` / `SimCameras` adapters. The config stays
`simulation_only=true`; no hardware shim or placeholder calibration is created. Old `*.hardware-shim.json`
files are no longer used. The simulator keeps running in real time while Astra and Jev are called,
so stale-observation handling is exercised.

The baseline overhead camera is steep enough to inspect the cup interior after retraction. Regenerate
the reference image with the export command above after changing camera position or scene geometry.
Astra uses the two views together; normal wrist-camera blind spots are not blocking uncertainties when
the other camera supplies the needed evidence. Unknown facts needed for the current skill still block
motion, and reported hazards still latch a stop. A controlled stop prints the
actual reason, including reported hazards, rather than a Python traceback.

Jev receives the current stage, observe, and hold as its eligible choices, along with explicit
prerequisites and a description of the whole primitive. For example, grasp descends before closing;
place transfers the held block before aligning it with the opening. The controller independently
checks the observed prerequisites, skill order, freshness, probabilities, and joint limits.

The simulation uses an experimental `max_unsafe_probability=0.20`: initial live replays gave clear
scenes scores around 0.08–0.11, which repeatedly tripped the physical default of 0.10. The physical
default remains 0.10. This small replay set is not a calibration study or evidence of hardware safety.
Reported hazards, humans, unknown required facts, and contradictory prerequisites still block motion
independently of this score. Jev retains all scene facts and object descriptions; uncalibrated pixel
boxes and joint angles are omitted from its decision input. Full observations remain in the log.

Each API run saves a `.summary.json` beside its JSONL log, plus final `.overhead.jpg` and `.wrist.jpg`
frames. The summary separates model completion from the independent physics evaluator's success.
A controlled stop or disagreement with the evaluator exits with a nonzero status.

Live baseline verification: `logs/sim_experimental_cutoff.jsonl` completed all six motion stages plus
two fresh verification observations in 137.28 seconds using real Astra and Jev requests. Both model
completion and the physics evaluator reported success; the block settled inside the upright cup and
the arm was clear. This is one successful baseline run, not a measured reliability rate or hardware validation.

The exported simulation config permits observations up to 30 seconds old, accommodating measured
Astra latency of 13–18 seconds in this fixed-layout scene. The physical-arm default remains 8 seconds.
Camera freshness, skill order, uncertainty and hazard gates still apply. The session limit remains
180 seconds. A timeout now reports the stage and last controller status; it is distinct from a local
`STOP` request. This simulation allowance is not a validated setting for a changing physical workspace.

## Known limits and tuning notes

* The 9 oz cup is small relative to the gripper: release must happen with the block inside the rim, gripper
  opened to no more than about 24% and tilted no more than ~12 deg; wider openings, more tilt, or dropping from
  above the rim tips the cup. These margins are what the physical task also has, so they are left tight on
  purpose. Cup mass is 0.030 kg (a real empty cup is 10-15 g; heavier makes it slightly less tippy).
* Stepped position commands (no ramp) are harsher than the real controller and occasionally shake the block
  out during the lift; the executor ramps by default.
* Joint-space interpolation between taught waypoints is not a straight Cartesian path; the planned points were
  chosen so the segments clear the table and cup.
* No camera noise, motion blur, or auto-exposure; the visual task is easier than the photo. Add jitter and
  `cup_transparent_alpha` when you want it harder.
* No IK/vision-based re-targeting to moved objects yet; that is the next step once the fixed layout works.
