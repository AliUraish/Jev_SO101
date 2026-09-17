# Astra + Jev: block into a glass

A Python prototype for an SO-101 with an overhead USB camera and a wrist camera.
Astra interprets both views, Jev selects a typed skill and evaluates unsafe/done questions,
and a separate local controller executes taught joint waypoints with a 30 Hz target.

**Current status:** API adapters, dual-camera capture, waypoint recording, controller,
offline demo, and automated tests are implemented. Live provider calls and physical
motion still need your keys, calibration, camera indexes, and taught workspace poses.
The demo is synthetic orchestration, not a physics simulation or proof of grasp success.

## Install and try the offline demo

From the project directory, using Python 3.11:

```sh
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-dev.lock
.venv/bin/python -m jev_arm demo
.venv/bin/python -m pytest -q
```

The local `.venv` has already been created and tested. `requirements-dev.lock` pins the
tested camera, API, and test dependencies. `pyproject.toml` defines the package and an
optional `jev-arm` entry point if installed with `pip install -e .`.

LeRobot is intentionally separate: use your existing calibrated LeRobot environment
and install this package into it, or install LeRobot with Feetech support into this venv
following the [official installation guide](https://huggingface.co/docs/lerobot/installation).
The adapter supports the documented `so_follower` imports and the older
`so101_follower` import location. Hardware compatibility with your installed version
must be checked when you provide the calibration; LeRobot has not been installed here.

## Configure credentials and the two cameras

```sh
cp .env.example .env
cp config.example.json config.local.json
```

Edit `.env` locally:

```dotenv
OPENAI_API_KEY=your_openai_api_key
TYPESAFE_API_KEY=your_typesafe_api_key
```

Both values are loaded from the environment or the `.env` beside your config.
They are not stored in scene logs. `.env`, local configuration, calibration, captures,
and logs are gitignored. Camera images are sent to OpenAI; scene JSON and robot state
are sent to TypeSafe. No API key is needed for the offline demo or camera capture.

Set `cameras.overhead.source` and `cameras.wrist.source` in `config.local.json`.
The example's `0` and `1` are placeholders; verify which device is which. Local USB
device paths may also be used where OpenCV supports them. macOS may require camera
permission for the terminal that launches Python. Each camera has its own capture
thread, and only its newest frame is kept.

```sh
.venv/bin/python -m jev_arm capture --output-dir captures/setup
```

Inspect both resulting JPEGs. With the block on its pickup mark and the upright glass
at its fixed placement mark, set `overhead_reference` to `captures/setup/overhead.jpg`.
This reference anchors the taught layout; the wrist view moves with the arm.

## Check Astra and Jev without moving the arm

```sh
.venv/bin/python -m jev_arm observe
```

This sends a fresh overhead/wrist pair to Astra, then sends the validated scene to Jev.
It prints the scene and decisions and saves `logs/observation.json`. It never connects
to the robot. You can also evaluate saved images:

```sh
.venv/bin/python -m jev_arm observe \
  --overhead captures/setup/overhead.jpg \
  --wrist captures/setup/wrist.jpg \
  --stage approach
```

An observation contains image regions, object identities, gripper evidence, hazards,
unknowns, glass alignment, and visible evidence of a block inside the glass. Pixels
are never treated as robot coordinates. Missing/ambiguous information produces a hold.

## Load your existing calibration

In `config.local.json`, set:

- `robot_id`: the ID used during LeRobot calibration.
- `robot_port`: your follower arm's serial device.
- `calibration_file`: the path to that calibration JSON, named `<robot_id>.json`.

Paths are relative to the config file unless absolute. Existing calibration is loaded
and compared with the motors; it is never overwritten or automatically regenerated.
A mismatch must be reconciled through LeRobot before motion. Arm targets use degrees;
the gripper uses LeRobot's 0–100 scale. Do not reuse older normalized joint targets
without converting or recording them again.

## Teach the block and glass positions

Motor calibration establishes encoder coordinates. It does not locate the block,
glass, table, or a collision-free path. This version uses your taught path; it does
not yet include inverse kinematics or arbitrary object relocation.

Position the arm using your established LeRobot/manual setup, then release the serial
port before recording. `record-pose` reads the already positioned arm; it does not
change torque or send position goals. Maintain/support the pose as appropriate to
your setup. It cannot share the serial port with a running teleoperation process.

Record these in order. Repeat a target to append intermediate waypoints:

| Target | Pose to record |
| --- | --- |
| `start` | Clear starting pose with open gripper |
| `approach` | Open gripper above the marked block |
| `grasp` | Descend beside the block with open gripper; add intermediate poses as needed |
| `grasp --gripper-only` | At that same arm pose, the measured gripper opening that holds the block |
| `lift` | Block held, raised clear of table and glass rim |
| `place` | Clearance path ending centered above the opening, at the chosen low release height |
| `release` | Gripper opened while arm stays at the final place pose |
| `retract` | Open gripper withdrawn to clear both the glass and camera view |

For example, **after physically positioning the arm** at each appropriate pose:

```sh
.venv/bin/python -m jev_arm record-pose --target start
.venv/bin/python -m jev_arm record-pose --target approach
.venv/bin/python -m jev_arm record-pose --target grasp
.venv/bin/python -m jev_arm record-pose --target grasp --gripper-only
.venv/bin/python -m jev_arm record-pose --target lift
.venv/bin/python -m jev_arm record-pose --target place
.venv/bin/python -m jev_arm record-pose --target release
.venv/bin/python -m jev_arm record-pose --target retract
```

These commands update `config.local.json`. Ordinary waypoints retain the previously
taught gripper opening. `--gripper-only` and `release` retain the preceding arm pose
and record only the measured gripper opening. This prevents encoder noise from
introducing unwanted changes to joints that should remain stationary. To reteach an
existing path, clear its waypoint list in the config before recording again.

Configure `joint_limits` for all six names: `shoulder_pan`, `shoulder_lift`,
`elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`. Each needs `minimum`, `maximum`,
`speed_per_s`, `tracking_tolerance`, and `arrival_tolerance`. Determine these bounds
for your actual setup; no made-up hardware bounds are shipped. Arm values are degrees
and degrees/second; gripper values are 0–100 units and units/second. Arrival tolerance
must be no larger than tracking tolerance. Leave room for normal grasp contact without
using a fully closed target that stalls against the block.

Each waypoint has a `duration_s` (default 2). The controller stretches it if necessary
to honor joint speed limits. Each entire primitive must finish within
`command_timeout_s` (default 20), including settling. Dense taught waypoints can help
describe clearance, but joint interpolation does not guarantee a straight Cartesian
path or avoid collisions. Check the full taught path on your setup before a complete run.

## Run the physical controller

Once setup is complete and the arm is at the recorded start pose:

```sh
.venv/bin/python -m jev_arm check
.venv/bin/python -m jev_arm run --enable-motion
```

`check` validates config/files and key presence without connecting or calling APIs.
The run verifies both cameras first, checks calibration and position mode, checks the
start pose, seeds motor goals with measured positions, and then enables torque.
It preserves the motor configuration/PID/current limits from your LeRobot setup.
It will not automatically home the arm. Keep the overhead camera, pickup mark, and
glass fixed relative to the taught layout.

The sequence is `approach → grasp → lift → place → release → retract → verify`.
Jev may request another observation or hold, but cannot skip stages. Grasp closure
and glass release are separate stationary gripper motions. Release requires positive
visual evidence of alignment over the glass. After retracting, completion requires
two distinct observations of the block inside the upright glass and out of the gripper.
An occluded/transparent glass may prevent confirmation; disappearance alone is insufficient.

## Timing, stopping, and current limits

- The main thread owns all robot reads/writes. Camera workers and the model worker do
  not access the motor bus. Models are called between primitives; the arm holds while
  those requests run. Bounded primitives run without waiting on models.
- 30 Hz is a scheduling target, not a hard real-time guarantee. OS scheduling and serial
  calls can block. The code checks excessive tick gaps and read latency before issuing
  further motion, but cannot interrupt a hung driver call or protect against process death.
- Press Ctrl+C or create the configured `STOP` file to request a local hold. From another
  terminal in the project directory: `touch STOP`. Remove it only after inspecting the
  cause and resetting the arm for a new run. Faults are latched and do not auto-resume.
- On exit, the program attempts to hold measured positions and leaves torque enabled
  to avoid dropping a loaded arm. **Software hold is not a hardware emergency stop.**
  Communication failure may prevent it. Use the arm's physical power/stop provision
  when necessary, accounting for the arm dropping if power is removed.
- Both USB streams must remain fresh during a primitive. A camera heartbeat proves
  frame delivery, not that the scene is safe. Semantic model hazards are evaluated
  between primitives; this version has no local human detector or collision sensor.
- Decisions carry local capture times and controller revisions. Older-stage, duplicate,
  or expired observations cannot start motion. Default maximum observation age is
  8 seconds; HTTP timeout is 20 seconds with no automatic model retries. Slow inference
  produces a hold and a fresh observation, never a stale action.
- Model thresholds are prototype settings for evaluation on this task, not safety
  certifications: Choice confidence ≥ 0.7, selected probability ≥ 0.8, unsafe < 0.1,
  and done ≥ 0.9. Jev's confidence summarizes its distribution; it is not a separate
  learned estimate of robot safety. Measure false positives/negatives on your scene logs.
- The session stops after 180 seconds by default. Logs contain decisions, observations,
  measured stage state, and model latency; they do not contain image bytes or API keys.

## Simulation

`so101_sim/` is a MuJoCo simulation of this setup (SO-101, table, block, cup, overhead and wrist cameras) with
the same joint names and controller units, real-time physics while models are called, and evaluator-only ground
truth. See [so101_sim/README.md](so101_sim/README.md) for the interface, the joint mapping, and how to run the
pipeline against it. `config.sim.json` holds IK-planned waypoints for the simulated layout.

## Implementation and sources

| File | Responsibility |
| --- | --- |
| `jev_arm/clients.py` | Astra Responses API and TypeSafe `POST /v1/systemone` |
| `jev_arm/models.py` | Scene, typed answers, configuration and waypoint validation |
| `jev_arm/cameras.py` | Independent overhead/wrist capture and freshness checks |
| `jev_arm/controller.py` | Ordered skills, gates, interpolation and local limits |
| `jev_arm/robot.py` | LeRobot bus adapter and synthetic robot |
| `jev_arm/runtime.py` | Background decisions, main control loop and shutdown |
| `jev_arm/cli.py` | Demo, observation, capture, recording and run commands |

Verified against [TypeSafe's quickstart](https://docs.typesafe.ai/introduction/quickstart),
[Choice](https://docs.typesafe.ai/primitives/choice),
[Noul](https://docs.typesafe.ai/primitives/noul), and
[confidence documentation](https://docs.typesafe.ai/confidence);
the [Astra model page](https://developers.openai.com/api/docs/models/gpt-6-astra) and
[Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs);
and [LeRobot's robot API](https://huggingface.co/docs/lerobot/main/api/robots) plus the
[SO follower source](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/so_follower.py).

Future work: camera-to-robot geometry and IK for changing object positions; measured
task-specific decision thresholds; a trained SmolVLA controller behind the same skill
boundary. Those capabilities are not part of this first implementation.
