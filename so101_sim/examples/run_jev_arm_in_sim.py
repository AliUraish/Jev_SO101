"""Run the existing jev_arm pipeline (Astra + Jev + controller) against the simulator.

Needs OPENAI_API_KEY and TYPESAFE_API_KEY in the environment or .env, plus a config produced by
scripted_pick_place.py --export-config. Simulation adapters are passed explicitly to the
shared runtime; simulation poses remain marked simulation_only and no calibration shim is created.

    .venv/bin/python -m so101_sim.examples.scripted_pick_place --fast --export-config config.sim.json --reference captures/sim/overhead_reference.jpg
    .venv/bin/python -m so101_sim.examples.run_jev_arm_in_sim --config config.sim.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv

from jev_arm import runtime
from jev_arm.clients import require_keys
from jev_arm.models import load_config
from so101_sim import SO101Sim, SceneConfig
from so101_sim.jev_arm_adapter import SimArm, SimCameras, encode_jpeg
from so101_sim.live_view import LiveView


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("config.sim.json"))
    ap.add_argument("--log", type=Path, default=Path("logs/sim_run.jsonl"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--headless", action="store_true", help="run without the live browser viewer")
    args = ap.parse_args()
    load_dotenv(Path(".env"), override=False)
    require_keys()

    config = load_config(args.config)

    sim = SO101Sim(SceneConfig(), realtime=True)
    sim.reset(seed=args.seed)
    error = "Simulation did not complete"
    started = time.monotonic()
    viewer = None
    try:
        if not args.headless:
            viewer = LiveView(sim)
            viewer.start()
        runtime.run_simulation(config, args.log, SimArm(sim),
                               SimCameras(sim, config.camera_max_age_s, config.camera_max_skew_s),
                               status_queue=viewer.updates if viewer else None,
                               stop_event=viewer.stop_requested if viewer else None)
        error = None
    except KeyboardInterrupt:
        error = "Interrupted by user"
        if viewer:
            viewer.closed.set()
        print(f"Simulation stopped: {error}", flush=True)
    except (RuntimeError, ValueError) as exc:
        error = str(exc)
        print(f"Simulation stopped: {error}", flush=True)
    finally:
        gt = sim.get_ground_truth()
        print("ground truth:", json.dumps({"flags": gt["flags"], "cup_tilt_deg": gt["cup"]["tilt_deg"], "block": gt["block"]["position"]}, indent=2))
        try:
            args.log.parent.mkdir(parents=True, exist_ok=True)
            summary = {
                "model_completed": error is None,
                "evaluator_success": bool(gt["flags"]["success"]),
                "error": error,
                "wall_seconds": round(time.monotonic() - started, 2),
                "flags": gt["flags"],
            }
            args.log.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            for name, frame in sim.render_now().items():
                args.log.with_name(args.log.stem + f".{name}.jpg").write_bytes(encode_jpeg(frame.rgb))
        finally:
            try:
                if viewer:
                    viewer.finish(error is None and gt["flags"]["success"], error)
            finally:
                sim.close()
    if viewer:
        try:
            viewer.wait_closed()
        finally:
            viewer.close()
    if error:
        raise SystemExit(1)
    if not gt["flags"]["success"]:
        print("Verification failed: model completion disagrees with simulator ground truth.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
