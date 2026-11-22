"""Reuse fixed fitted teachers for a bounded, train/validation-only speed calibration."""

from __future__ import annotations

import collections
import dataclasses
import datetime
import hashlib
import json
from pathlib import Path
import time

from filelock import FileLock
import numpy as np
from scipy.spatial.transform import Rotation

from openpi.datasets.libero import fields
from openpi.datasets.libero import replay_teacher
from openpi.datasets.libero.reader import ProcessedLiberoDataset
from openpi.datasets.libero.reader import atomic_json
from openpi.fields import primitive
from openpi.fields import seds


def reachability_lower_bound(position, time, max_speed):
    """Necessary speed-only bound, independent of the field or robot controller.

    A path starting at p0 with ||v|| <= vmax cannot travel farther than vmax*t.
    Violation proves this synchronous tracking criterion is impossible; passing
    says nothing about obstacle, field, or contact feasibility.
    """
    distance = np.linalg.norm(position - position[0], axis=1)
    lower = np.maximum(0.0, distance - max_speed * (time - time[0]))
    return float(np.sqrt(np.average(lower**2, weights=seds.time_weights(time))))


def recheck_episode(original, data, control, selection=fields.DEFAULT_SELECTION, *, representation="fitted"):
    started = time.monotonic()
    records = []
    rotations = Rotation.from_quat(data["trajectory"]["quaternion_xyzw"]).as_matrix() @ data["body_to_control_rotation"]
    intervals = {i["command_interval_id"]: i for i in data["command_intervals"]}
    for source in original["records"]:
        record = {**source}
        candidate = source["candidate"]
        start, end = candidate["start"], candidate["end"]
        record["speed_only_position_rms_lower_bound_m"] = reachability_lower_bound(
            data["trajectory"]["position"][start : end + 1], data["time"][start : end + 1], control.max_speed
        )
        if representation == "linear" and source.get("fit", {}).get("kind") != "analytic_hold":
            rate = source.get("fit", {}).get("linear_baseline_rate_per_second")
            if rate is not None:
                field = seds.PositionField(seds.linear_factors(rate), data["trajectory"]["position"][end])
                baseline = primitive.Primitive(field, rotations[end], candidate["command"])
                record["primitive"] = baseline.record()
                record["fit"] = {
                    "kind": "linear_baseline",
                    "rate_per_second": rate,
                    "parameter_supervision": False,
                    "source_fit_kind": source["fit"]["kind"],
                }
        elif representation not in {"fitted", "linear"}:
            raise ValueError("Unknown teacher representation")
        if record.get("primitive") is not None:
            current = primitive.Primitive.from_record(record["primitive"])
            nominal = primitive.nominal_replay(
                current,
                data["trajectory"]["position"][start : end + 1],
                rotations[start : end + 1],
                data["time"][start : end + 1],
                control,
            )
            record.update(nominal=nominal, accepted=nominal["passed"], reasons=nominal["reasons"], cost=None)
            if nominal["passed"]:
                interval = intervals[candidate["command_interval_id"]]
                fraction = (data["time"][end] - data["time"][start]) / (
                    data["time"][interval["end"]] - data["time"][interval["start"]]
                )
                record["cost"] = float(
                    fraction
                    * (
                        (nominal["position_rms_m"] / control.position_rms_tolerance) ** 2
                        + (nominal["rotation_rms_rad"] / control.rotation_rms_tolerance) ** 2
                    )
                    / 2
                    + selection.field_penalty
                )
        records.append(record)
    sequences = fields.k_best_sequences(records, len(data["actions"]), selection)
    result = {
        **original,
        "records": records,
        "sequences": sequences,
        "control": dataclasses.asdict(control),
        "training_ready": False,
        "continuous_execution_status": "not_run",
        "teacher_representation": representation,
    }
    result["summary"] = {
        **original["summary"],
        "accepted_candidates": sum(r["accepted"] for r in records),
        "wall_seconds_this_invocation": time.monotonic() - started,
        "complete_offline_sequence": bool(sequences),
        "speed_only_impossible_candidates": sum(
            r["speed_only_position_rms_lower_bound_m"] > control.position_rms_tolerance for r in records
        ),
        "failure_counts": dict(collections.Counter(reason for r in records for reason in r.get("reasons", []))),
    }
    return result


def calibrate_fields(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / ".calibration.lock"), timeout=0):
        if (output / "run.json").exists():
            raise ValueError("Calibration already exists; use a new output directory")
        parent = json.loads((args.fields_dir / "run.json").read_text())
        if parent["status"] != "complete" or parent["config"]["split"] not in ("train", "validation"):
            raise ValueError("Calibration requires a completed train or validation field pool")
        dataset = ProcessedLiberoDataset(args.dataset)
        manifest_hash = hashlib.sha256((args.dataset / "manifest.json").read_bytes()).hexdigest()
        if manifest_hash != parent["config"]["source_manifest_sha256"]:
            raise ValueError("Calibration source manifest differs from the fitted pool")
        originals = [json.loads((args.fields_dir / entry["file"]).read_text()) for entry in parent["episodes"]]
        data = [replay_teacher.field_input(dataset, o["source_episode_index"], args.fields_dir) for o in originals]
        for original, episode in zip(originals, data, strict=True):
            if original["source_episode"]["signature"] != episode["entry"]["signature"]:
                raise ValueError("Calibration trajectory does not match the fitted source")
        controls = [
            dataclasses.replace(primitive.ControlConfig(**parent["config"]["control"]), max_speed=speed)
            for speed in (0.1, 0.2, 0.3)
        ]
        config = {
            **parent["config"],
            "calibration": "three fixed speed candidates; identical fitted fields; exploration",
            "parent_run_sha256": hashlib.sha256((args.fields_dir / "run.json").read_bytes()).hexdigest(),
            "calibration_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "calibration_controls": [dataclasses.asdict(c) for c in controls],
            "runtime_sha256": fields.implementation_hashes(),
        }
        fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        run = {
            "fingerprint": fingerprint,
            "config": config,
            "status": "running",
            "started_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "training_ready": False,
        }
        replay_teacher.copy_artifact(args.fields_dir, output, parent["config"].get("teacher_source"))
        fields.save_source_snapshot(output, extra=(Path(__file__),))
        atomic_json(output / "run.json", run)
        started, curve, results = time.monotonic(), [], []
        try:
            selection = fields.SelectionConfig(**parent["config"]["selection"])
            for index, control in enumerate(controls):
                current = [recheck_episode(o, d, control, selection) for o, d in zip(originals, data, strict=True)]
                row = {
                    "config_index": index,
                    "control": dataclasses.asdict(control),
                    "episodes": [r["summary"] for r in current],
                    "complete_offline_sequences": sum(bool(r["sequences"]) for r in current),
                    "accepted_candidates": sum(r["summary"]["accepted_candidates"] for r in current),
                    "selected_fields": sum(len(r["sequences"][0]["record_indices"]) for r in current if r["sequences"]),
                }
                curve.append(row)
                results.append(current)
                atomic_json(output / "calibration_curve.json", curve)
                print(json.dumps(row, ensure_ascii=False), flush=True)
            # Predeclared lexicographic rule: coverage, accepted local spans, fewer fields, lower speed.
            best = max(
                range(len(curve)),
                key=lambda i: (
                    curve[i]["complete_offline_sequences"],
                    curve[i]["accepted_candidates"],
                    -curve[i]["selected_fields"],
                    -controls[i].max_speed,
                ),
            )
            run["episodes"] = []
            for result in results[best]:
                index = result["source_episode_index"]
                name = f"episode_{index:05d}.json"
                result.update(
                    fingerprint=fingerprint, calibration_config_index=best, parent_fingerprint=parent["fingerprint"]
                )
                atomic_json(output / name, result)
                run["episodes"].append({"episode_index": index, "file": name, **result["summary"]})
            run.update(
                status="complete",
                selected_config_index=best,
                complete_offline_sequences=curve[best]["complete_offline_sequences"],
                calibration_configs_evaluated=len(curve),
                selected_control=dataclasses.asdict(controls[best]),
            )
        except BaseException as error:
            run.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
            raise
        finally:
            run["seconds"] = time.monotonic() - started
            atomic_json(output / "run.json", run)
    return 0


def rescore_fields(args):
    """Apply one declared representation/reference alternative to a frozen candidate pool."""
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / ".rescore.lock"), timeout=0):
        if (output / "run.json").exists():
            raise ValueError("Rescoring already exists; use a new output directory")
        parent_file = args.fields_dir / "run.json"
        parent = json.loads(parent_file.read_text())
        if parent["status"] != "complete":
            raise ValueError("Rescoring requires a complete offline field pool")
        dataset = ProcessedLiberoDataset(args.dataset)
        manifest_hash = hashlib.sha256((args.dataset / "manifest.json").read_bytes()).hexdigest()
        if manifest_hash != parent["config"]["source_manifest_sha256"]:
            raise ValueError("Rescoring source manifest differs from the fitted pool")
        control = primitive.ControlConfig(**parent.get("selected_control", parent["config"]["control"]))
        control = dataclasses.replace(
            control, reference_seconds=0.05 if args.reference_mode == "delta" else 2 / np.sqrt(150)
        )
        if args.arrival_position_tolerance is not None:
            control = dataclasses.replace(control, position_tolerance=args.arrival_position_tolerance)
        config = {
            **parent["config"],
            "control": dataclasses.asdict(control),
            "parent_run_sha256": hashlib.sha256(parent_file.read_bytes()).hexdigest(),
            "representation": args.representation,
            "reference_mode": args.reference_mode,
            "rescore_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "runtime_sha256": fields.implementation_hashes(),
        }
        fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        run = {
            "status": "running",
            "fingerprint": fingerprint,
            "config": config,
            "episodes": [],
            "training_ready": False,
            "started_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        }
        replay_teacher.copy_artifact(args.fields_dir, output, parent["config"].get("teacher_source"))
        fields.save_source_snapshot(output, extra=(Path(__file__),))
        atomic_json(output / "run.json", run)
        started = time.monotonic()
        try:
            selection = fields.SelectionConfig(**parent["config"]["selection"])
            for entry in parent["episodes"]:
                original = json.loads((args.fields_dir / entry["file"]).read_text())
                index = original["source_episode_index"]
                data = replay_teacher.field_input(dataset, index, args.fields_dir)
                if data["entry"]["signature"] != original["source_episode"]["signature"]:
                    raise ValueError("Rescoring source episode mismatch")
                result = recheck_episode(original, data, control, selection, representation=args.representation)
                result.update(fingerprint=fingerprint, parent_fingerprint=parent["fingerprint"])
                atomic_json(output / entry["file"], result)
                row = {"episode_index": index, "file": entry["file"], **result["summary"]}
                run["episodes"].append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
            run.update(
                status="complete",
                complete_offline_sequences=sum(r["complete_offline_sequence"] for r in run["episodes"]),
            )
        except BaseException as error:
            run.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
            raise
        finally:
            run["seconds"] = time.monotonic() - started
            atomic_json(output / "run.json", run)
    return 0
