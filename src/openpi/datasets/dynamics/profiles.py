"""Alternate control-profile fitting for persisted dynamics collections.

The default collection is a source-calibrated playback result.  This module
creates a separate profile directory when a fixed production controller must
be checked, so the playback artifact remains reproducible and available.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
import dataclasses
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from openpi.datasets.libero.reader import atomic_json
from openpi.fields import sequence

SCHEMA = "vlds.dynamics.episode.v1"
PROFILE_RUN_SCHEMA = "vlds.dynamics.profile_run.v1"
PROFILE_INDEX_SCHEMA = "vlds.dynamics.profile_index.v1"
ROOT = Path(__file__).resolve().parents[4]


@dataclasses.dataclass(frozen=True)
class DynamicsProfile:
    name: str
    label: str
    maximum_speed_native_per_second: float | None
    maximum_angular_speed_rad_per_second: float | None
    rotation_gain_per_second: float | None
    fit_margin: float = 0.9
    maximum_time_scale: int = 1

    def as_dict(self):
        return dataclasses.asdict(self)


PROFILES = {
    "production_libero": DynamicsProfile(
        name="production_libero",
        label="LIBERO 原生产控制约束",
        maximum_speed_native_per_second=0.3,
        maximum_angular_speed_rad_per_second=0.5,
        rotation_gain_per_second=2.0,
    ),
    "production_retimed": DynamicsProfile(
        name="production_retimed",
        label="LIBERO 生产约束/放慢时间",
        maximum_speed_native_per_second=0.3,
        maximum_angular_speed_rad_per_second=0.5,
        rotation_gain_per_second=2.0,
        maximum_time_scale=16,
    ),
}


def profile(name):
    try:
        return PROFILES[name]
    except KeyError as error:
        raise ValueError(f"Unknown dynamics profile: {name}") from error


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _profile_fingerprint(base_fingerprint, selected, implementation):
    payload = {
        "schema": PROFILE_RUN_SCHEMA,
        "base_fingerprint": base_fingerprint,
        "profile": selected.as_dict(),
        "implementation_sha256": implementation,
        "model": sequence.VERSION,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _fit(base_result, archive, selected):
    position = archive["position"]
    time_axis = archive["time"]
    tolerance = archive["tolerance"]
    rotations = archive.get("rotation", None)
    if selected.maximum_speed_native_per_second is None:
        maximum_speed = None
    else:
        maximum_speed = selected.maximum_speed_native_per_second / float(np.min(tolerance))
    margins = (selected.fit_margin, 0.5, 0.1, 0.0) if selected.maximum_time_scale == 1 else (selected.fit_margin,)
    scales = [2**power for power in range(selected.maximum_time_scale.bit_length())]
    trials = []
    for scale, margin in ((scale, margin) for scale in scales for margin in margins):
        records, summary, nominal, limits = sequence.fit_sequence(
            position,
            time_axis[0] + (time_axis - time_axis[0]) * scale,
            base_result["source_intervals"],
            tolerance,
            rotations=rotations,
            maximum_speed=maximum_speed,
            maximum_angular_speed=selected.maximum_angular_speed_rad_per_second,
            rotation_gain=selected.rotation_gain_per_second,
            integration_substeps=scale,
            config=sequence.SequenceConfig(
                fit_margin=margin,
                arrival_speed_scaled_per_second=2.0 if selected.maximum_time_scale > 1 else None,
                arrival_angular_speed_rad_per_second=0.1 if selected.maximum_time_scale > 1 else None,
                arrival_dwell_seconds=0.1 if selected.maximum_time_scale > 1 else 0,
            ),
        )
        trials.append(
            {"time_scale": scale, "fit_margin": margin, "passed": summary["passed"], "segments": len(records)}
        )
        if summary["passed"]:
            break
    reasons = Counter(reason for record in records for reason in record["report"]["reasons"])
    summary.update(
        {
            "profile": selected.name,
            "profile_label": selected.label,
            "fit_margin": margin,
            "profile_passed": bool(summary["passed"]),
            "failure_reasons": dict(sorted(reasons.items())),
            "maximum_speed_native_per_second": selected.maximum_speed_native_per_second,
            "maximum_angular_speed_rad_per_second": selected.maximum_angular_speed_rad_per_second,
            "rotation_gain_per_second": selected.rotation_gain_per_second,
            "time_scale": scale,
            "source_seconds": float(time_axis[-1] - time_axis[0]),
            "nominal_seconds": float((time_axis[-1] - time_axis[0]) * scale),
            "time_scale_trials": trials,
        }
    )
    limits.update(time_scale=scale, source_sample_dt=float(np.median(np.diff(time_axis))))
    return records, summary, nominal, limits


def _write_trace(path, arrays):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _build_one(base_directory, output_directory, index, selected, fingerprint):
    base_directory, output_directory = Path(base_directory), Path(output_directory)
    base_path = base_directory / f"episode_{index:05d}.json"
    base_result = json.loads(base_path.read_text())
    base_trace = base_directory / base_result["trace_file"]
    if _digest(base_trace) != base_result["trace_sha256"]:
        raise ValueError("Base trace hash mismatch; rebuild the base collection")
    output_path = output_directory / base_path.name
    trace_path = output_path.with_suffix(".npz")
    if output_path.exists() and trace_path.exists():
        previous = json.loads(output_path.read_text())
        if (
            previous.get("fingerprint") == fingerprint
            and previous.get("base_episode_sha256") == _digest(base_path)
            and previous.get("trace_sha256") == _digest(trace_path)
        ):
            return {
                "index": index,
                "file": output_path.name,
                "source_key": base_result["source_key"],
                **previous["summary"],
                "resumed": True,
            }
    with np.load(base_trace, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
        records, summary, nominal, limits = _fit(base_result, arrays, selected)
        scale = summary["time_scale"]
        saved = {
            "position": arrays["position"],
            "time": arrays["time"][0] + (arrays["time"] - arrays["time"][0]) * scale,
            "source_time": arrays["time"],
            "nominal": nominal,
            "tolerance": arrays["tolerance"],
        }
        if "rotation" in arrays:
            saved["rotation"] = arrays["rotation"]
            current = arrays["rotation"][0]
            orientations = [current]
            for record in records:
                path = sequence.integrate_rotation(
                    current,
                    np.asarray(record["goal_rotation"]),
                    (record["end"] - record["start"]) * scale,
                    limits["dt"],
                    limits["maximum_angular_speed_rad_per_second"],
                    limits["rotation_gain_per_second"],
                )
                orientations.extend(path[scale::scale])
                current = path[-1]
            saved["nominal_rotation"] = np.asarray(orientations)
    _write_trace(trace_path, saved)
    result = {
        "schema": SCHEMA,
        "model": sequence.VERSION,
        "profile": selected.name,
        "profile_label": selected.label,
        "fingerprint": fingerprint,
        "base_fingerprint": base_result["fingerprint"],
        "base_episode_sha256": _digest(base_path),
        "source_episode_index": index,
        "source_key": base_result["source_key"],
        "source_sha256": base_result["source_sha256"],
        "source_entry": base_result["source_entry"],
        "coordinates": base_result["coordinates"],
        "tolerance": base_result["tolerance"],
        "limits": limits,
        "source_intervals": base_result["source_intervals"],
        "records": records,
        "summary": summary,
        "trace_file": trace_path.name,
        "trace_sha256": _digest(trace_path),
        "scope": "offline nominal source-time playback under a fixed controller profile; physical execution remains unvalidated",
    }
    atomic_json(output_path, result)
    return {
        "index": index,
        "file": output_path.name,
        "source_key": base_result["source_key"],
        **summary,
        "resumed": False,
    }


def build_profile(base_directory, name="production_libero", *, workers=8, indices=None, output=None):
    base_directory = Path(base_directory).resolve()
    selected = profile(name)
    base_run = json.loads((base_directory / "run.json").read_text())
    base_index = json.loads((base_directory / "index.json").read_text())
    if base_run["status"] != "complete" or not base_run["summary"]["full_collection"]:
        raise ValueError("The base dynamics collection must be complete before profile fitting")
    if base_run["config"]["kind"] != "processed_libero":
        raise ValueError("Production profiles are defined only for processed LIBERO")
    chosen = list(range(len(base_index["episodes"]))) if indices is None else list(indices)
    if not chosen or len(chosen) != len(set(chosen)) or any(i < 0 or i >= len(base_index["episodes"]) for i in chosen):
        raise ValueError("Select distinct existing source episodes")
    output_directory = base_directory / "profiles" / name if output is None else Path(output).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    implementation = {
        "src/openpi/datasets/dynamics/profiles.py": _digest(Path(__file__)),
        "src/openpi/fields/sequence.py": _digest(Path(sequence.__file__)),
    }
    fingerprint = _profile_fingerprint(base_run["fingerprint"], selected, implementation)
    run_path = output_directory / "run.json"
    config = {
        "schema": PROFILE_RUN_SCHEMA,
        "base_directory": os.path.relpath(base_directory, output_directory),
        "base_fingerprint": base_run["fingerprint"],
        "base_source_manifest_sha256": base_run["config"]["source_manifest_sha256"],
        "source_manifest_sha256": base_run["config"]["source_manifest_sha256"],
        "kind": base_run["config"]["kind"],
        "source_relative": os.path.relpath((base_directory / base_run["source_relative"]).resolve(), output_directory),
        "source_episodes": base_run["config"]["source_episodes"],
        "indices": chosen,
        "profile": selected.as_dict(),
        "model": sequence.VERSION,
        "implementation_sha256": implementation,
    }
    if run_path.exists():
        previous_run = json.loads(run_path.read_text())
        if previous_run.get("fingerprint") not in (None, fingerprint):
            raise ValueError("Profile configuration differs; choose a new output directory")
    for relative in implementation:
        snapshot = output_directory / "source_snapshot" / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes((ROOT / relative).read_bytes())
    started = time.monotonic()
    run = {
        "fingerprint": fingerprint,
        "config": config,
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "command": sys.orig_argv,
        "started_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "status": "running",
    }
    atomic_json(run_path, run)
    results, errors = [], []
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending = {
                pool.submit(_build_one, str(base_directory), str(output_directory), index, selected, fingerprint): index
                for index in chosen
            }
            for future in as_completed(pending):
                index = pending[future]
                try:
                    value = future.result()
                    results.append(value)
                except Exception as error:  # keep all failed episodes visible in the report
                    errors.append({"index": index, "error": repr(error)})
                completed = len(results) + len(errors)
                if completed % 100 == 0 or completed == len(chosen):
                    print(
                        json.dumps(
                            {
                                "profile": name,
                                "finished": completed,
                                "errors": len(errors),
                                "passed": sum(r.get("profile_passed", False) for r in results),
                                "seconds": round(time.monotonic() - started, 1),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
        results.sort(key=lambda item: item["index"])
        atomic_json(
            output_directory / "index.json",
            {
                "schema": PROFILE_INDEX_SCHEMA,
                "fingerprint": fingerprint,
                "base_fingerprint": base_run["fingerprint"],
                "profile": selected.as_dict(),
                "episodes": [
                    {
                        "index": item["index"],
                        "file": item["file"],
                        "source_key": item.get("source_key"),
                        "passed": item["profile_passed"],
                    }
                    for item in results
                ],
            },
        )
        reason_counts = Counter()
        for item in results:
            reason_counts.update(item.get("failure_reasons", {}))
        summary = {
            "profile": selected.name,
            "profile_label": selected.label,
            "episodes": len(results),
            "expected_episodes": len(base_index["episodes"]),
            "selected_episodes": len(chosen),
            "passed_episodes": sum(item["profile_passed"] for item in results),
            "segments": sum(item["segments"] for item in results),
            "accepted_segments": sum(item["accepted_segments"] for item in results),
            "transitions_covered": sum(item["actions_covered"] for item in results),
            "single_transition_segments": sum(item["single_transition_segments"] for item in results),
            "failure_reasons": dict(sorted(reason_counts.items())),
            "full_collection": indices is None and not errors and len(results) == len(base_index["episodes"]),
            "errors": errors,
            "seconds": time.monotonic() - started,
            "maximum_scaled_error": max((item["maximum_scaled_error"] for item in results), default=None),
            "maximum_speed_native_per_second": selected.maximum_speed_native_per_second,
            "maximum_angular_speed_rad_per_second": selected.maximum_angular_speed_rad_per_second,
            "rotation_gain_per_second": selected.rotation_gain_per_second,
            "physical_execution_validated": False,
            "training_ready": False,
            "time_scale_episode_counts": dict(sorted(Counter(item["time_scale"] for item in results).items())),
            "all_nominal_passed": bool(results) and all(item["profile_passed"] for item in results),
        }
        atomic_json(output_directory / "summary.json", summary)
        run.update(status="complete" if not errors else "failed", summary=summary)
    except BaseException as error:
        run.update(status="interrupted", error=repr(error))
        raise
    finally:
        run["seconds"] = time.monotonic() - started
        atomic_json(run_path, run)
    return run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path, help="complete dynamics collection directory")
    parser.add_argument("--profile", default="production_libero", choices=sorted(PROFILES))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--episodes", nargs="+", type=int)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("workers must be positive")
    result = build_profile(args.base, args.profile, workers=args.workers, indices=args.episodes, output=args.output)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
