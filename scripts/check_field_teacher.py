"""Collect auditable field-teacher diagnostics without promoting them to training data."""

from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

from openpi.datasets.libero import fields
from openpi.datasets.libero.operation_audit import EvidenceConfig
from openpi.datasets.libero.operation_audit import summarize_execution
from openpi.datasets.libero.reader import atomic_json
from openpi.fields import primitive


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(root, test_report=None):
    offline, executions, errors, running = [], [], [], []
    for path in sorted(root.glob("*/run.json")):
        run = json.loads(path.read_text())
        if run["status"] == "running":
            running.append(str(path.relative_to(root)))
        teacher = run["config"].get("teacher_source")
        if teacher and digest(path.parent / teacher["file"]) != teacher["sha256"]:
            errors.append(f"Derived teacher archive mismatch: {path.parent.name}")
        row = {
            "directory": path.parent.name,
            "status": run["status"],
            "run_sha256": digest(path),
            "fingerprint": run["fingerprint"],
            "episodes": [],
            "config": {
                key: run["config"].get(key)
                for key in ("field", "control", "selection", "representation", "reference_mode", "teacher_source")
            },
        }
        for episode in run.get("episodes", []):
            artifact = path.parent / episode["file"]
            data = json.loads(artifact.read_text())
            if data["training_ready"]:
                errors.append(f"Unverified field pool marked ready: {artifact.relative_to(root)}")
            if data["source_episode"]["split"] != "train":
                errors.append(f"Pilot touched a nontraining episode: {artifact.relative_to(root)}")
            for record in data["records"]:
                if record.get("primitive") is not None:
                    try:
                        primitive.Primitive.from_record(record["primitive"])
                    except ValueError as error:
                        errors.append(f"{artifact.relative_to(root)}: {error}")
                if record["accepted"] and (
                    not record.get("nominal", {}).get("passed") or record.get("primitive") is None
                ):
                    errors.append(f"Accepted edge lacks verified nominal artifact: {artifact.relative_to(root)}")
            for sequence in data["sequences"]:
                end = 0
                for index in sequence["record_indices"]:
                    record = data["records"][index]
                    if not record["accepted"] or record["candidate"]["start"] != end:
                        errors.append(f"Incomplete selected sequence: {artifact.relative_to(root)}")
                    end = record["candidate"]["end"]
                if end != data["source_episode"]["actions"]:
                    errors.append(f"Selected sequence drops source actions: {artifact.relative_to(root)}")
            kinds = collections.Counter()
            if data["sequences"]:
                kinds.update(data["records"][i]["fit"]["kind"] for i in data["sequences"][0]["record_indices"])
            row["episodes"].append(
                {
                    "episode_index": data["source_episode_index"],
                    "task": data["source_episode"]["task"],
                    "artifact_sha256": digest(artifact),
                    "summary": data["summary"],
                    "best_sequence_fields": len(data["sequences"][0]["record_indices"]) if data["sequences"] else None,
                    "best_sequence_teacher_kinds": dict(kinds),
                }
            )
        offline.append(row)
    for path in sorted(root.glob("*/result.json")):
        result = json.loads(path.read_text())
        if result["status"] == "running":
            running.append(str(path.relative_to(root)))
        worker = json.loads((path.parent / "worker.json").read_text())
        if result.get("training_ready"):
            errors.append(f"Diagnostic execution marked ready: {path.relative_to(root)}")
        request = json.loads((path.parent / "request.json").read_text())
        trials = []
        for trial in result.get("sequence_results", []):
            completed = trial["status"] == "geometric_sequence_complete"
            trials.append(
                {
                    "trial": trial["trial"],
                    "task_predicate": trial["task_success"],
                    "geometric_sequence_complete": completed,
                    "control_steps": trial["control_steps"],
                    "completed_fields": trial["completed_fields"],
                    "status": trial["status"],
                    "intermediate_resets": trial["intermediate_resets"],
                    "trajectory_sha256": digest(path.parent / trial["trajectory_file"]),
                    "final_goal_predicates": trial.get("final_goal_predicates"),
                }
            )
            if trial["intermediate_resets"] != 0:
                errors.append(f"Field handoff reset: {path.relative_to(root)}")
        executions.append(
            {
                "directory": path.parent.name,
                "status": result["status"],
                "worker_status": worker["status"],
                "source_episode_index": result.get("source_episode_index"),
                "mode": result.get("mode"),
                "environment": result.get("environment"),
                "max_steps": request["max_steps"],
                "source_sha256": result.get("source_sha256"),
                "error": result.get("error"),
                "wall_seconds": result.get("wall_seconds"),
                "gripper_feedback_required": request.get("gripper_feedback", False),
                "result_sha256": digest(path),
                "request_sha256": digest(path.parent / "request.json"),
                "trials": trials,
            }
        )
    operation_reports = []
    for path in sorted(root.glob("operation_evidence*.json")):
        evidence = json.loads(path.read_text())
        if evidence.get("training_ready"):
            errors.append(f"Unverified operation evidence marked ready: {path.name}")
        config = EvidenceConfig(**evidence["thresholds"])
        for key in ("execution", "source_replay"):
            saved = evidence.get(key)
            if saved is not None:
                recomputed = summarize_execution(root / saved["directory"], config)
                if saved != recomputed:
                    errors.append(f"Stale operation evidence: {path.name} / {key}")
                if any(c["verified"] or c["observable"] for t in saved["trials"] for c in t["candidates"]):
                    errors.append(f"Unreviewed operation candidate promoted: {path.name}")
        operation_reports.append({"file": path.name, "sha256": digest(path), **evidence})
    checks = {
        "passed": not errors and not running,
        "errors": errors,
        "running_artifacts": running,
        "no_training_dataset_released": not any("ready" in error for error in errors),
        "scope": "archive factors/physical agreement, nominal acceptance flags, full action coverage, source splits and terminal artifact states",
    }
    report = {
        "schema": "vlds.validation.field_teacher.v1",
        "date_local": datetime.datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat(),
        "timezone": "Asia/Shanghai",
        "generated_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "checks": checks,
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=fields.ROOT, text=True).strip(),
        "implementation_sha256": fields.implementation_hashes(),
        "collector_sha256": digest(Path(__file__)),
        "offline": offline,
        "executions": executions,
        "operation_evidence": operation_reports,
        "interpretation": "Exploratory training episodes. Alternate paths on the same initial state are not independent trials. No learned VLDS, force perturbation, semantic observer or main training dataset is evaluated.",
    }
    if test_report:
        tree = ET.parse(test_report)
        suites = tree.findall(".//testsuite")
        report["tests"] = {
            key: sum(int(suite.get(key, "0")) for suite in suites) for key in ("tests", "failures", "errors", "skipped")
        }
        report["tests"]["report_sha256"] = digest(test_report)
        if report["tests"]["failures"] or report["tests"]["errors"]:
            errors.append("Recorded test report contains failures or errors")
            checks["passed"] = False
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-report", type=Path)
    args = parser.parse_args()
    report = collect(args.run_root, args.test_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "checks": report["checks"],
                "offline_runs": len(report["offline"]),
                "execution_runs": len(report["executions"]),
                "operation_reports": len(report["operation_evidence"]),
                "tests": report.get("tests"),
            },
            indent=2,
        )
    )
    return 0 if report["checks"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
