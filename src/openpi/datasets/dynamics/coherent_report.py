"""Summarize original observed lengths, fit coverage and unresolved long spans."""

from __future__ import annotations

import argparse
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

from openpi.datasets.libero.reader import atomic_json

DATASETS = ("libero", "lerobot_libero", "aloha", "pusht", "xarm")


def collection(directory):
    directory = Path(directory)
    lengths, seconds, accepted_lengths = [], [], []
    counts, reasons = Counter(), Counter()
    passed, episodes = 0, 0
    chain = hashlib.sha256()
    for path in sorted(directory.glob("episode_*.json")):
        result = json.loads(path.read_text())
        dt = result["limits"].get("sample_dt", result["limits"]["dt"])
        source_dt = dt / result["limits"].get("time_scale", 1)
        episodes += 1
        passed += int(result["summary"]["passed"])
        chain.update(path.name.encode() + hashlib.sha256(path.read_bytes()).digest())
        for record in result["records"]:
            n = record["end"] - record["start"]
            quality = record.get("quality", {})
            duration = quality.get("source_seconds", n * source_dt)
            eligible = n >= 10 and duration >= 0.5 - 1e-7
            lengths.append(n)
            seconds.append(duration)
            if record["accepted"]:
                accepted_lengths.append(n)
            counts.update(
                segments=1,
                observed_transitions=n,
                accepted_segments=int(record["accepted"]),
                length_eligible_segments=int(eligible),
                short_segments=int(not eligible),
                short_command_events=int(quality.get("short_command_event", False)),
                eligible_accepted_segments=int(eligible and record["accepted"]),
                eligible_accepted_transitions=n * int(eligible and record["accepted"]),
            )
            reasons.update(record["report"].get("reasons", []))
    if not lengths:
        return None
    run_path, audit_path = directory / "run.json", directory / "audit.json"
    run = json.loads(run_path.read_text()) if run_path.exists() else {}
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    return {
        "episodes": episodes,
        "passed_episodes": passed,
        "expected_episodes": run.get("config", {}).get("source_episodes"),
        "status": run.get("status", "unknown"),
        "full_collection": run.get("summary", {}).get("full_collection", False),
        "counts": dict(counts),
        "source_transitions_p10_p50_p90": np.percentile(lengths, [10, 50, 90]).tolist(),
        "source_seconds_p10_p50_p90": np.percentile(seconds, [10, 50, 90]).tolist(),
        "minimum_source_transitions": min(lengths),
        "minimum_source_seconds": min(seconds),
        "accepted_transition_fraction": sum(accepted_lengths) / sum(lengths),
        "eligible_accepted_transition_fraction": counts["eligible_accepted_transitions"] / sum(lengths),
        "rejected_segments_by_reason": dict(reasons),
        "fingerprint": run.get("fingerprint"),
        "episode_json_sha256_chain": chain.hexdigest(),
        "audit_integrity_passed": audit.get("integrity_passed"),
        "audit_full_coverage": audit.get("full_source_episode_coverage"),
        "audit_failures": audit.get("failures"),
        "training_ready": False,
        "parameter_supervision": False,
        "physical_execution_validated": False,
    }


def report(base, output):
    base, output = Path(base), Path(output)
    datasets = {}
    for kind in DATASETS:
        old = collection(base / kind)
        new = collection(base / kind / "profiles/coherent")
        datasets[kind] = {"previous_linear": old, "coherent": new}
    result = {
        "schema": "vlds.dynamics.coherent_comparison.v1",
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "datasets": datasets,
        "complete": all(
            d["coherent"]
            and d["coherent"]["full_collection"]
            and d["coherent"]["audit_integrity_passed"]
            and d["coherent"]["audit_full_coverage"]
            for d in datasets.values()
        ),
        "scope": "length and nominal fitting; no parameter identifiability or physical execution claim",
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "comparison.json", result)
    rows = [
        "# Observed length and closed-loop fit comparison",
        "",
        f"Generated: {result['generated_at']}",
        "",
        "All five builds and independent integrity audits complete: " + str(result["complete"]),
        "",
        "Only original source observations count toward the 10-transition / 0.5-second length gate.",
        "",
        "| Dataset | Processed / expected | Previous fields | New fields | Original median transitions | Original median seconds | Accepted long-span transition coverage | Passed episodes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for kind, values in datasets.items():
        old, new = values["previous_linear"], values["coherent"]
        if not new:
            rows.append(f"| {kind} | pending | {old['counts']['segments']} | — | — | — | — | — |")
            continue
        rows.append(
            f"| {kind} | {new['episodes']} / {new['expected_episodes']} | {old['counts']['segments']} | "
            f"{new['counts']['segments']} | {new['source_transitions_p10_p50_p90'][1]:.1f} | "
            f"{new['source_seconds_p10_p50_p90'][1]:.3f} | "
            f"{new['eligible_accepted_transition_fraction']:.1%} | {new['passed_episodes']} / {new['episodes']} |"
        )
    rows.extend(
        [
            "",
            "Partial collections are marked by processed / expected counts and must not be compared as complete datasets.",
            "Failed long spans and short command events remain in the saved data. All training and physical validation flags remain false.",
            "",
        ]
    )
    (output / "comparison.md").write_text("\n".join(rows))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = report(args.base, args.output)
    print(json.dumps({"complete": result["complete"], "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
