"""Build all complete local dataset families and register their playback results."""

import argparse
import json
import os
from pathlib import Path

from openpi.datasets.dynamics.build import build
from openpi.datasets.libero.reader import atomic_json

SOURCES = (
    ("processed/libero_v2", "libero"),
    ("datasets/lerobot/aloha_sim_transfer_cube_human", "aloha"),
    ("datasets/lerobot/pusht", "pusht"),
    ("datasets/lerobot/xarm_lift_medium", "xarm"),
    ("datasets/physical-intelligence/libero", "lerobot_libero"),
)


def register_alias(source, output):
    source, output = Path(source), Path(output)
    run = json.loads((output / "run.json").read_text())
    if run["status"] != "complete" or not run["summary"]["full_collection"]:
        raise ValueError("Only complete offline collections can be registered")
    pointer = source / ".vlds/dynamics.json"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(
        pointer,
        {
            "schema": "vlds.dynamics.pointer.v1",
            "directory": os.path.relpath(output, pointer.parent),
            "fingerprint": run["fingerprint"],
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--register-only", action="store_true", help="Register already completed outputs without rebuilding"
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("workers must be positive")
    results = {}
    for relative, name in SOURCES:
        root = args.data / relative
        if not root.exists():
            continue
        output = args.output / name
        if not args.register_only:
            build(root, output, workers=args.workers)
        register_alias(root, output)
        results[name] = json.loads((output / "summary.json").read_text())
        if name == "libero":
            raw = args.data / "datasets/raw/libero"
            if raw.exists():
                register_alias(raw, output)
        else:
            preview = args.data / "dataset_previews" / relative.removeprefix("datasets/").replace("/", "--")
            if preview.exists():
                register_alias(preview, output)
    atomic_json(
        args.output / "collections.json",
        {
            "schema": "vlds.dynamics.collections.v1",
            "datasets": results,
            "episodes": sum(r["episodes"] for r in results.values()),
            "segments": sum(r["segments"] for r in results.values()),
            "all_nominal_passed": all(r["passed_episodes"] == r["expected_episodes"] for r in results.values()),
        },
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
