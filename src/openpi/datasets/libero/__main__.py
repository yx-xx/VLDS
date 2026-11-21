"""Process, audit and inspect LIBERO data from one command entry."""

import argparse
from pathlib import Path

from openpi.datasets.libero import processing
from openpi.datasets.libero import reader


def _inspect(args):
    from openpi.datasets.libero.inspection import inspect

    return inspect(args)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    process = commands.add_parser("process", help="Build aligned demonstrations and command-interval candidates")
    process.add_argument("--source", type=Path, default=processing.ROOT / "data/datasets/raw/libero")
    process.add_argument("--output", type=Path, default=processing.ROOT / "data/processed/libero_v2")
    process.add_argument("--suites", nargs="+", choices=processing.SUITES)
    process.add_argument("--tasks", nargs="+")
    process.add_argument("--limit-tasks", type=int)
    process.add_argument("--workers", type=int, default=4)
    process.add_argument("--seed", type=int, default=20261121)
    process.add_argument("--max-candidates", type=int, default=128)
    process.add_argument("--observation-timing", choices=("release_profile", "post_action"), default="release_profile")
    process.add_argument(
        "--libero-root", type=Path, default=processing.ROOT / "examples/libero/.cache/LIBERO/libero/libero"
    )
    process.add_argument(
        "--robosuite-root",
        type=Path,
        default=processing.ROOT / "examples/libero/.venv/lib/python3.8/site-packages/robosuite",
    )
    process.add_argument("--no-resume", dest="resume", action="store_false")
    process.set_defaults(handler=processing.process)
    audit = commands.add_parser("audit", help="Independently verify coverage, images, splits and normalization")
    audit.add_argument("dataset", type=Path)
    audit.add_argument("--skip-images", action="store_true")
    audit.set_defaults(handler=reader.audit)
    inspect = commands.add_parser("inspect", help="Plot an episode or the whole-dataset quality summary")
    inspect.add_argument("dataset", type=Path)
    inspect.add_argument("--episode", type=int, default=0)
    inspect.add_argument("--output", type=Path)
    view = inspect.add_mutually_exclusive_group()
    view.add_argument("--summary", action="store_true")
    view.add_argument(
        "--segments", action="store_true", help="Color the 3D trajectory and timeline by command interval"
    )
    inspect.set_defaults(handler=_inspect)
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (ValueError, RuntimeError, OSError, KeyError, IndexError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
