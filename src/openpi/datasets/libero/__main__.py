"""Process LIBERO demonstrations, fit field teachers, and inspect full-environment diagnostics."""

import argparse
from pathlib import Path

from openpi.datasets.libero import processing
from openpi.datasets.libero import reader


def _inspect(args):
    from openpi.datasets.libero.inspection import inspect

    return inspect(args)


def _fit_fields(args):
    from openpi.datasets.libero.fields import fit_fields

    return fit_fields(args)


def _execute_fields(args):
    from openpi.simulation.runtime.field_execution import execute_fields

    return execute_fields(args)


def _calibrate_fields(args):
    from openpi.datasets.libero.calibration import calibrate_fields

    return calibrate_fields(args)


def _rescore_fields(args):
    from openpi.datasets.libero.calibration import rescore_fields

    return rescore_fields(args)


def _inspect_fields(args):
    from openpi.datasets.libero.field_inspection import inspect_fields

    return inspect_fields(args)


def _audit_fields(args):
    from openpi.datasets.libero.operation_audit import audit_fields

    return audit_fields(args)


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
    fields = commands.add_parser("fit-fields", help="Fit offline stable fields and select finite complete sequences")
    fields.add_argument("dataset", type=Path)
    fields.add_argument("--output", type=Path, required=True)
    fields.add_argument("--episodes", nargs="+", type=int)
    fields.add_argument("--split", choices=("train", "validation", "test"), default="train")
    fields.add_argument("--suite", choices=processing.SUITES)
    fields.add_argument("--max-episodes", type=int, default=3)
    fields.add_argument("--max-speed", type=float, choices=(0.1, 0.2, 0.3), default=0.1)
    fields.add_argument("--grid-seconds", type=float, choices=(0.25, 0.5), default=0.5)
    fields.add_argument("--stability-margin", type=float, choices=(0.01, 0.1, 0.5), default=0.01)
    fields.add_argument("--representation", choices=("fitted", "linear"), default="fitted")
    fields.add_argument(
        "--source-replay", type=Path, help="Use a complete successful source replay as derived pose teacher"
    )
    fields.set_defaults(handler=_fit_fields)
    execute = commands.add_parser("execute-fields", help="Run field or source-action diagnostics in full LIBERO")
    execute.add_argument("dataset", type=Path)
    execute.add_argument("--episode", type=int, required=True)
    execute.add_argument("--fields", type=Path)
    execute.add_argument("--source-actions", action="store_true")
    execute.add_argument(
        "--gripper-feedback",
        action="store_true",
        help="Require measured settling after command changes; not a grasp detector",
    )
    execute.add_argument(
        "--audit-stride",
        type=int,
        default=0,
        help="Save evaluation-only images every N control steps; 0 disables extra images",
    )
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--max-sequences", type=int, choices=(1, 2, 3), default=3)
    execute.add_argument("--seed", type=int, default=20261122)
    execute.add_argument("--budget-multiplier", type=float, choices=(1, 1.5), default=1)
    execute.add_argument("--timeout", type=float, default=180)
    execute.set_defaults(handler=_execute_fields)
    calibration = commands.add_parser("calibrate-fields", help="Recheck fixed field teachers at three allowed speeds")
    calibration.add_argument("dataset", type=Path)
    calibration.add_argument("--fields-dir", type=Path, required=True)
    calibration.add_argument("--output", type=Path, required=True)
    calibration.set_defaults(handler=_calibrate_fields)
    rescore = commands.add_parser("rescore-fields", help="Recheck a frozen field pool with one baseline/reference mode")
    rescore.add_argument("dataset", type=Path)
    rescore.add_argument("--fields-dir", type=Path, required=True)
    rescore.add_argument("--representation", choices=("fitted", "linear"), default="fitted")
    rescore.add_argument("--reference-mode", choices=("delta", "damping-compensated"), default="delta")
    rescore.add_argument("--arrival-position-tolerance", type=float, choices=(0.002, 0.005))
    rescore.add_argument("--output", type=Path, required=True)
    rescore.set_defaults(handler=_rescore_fields)
    field_view = commands.add_parser("inspect-fields", help="Plot actual field execution, goals and query images")
    field_view.add_argument("dataset", type=Path)
    field_view.add_argument("--fields", type=Path, required=True)
    field_view.add_argument("--execution", type=Path, required=True)
    field_view.add_argument("--trial", type=int, default=0)
    field_view.add_argument("--output", type=Path, required=True)
    field_view.set_defaults(handler=_inspect_fields)
    operations = commands.add_parser(
        "audit-fields", help="Audit operation evidence and propose unverified source events"
    )
    operations.add_argument("execution", type=Path)
    operations.add_argument("--compare-source", type=Path)
    operations.add_argument("--output", type=Path, required=True)
    operations.add_argument("--figure", type=Path)
    operations.set_defaults(handler=_audit_fields)
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (ValueError, RuntimeError, OSError, KeyError, IndexError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
