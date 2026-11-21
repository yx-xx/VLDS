"""Repository and standalone worker locations, independent of the current working directory."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LIBERO_ROOT = ROOT / "examples/libero/.cache/LIBERO/libero/libero"
WORKERS_ROOT = Path(__file__).resolve().parent / "workers"
