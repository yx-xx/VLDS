"""Measure boundary slowdown separately from state continuity and source pauses."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from openpi.datasets.libero.reader import atomic_json


def measure(position, nominal, time, records, tolerance):
    position, nominal, time, tolerance = map(np.asarray, (position, nominal, time, tolerance))
    if (
        position.ndim != 2
        or len(position) < 2
        or nominal.shape != position.shape
        or time.shape != (len(position),)
        or tolerance.shape != (position.shape[1],)
        or np.any(tolerance <= 0)
        or np.any(np.diff(time) <= 0)
        or not all(np.isfinite(x).all() for x in (position, nominal, time, tolerance))
    ):
        raise ValueError("Expected aligned finite states, positive time steps and tolerance")
    source_velocity = np.diff(position, axis=0) / np.diff(time)[:, None]
    nominal_velocity = np.diff(nominal, axis=0) / np.diff(time)[:, None]
    boundaries = []
    for first, second in itertools.pairwise(records):
        b = first["end"]
        if second["start"] != b or not 0 < b < len(position) - 1:
            raise ValueError("Boundary observations must be shared without gaps or resets")
        source_in, source_out = source_velocity[b - 1 : b + 1]
        nominal_in, nominal_out = nominal_velocity[b - 1 : b + 1]
        before, after = source_in / tolerance, source_out / tolerance
        source_speed = float(np.linalg.norm(source_in))
        generated_speed = float(np.linalg.norm(nominal_in))
        # Dimensionless tolerance units/s make this diagnostic applicable to
        # Cartesian positions, pixels and joint/gripper state projections.
        moving = bool(min(np.linalg.norm(before), np.linalg.norm(after)) > 0.2 and np.dot(before, after) > 0)
        scaled_speed = float(np.linalg.norm(before))
        ratio = float(np.linalg.norm(nominal_in / tolerance)) / scaled_speed if scaled_speed > 1e-12 else None
        boundaries.append(
            {
                "frame": int(b),
                "source_time": float(time[b] - time[0]),
                "source_continues": moving,
                "source_in_speed_native": source_speed,
                "source_out_speed_native": float(np.linalg.norm(source_out)),
                "nominal_in_speed_native": generated_speed,
                "nominal_out_speed_native": float(np.linalg.norm(nominal_out)),
                "end_speed_ratio": ratio,
                "excess_velocity_jump_scaled": float(
                    np.linalg.norm(((nominal_out - nominal_in) - (source_out - source_in)) / tolerance)
                ),
                "slowdown_candidate": bool(moving and ratio is not None and ratio < 0.5),
            }
        )
    ratios = [b["end_speed_ratio"] for b in boundaries if b["source_continues"]]
    errors = np.linalg.norm((nominal - position) / tolerance, axis=-1)
    return {
        "boundary_count": len(boundaries),
        "moving_boundaries": len(ratios),
        "slowdown_candidates": sum(b["slowdown_candidate"] for b in boundaries),
        "median_end_speed_ratio": float(np.median(ratios)) if ratios else None,
        "maximum_scaled_error": float(errors.max()),
        "boundaries": boundaries,
        "scope": "source-time increments; diagnostic, not a C1 or physical execution certificate",
    }


def _load(path):
    path = Path(path)
    result = json.loads(path.read_text())
    trace = path.parent / result["trace_file"]
    if hashlib.sha256(trace.read_bytes()).hexdigest() != result["trace_sha256"]:
        raise ValueError("Saved trace hash mismatch")
    with np.load(trace, allow_pickle=False) as archive:
        arrays = dict(archive)
    metrics = measure(*(arrays[k] for k in ("position", "nominal", "time")), result["records"], arrays["tolerance"])
    return result, arrays, metrics


def compare(path, output, baseline=None):
    result, arrays, metrics = _load(path)
    previous = None
    if baseline is not None:
        old_result, old_arrays, old_metrics = _load(baseline)
        if any(not np.array_equal(arrays[k], old_arrays[k]) for k in ("position", "time", "tolerance")):
            raise ValueError("Comparison requires identical source observations, time and tolerance")
        previous = (old_result, old_arrays, old_metrics)
    report = {
        "schema": "vlds.dynamics.boundary_diagnostic.v1",
        "result_file": str(Path(path)),
        "result_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "fingerprint": result["fingerprint"],
        "metrics": metrics,
        "baseline": None
        if previous is None
        else {
            "file": str(Path(baseline)),
            "fingerprint": previous[0]["fingerprint"],
            "result_sha256": hashlib.sha256(Path(baseline).read_bytes()).hexdigest(),
            "metrics": previous[2],
        },
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "boundary_report.json", report)
    _plot(output / "boundary_speed.png", result, arrays, previous)
    return report


def _plot(path, result, arrays, previous):
    import matplotlib as mpl

    mpl.use("Agg")
    from matplotlib import pyplot as plt

    time = arrays["time"] - arrays["time"][0]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, layout="constrained")
    axes[0].plot(
        time[1:],
        np.linalg.norm(np.diff(arrays["position"], axis=0), axis=-1) / np.diff(time),
        color="#de881b",
        linewidth=2,
        label="Source demonstration",
    )
    curves = ([] if previous is None else [(previous[1]["nominal"], "#a84ba8", "Endpoint attractors")]) + [
        (arrays["nominal"], "#2176ae", "Pass-through targets")
    ]
    for curve, color, label in curves:
        axes[0].plot(
            time[1:],
            np.linalg.norm(np.diff(curve, axis=0), axis=-1) / np.diff(time),
            color=color,
            linewidth=1.5,
            label=label,
        )
        axes[1].plot(
            time, np.linalg.norm((curve - arrays["position"]) / arrays["tolerance"], axis=-1), color=color, label=label
        )
    for record in result["records"][:-1]:
        for axis in axes:
            axis.axvline(time[record["end"]], color="#888888", linestyle=":", linewidth=0.8)
    units = result.get("coordinates", {}).get("units", "native units")
    axes[0].set_ylabel(f"Speed ({units} / s)")
    axes[1].set(ylabel="Position error / tolerance", xlabel="Source time (s)")
    axes[1].axhline(1, linestyle="--", color="#b34133", label="Maximum error gate")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(loc="upper right", fontsize=8)
    fig.suptitle(
        "Boundary slowdown diagnostic | saved field integration, no state resets\n"
        "Dotted lines: pass-through segment boundaries; source replay is unchanged",
        fontsize=11,
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = compare(args.result, args.output, args.baseline)
    print(json.dumps({k: v for k, v in report["metrics"].items() if k != "boundaries"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
