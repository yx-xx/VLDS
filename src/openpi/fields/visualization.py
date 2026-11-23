"""Sample a fixed pose field and replay it with the same nominal acceptance integrator."""

import dataclasses
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from openpi.fields import primitive as primitives


def sample_field(
    primitive,
    position,
    rotations,
    time,
    control=primitives.DEFAULT_CONTROL,
    *,
    density=7,
    saved_trace=None,
    saved_report=None,
):
    """Return physical velocities, display arrows and time-aligned errors; no robot dynamics."""
    if not 3 <= density <= 11:
        raise ValueError("Field grid density must be between 3 and 11")
    position, rotations, time = np.asarray(position), np.asarray(rotations), np.asarray(time)
    if saved_trace is None:
        report = primitives.nominal_replay(primitive, position, rotations, time, control, keep_trace=True)
        trace = report.pop("trace")
    else:
        trace, report = saved_trace, dict(saved_report)
        if (
            trace["position"].shape != position.shape
            or trace["rotation"].shape != rotations.shape
            or not np.isfinite(trace["position"]).all()
            or not np.isfinite(trace["rotation"]).all()
        ):
            raise ValueError("Saved continuous trace is not aligned with the demonstration")
    nominal = trace["position"]
    points = np.vstack([position, primitive.field.goal])
    lower, upper = points.min(axis=0), points.max(axis=0)
    padding = np.maximum((upper - lower) * 0.15, 0.025)
    axes = [np.linspace(a, b, density) for a, b in zip(lower - padding, upper + padding, strict=True)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    # Use the exact limited field used by nominal integration, including the Euler descent bound.
    velocity = np.array([primitives.feedback(primitive, p, primitive.rotation, control)[0] for p in grid])
    speed = np.linalg.norm(velocity, axis=1)
    arrow_seconds = float(max(np.diff(axis).min() for axis in axes) * 0.7 / max(speed.max(), 1e-12))
    demo_field = np.array(
        [primitives.feedback(primitive, p, r, control)[0] for p, r in zip(position, rotations, strict=True)]
    )
    angle_error = Rotation.from_matrix(rotations @ trace["rotation"][: len(time)].transpose(0, 2, 1)).magnitude()
    arrays = {
        "grid": grid,
        "velocity": velocity,
        "arrows": velocity * arrow_seconds,
        "demo": position,
        "nominal": nominal,
        "goal": primitive.field.goal,
        "time": time - time[0],
        "position_error": np.linalg.norm(nominal[: len(time)] - position, axis=1),
        "rotation_error": angle_error,
        "field_speed_on_demo": np.linalg.norm(demo_field, axis=1),
    }
    metadata = {
        "schema": "vlds.field_preview.v1",
        "frame": "world",
        "control_point": "gripper0_grip_site",
        "scope": "nominal field integration; no robot dynamics or operation-success validation",
        "control": dataclasses.asdict(control),
        "primitive": primitive.record(),
        "report": report,
        "arrow_seconds": arrow_seconds,
        "density": density,
    }
    return arrays, metadata


def save_snapshot(path, arrays, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez_compressed(stream, metadata=json.dumps(metadata, ensure_ascii=False, allow_nan=False), **arrays)


def plot_field(path, arrays, metadata):
    """Export a stand-alone diagnostic with a physical, equal-scale 3D field and time plots."""
    import matplotlib as mpl

    mpl.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.colors import Normalize

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(13, 7), layout="constrained")
    layout = fig.add_gridspec(2, 2, width_ratios=(1.6, 1))
    ax = fig.add_subplot(layout[:, 0], projection="3d")
    grid, arrows = arrays["grid"], arrays["arrows"]
    speed = np.linalg.norm(arrays["velocity"], axis=1)
    norm = Normalize(0, max(speed.max(), 1e-9))
    colors = plt.colormaps["viridis"](norm(speed))
    ax.quiver(*grid.T, *arrows.T, colors=colors, linewidth=0.55, alpha=0.6, arrow_length_ratio=0.25)
    ax.plot(*arrays["demo"].T, color="#ed8b23", linewidth=3, label="Demonstration")
    count = len(arrays["demo"])
    ax.plot(*arrays["nominal"][:count].T, color="#cf359d", linewidth=2.2, label="Nominal field rollout")
    ax.plot(*arrays["nominal"][count - 1 :].T, color="#cf359d", linestyle="--", label="Extra convergence")
    ax.scatter(*arrays["demo"][0], color="#ed8b23", s=45, label="Start")
    ax.scatter(*arrays["goal"], color="#1c9b51", marker="*", s=180, label="Goal")
    extent = np.ptp(grid, axis=0)
    ax.set_box_aspect(extent)
    ax.set(xlabel="World x (m)", ylabel="World y (m)", zlabel="World z (m)")
    ax.view_init(elev=25, azim=-55)
    ax.legend(loc="upper left", fontsize=8)
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax, shrink=0.55, label="Limited field speed (m/s)"
    )
    errors = fig.add_subplot(layout[0, 1])
    errors.plot(arrays["time"], arrays["position_error"] * 1000, color="#cf359d", label="Position error")
    errors.axhline(
        metadata["control"]["position_rms_tolerance"] * 1000, linestyle="--", color="gray", label="RMS gate (reference)"
    )
    errors.set(xlabel="Source time (s)", ylabel="Position error (mm)")
    errors.grid(alpha=0.2)
    errors.legend(fontsize=8)
    speeds = fig.add_subplot(layout[1, 1])
    if "demo_speed" in arrays:
        speeds.plot(arrays["time"], arrays["demo_speed"], color="#ed8b23", label="Measured demo speed")
    speeds.plot(arrays["time"], arrays["field_speed_on_demo"], color="#217c99", label="Field speed at demo positions")
    speeds.set(xlabel="Source time (s)", ylabel="Speed (m/s)")
    speeds.grid(alpha=0.2)
    speeds.legend(fontsize=8)
    report = metadata["report"]
    status = "PASS" if report["passed"] else "REJECT: " + ", ".join(report["reasons"])
    interval = metadata.get("interval", {})
    fig.suptitle(
        f"{metadata.get('label', 'Fixed dynamical field')} | [{interval.get('start', '?')}, {interval.get('end', '?')}]\n"
        f"Nominal {status} | position RMS {report['position_rms_m'] * 1000:.2f} mm | "
        f"rotation RMS {np.degrees(report['rotation_rms_rad']):.2f} deg\n"
        "Position arrows only; nominal rollout is not physical task execution",
        fontsize=11,
    )
    try:
        fig.savefig(path, dpi=170)
    finally:
        plt.close(fig)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    save_snapshot(path.with_suffix(".npz"), arrays, metadata)
    return path
