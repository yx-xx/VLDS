"""Export one sampled HDF5 demonstration with an explicit original-scene replay link."""

import argparse
import io
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

from openpi.simulation.data.reader import EpisodeReader
from openpi.simulation.data.reader import scan_dataset


def export_episode(source, group, output, *, stride=1, start=0, stop=None, camera=None):
    source, output = Path(source).expanduser().resolve(), Path(output).expanduser().resolve()
    if not source.is_file() or source.suffix.lower() not in (".h5", ".hdf5"):
        raise ValueError("Source must be an original HDF5 demonstration file.")
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    if stride <= 0 or start < 0:
        raise ValueError("Stride must be positive and start must be nonnegative.")
    episode = next((ep for ep in scan_dataset(source).episodes if ep.group == group), None)
    if episode is None or not episode.replay or episode.replay_source:
        raise ValueError("Select an original HDF5 group containing model_file and states.")
    stop = episode.frames if stop is None else stop
    if not start < stop <= episode.frames:
        raise ValueError(f"Require 0 <= start < stop <= {episode.frames}.")
    camera = camera or next(iter(episode.cameras), None)
    if camera is not None and camera not in episode.cameras:
        raise ValueError(f"Unknown camera: {camera}")
    indices = list(range(start, stop, stride))
    columns = {key: [] for key in ("index", "episode_index", "frame_index", "task_index", "timestamp")}
    fps = episode.fps / stride
    features = {
        **{key: {"dtype": "int64", "shape": [1], "names": None} for key in columns if key != "timestamp"},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    }
    reader = EpisodeReader(episode)
    try:
        for frame, source_frame in enumerate(indices):
            image, values = reader.read(source_frame, camera)
            for key, value in (
                ("index", frame),
                ("episode_index", 0),
                ("frame_index", frame),
                ("task_index", 0),
                ("timestamp", frame / fps),
            ):
                columns[key].append(value)
            numeric = {}
            if "actions" in values:
                numeric["actions"] = values["actions"]
            if "states" in values:
                numeric["simulation.state"] = values["states"]
            if "obs/ee_states" in values and "obs/gripper_states" in values:
                numeric["state"] = [*values["obs/ee_states"], *values["obs/gripper_states"]]
            for key, value in numeric.items():
                vector = np.asarray(value, dtype=np.float32)
                if vector.ndim != 1 or not np.isfinite(vector).all():
                    raise ValueError(f"Invalid vector {key} at source frame {source_frame}.")
                columns.setdefault(key, []).append(vector.tolist())
                features[key] = {"dtype": "float32", "shape": list(vector.shape), "names": None}
            if image is not None:
                buffer = io.BytesIO()
                Image.fromarray(image).save(buffer, format="PNG")
                columns.setdefault("image", []).append({"bytes": buffer.getvalue(), "path": None})
                features["image"] = {
                    "dtype": "image",
                    "shape": list(image.shape),
                    "names": ["height", "width", "channels"],
                }
    finally:
        reader.close()
    # Write a complete new directory, publishing it only after all files are ready.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary) / "dataset"
        (staging / "meta").mkdir(parents=True)
        (staging / "data/chunk-000").mkdir(parents=True)
        arrays = {}
        for key, values in columns.items():
            feature = features[key]
            dtype = feature["dtype"]
            if dtype == "image":
                arrays[key] = pa.array(values)
            elif isinstance(values[0], list):
                arrays[key] = pa.array(values, type=pa.list_(pa.float32(), feature["shape"][0]))
            else:
                arrays[key] = pa.array(values, type=pa.float32() if dtype == "float32" else pa.int64())
        pq.write_table(pa.table(arrays), staging / "data/chunk-000/episode_000000.parquet", row_group_size=128)
        info = {
            "codebase_version": "v2.1",
            "robot_type": "simulation",
            "fps": fps,
            "total_episodes": 1,
            "total_frames": len(indices),
            "total_tasks": 1,
            "total_chunks": 1,
            "total_videos": 0,
            "chunks_size": 1000,
            "splits": {"train": "0:1"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "features": features,
        }
        (staging / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")
        (staging / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": episode.task}) + "\n")
        (staging / "meta/episodes.jsonl").write_text(
            json.dumps({"episode_index": 0, "tasks": [episode.task], "length": len(indices)}) + "\n"
        )
        manifest = {
            "version": 1,
            "episodes": {
                "0": {
                    "path": os.path.relpath(source, output),
                    "group": group,
                    "frame_indices": indices,
                }
            },
        }
        (staging / "meta/replay.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staging.rename(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--group", required=True, help="HDF5 group, for example data/demo_0.")
    parser.add_argument(
        "--output", type=Path, required=True, help="A new directory; existing data is never overwritten."
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--camera")
    args = parser.parse_args()
    try:
        result = export_episode(
            args.source,
            args.group,
            args.output,
            stride=args.stride,
            start=args.start,
            stop=args.stop,
            camera=args.camera,
        )
    except (OSError, ValueError) as error:
        parser.exit(1, f"Export failed: {error}\n")
    print(result)


if __name__ == "__main__":
    main()
