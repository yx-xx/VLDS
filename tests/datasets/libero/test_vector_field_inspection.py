import dataclasses
import json
from types import SimpleNamespace

import pytest

from openpi.datasets.libero import replay_teacher
from openpi.datasets.libero import vector_field_inspection
from openpi.fields.primitive import DEFAULT_CONTROL


def test_saved_field_rejects_wrong_episode_manifest_and_reference_trajectory(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "manifest.json").write_text('{"source":"fixture"}')
    entry = {"signature": "demo-signature", "shard": "task.h5", "group": "/demo", "split": "train"}
    dataset = SimpleNamespace(root=root, episodes=[entry])
    saved = tmp_path / "fields"
    saved.mkdir()
    path = saved / "episode_00000.json"
    result = {"schema": "vlds.libero.offline_fields.v1", "source_episode": entry, "fingerprint": "fit-run"}
    run = {
        "fingerprint": "fit-run",
        "config": {
            "source_manifest_sha256": replay_teacher.digest(root / "manifest.json"),
            "control": dataclasses.asdict(DEFAULT_CONTROL),
        },
    }
    (saved / "run.json").write_text(json.dumps(run))
    path.write_text(json.dumps(result))
    loaded, control = vector_field_inspection.load_result(dataset, 0, path)
    assert loaded == result
    assert control == DEFAULT_CONTROL
    result["source_episode"] = {**entry, "signature": "other-demo"}
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="another source episode"):
        vector_field_inspection.load_result(dataset, 0, path)
    result["source_episode"] = entry
    result["teacher_source"] = {"file": "different-reference.npz"}
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="reference trajectory"):
        vector_field_inspection.load_result(dataset, 0, path)
    result.pop("teacher_source")
    path.write_text(json.dumps(result))
    (root / "manifest.json").write_text('{"source":"changed"}')
    with pytest.raises(ValueError, match="manifest"):
        vector_field_inspection.load_result(dataset, 0, path)
