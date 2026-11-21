"""Full datasets replace previews only after the snapshot is verified."""

import json

from openpi.simulation.data.descriptions import dataset_source
from openpi.simulation.data.discovery import discover_datasets


def test_complete_snapshot_replaces_preview_but_partial_download_does_not(tmp_path):
    preview = tmp_path / "dataset_previews/lerobot--pusht"
    full = tmp_path / "datasets/lerobot/pusht"
    for path in (preview, full):
        (path / "meta").mkdir(parents=True)
        (path / "meta/info.json").write_text("{}")
    marker = full / "dataset_source.json"
    source = {"repo_id": "lerobot/pusht", "mode": "full_snapshot", "complete": False}
    marker.write_text(json.dumps(source))
    roots = [tmp_path / "dataset_previews", tmp_path / "datasets"]
    assert discover_datasets(roots) == [preview]
    source["complete"] = True
    marker.write_text(json.dumps(source))
    assert discover_datasets(roots, extra=[preview]) == [full]
    assert dataset_source(full) == "lerobot/pusht"


def test_raw_libero_is_discovered_when_hdf5_files_exist(tmp_path):
    raw = tmp_path / "datasets/raw/libero"
    suite = raw / "libero_spatial"
    suite.mkdir(parents=True)
    (raw / "dataset_source.json").write_text(
        json.dumps({"repo_id": "yifengzhu-hf/LIBERO-datasets", "format": "libero_hdf5"})
    )
    assert discover_datasets([tmp_path / "datasets"]) == []
    (suite / "example.hdf5").touch()
    assert discover_datasets([tmp_path / "datasets"]) == [raw]
