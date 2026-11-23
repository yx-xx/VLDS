import dataclasses
import json
import shutil

import pytest
from test_build import source_fixture

from openpi.datasets.dynamics.build import build as build_base
from openpi.datasets.dynamics.coherent import DEFAULT_CONFIG
from openpi.datasets.dynamics.coherent import build
from openpi.datasets.dynamics.coherent_audit import audit
from openpi.datasets.dynamics.playback import available_profiles
from openpi.datasets.dynamics.playback import load_saved_episode


def test_manual_partial_pilot_requires_audit_recomputes_fields_and_checks_episode_identity(tmp_path):
    root, source, episode = source_fixture(tmp_path)
    second = source.with_name("episode_000001.parquet")
    shutil.copyfile(source, second)
    with (root / "meta/episodes.jsonl").open("a") as stream:
        stream.write(json.dumps({"episode_index": 1, "length": 8}) + "\n")
    base, pilot = tmp_path / "base", tmp_path / "pilot"
    build_base(root, base, workers=1)
    pointer = (root / ".vlds/dynamics.json").read_bytes()
    config = dataclasses.replace(DEFAULT_CONFIG, rollout_iterations=0, goal_lookahead_steps=4)
    run = build(base, output=pilot, indices=[0], workers=1, config=config)
    assert not run["summary"]["full_collection"]
    path = pilot / "episode_00000.json"
    with pytest.raises(ValueError, match="独立完整性复核"):
        load_saved_episode(episode, path)
    assert audit(pilot)["integrity_passed"]
    bundle = load_saved_episode(episode, path)
    assert bundle["manual"]
    assert bundle["profile"] == "coherent_flow"
    assert len(bundle["arrays"]["nominal"]) == episode.frames
    assert "coherent_flow" not in available_profiles(episode)
    assert (root / ".vlds/dynamics.json").read_bytes() == pointer
    with pytest.raises(ValueError, match="索引"):
        load_saved_episode(dataclasses.replace(episode, path=second), path)
    # A stale audit.json cannot legitimize changed parameters, even if the
    # saved trace and source still have the expected hashes.
    altered = json.loads(path.read_text())
    altered["records"][0]["goal"][0] += 0.5
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="独立重算失败"):
        load_saved_episode(episode, path)
