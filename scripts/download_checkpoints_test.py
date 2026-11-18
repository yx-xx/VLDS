import base64
import dataclasses
import hashlib
import io
import json
import pathlib
import subprocess
from unittest import mock

import google_crc32c
import pytest

from scripts import download_checkpoints


@pytest.fixture
def checkpoint_object():
    content = b"verified checkpoint"
    return content, download_checkpoints.CheckpointObject(
        name="checkpoints/test/params/weights.bin",
        relative_path=pathlib.PurePosixPath("params/weights.bin"),
        size=len(content),
        generation="123",
        md5=hashlib.md5(content, usedforsecurity=False).hexdigest(),
        crc32c=google_crc32c.value(content).to_bytes(4, "big").hex(),
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://storage.googleapis.com/openpi-assets/checkpoints/test",
        "gs://other-bucket/checkpoints/test",
        "gs://openpi-assets/checkpoints",
        "gs://openpi-assets/checkpoints/../test",
        "gs://openpi-assets/checkpoints/test?query=value",
    ],
)
def test_reject_unsupported_checkpoint_locations(url):
    with pytest.raises(ValueError, match="Expected gs"):
        download_checkpoints.checkpoint_objects(url)


def test_composite_object_metadata_does_not_require_md5(checkpoint_object):
    content, item = checkpoint_object
    page = {
        "items": [
            {
                "name": item.name,
                "size": str(len(content)),
                "generation": item.generation,
                "crc32c": base64.b64encode(bytes.fromhex(item.crc32c)).decode(),
            }
        ]
    }
    with mock.patch.object(
        download_checkpoints.urllib.request, "urlopen", return_value=io.BytesIO(json.dumps(page).encode())
    ):
        checkpoint, objects = download_checkpoints.checkpoint_objects("gs://openpi-assets/checkpoints/test")
    assert str(checkpoint) == "checkpoints/test"
    assert objects[0].md5 is None
    assert objects[0].crc32c == item.crc32c
    assert objects[0].url.endswith("?generation=123")


def test_verify_checkpoint_file(checkpoint_object, tmp_path: pathlib.Path):
    content, item = checkpoint_object
    target = tmp_path / item.relative_path
    target.parent.mkdir()
    target.write_bytes(content)
    download_checkpoints.verify_objects(tmp_path, [item])
    download_checkpoints.verify_objects(tmp_path, [dataclasses.replace(item, md5=None)])
    target.write_bytes(b"x" * len(content))
    with pytest.raises(OSError, match="checksum mismatch"):
        download_checkpoints.verify_objects(tmp_path, [item])


def test_missing_file_is_not_verified(checkpoint_object, tmp_path: pathlib.Path):
    _, item = checkpoint_object
    with pytest.raises(OSError, match="Missing or incomplete"):
        download_checkpoints.verify_objects(tmp_path, [item])


def test_aria2_input_pins_generation_and_checksum(checkpoint_object, tmp_path: pathlib.Path):
    _, item = checkpoint_object
    input_file = tmp_path / "input.txt"
    download_checkpoints.write_input(input_file, tmp_path / "staging", [item])
    content = input_file.read_text()
    assert content.splitlines()[0].split("\t") == list(item.urls)
    assert "?generation=123" in content
    assert f"checksum=md5={item.md5}" in content
    assert "out=weights.bin" in content


def test_official_download_endpoints_pin_same_generation(checkpoint_object):
    _, item = checkpoint_object
    assert item.urls == (
        f"https://storage.googleapis.com/openpi-assets/{item.name}?generation={item.generation}",
        f"https://openpi-assets.storage.googleapis.com/{item.name}?generation={item.generation}",
    )


def test_only_verified_downloads_enter_cache(checkpoint_object, tmp_path: pathlib.Path):
    content, item = checkpoint_object
    destination = tmp_path / "openpi-assets/checkpoints/test"
    staging = destination.with_suffix(".partial")
    process = mock.Mock()
    process.wait.return_value = 0

    def start_download(command, env):
        target = staging / item.relative_path
        target.parent.mkdir(parents=True)
        target.write_bytes(content)
        return process

    with (
        mock.patch.object(
            download_checkpoints, "checkpoint_objects", return_value=(pathlib.PurePosixPath("checkpoints/test"), [item])
        ),
        mock.patch.object(download_checkpoints.download, "get_cache_dir", return_value=tmp_path),
        mock.patch.object(download_checkpoints.subprocess, "Popen", side_effect=start_download) as start_process,
    ):
        result = download_checkpoints.fetch_checkpoint(
            "gs://openpi-assets/checkpoints/test", pathlib.Path("/usr/bin/aria2c"), 8, 3
        )
        assert result == destination
        assert destination.is_dir()
        assert not staging.exists()
        download_checkpoints.fetch_checkpoint(
            "gs://openpi-assets/checkpoints/test", pathlib.Path("/usr/bin/aria2c"), 8, 3
        )
    start_process.assert_called_once()
    command = start_process.call_args.args[0]
    assert "--max-connection-per-server=8" in command
    assert "--split=16" in command


def test_interrupted_download_is_not_published(checkpoint_object, tmp_path: pathlib.Path):
    _, item = checkpoint_object
    process = mock.Mock()
    process.wait.return_value = 1
    with (
        mock.patch.object(
            download_checkpoints, "checkpoint_objects", return_value=(pathlib.PurePosixPath("checkpoints/test"), [item])
        ),
        mock.patch.object(download_checkpoints.download, "get_cache_dir", return_value=tmp_path),
        mock.patch.object(download_checkpoints.subprocess, "Popen", return_value=process),
        pytest.raises(subprocess.CalledProcessError),
    ):
        download_checkpoints.fetch_checkpoint(
            "gs://openpi-assets/checkpoints/test", pathlib.Path("/usr/bin/aria2c"), 8, 3
        )
    destination = tmp_path / "openpi-assets/checkpoints/test"
    assert not destination.exists()
    assert (destination.with_suffix(".partial") / ".download_manifest.json").is_file()
