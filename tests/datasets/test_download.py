"""Verify snapshot integrity, safe paths, resume behavior and completion markers."""

import hashlib
import json
from unittest import mock

import pytest

from openpi.datasets.download import DatasetDownloader
from openpi.datasets.sources import SOURCES


def manifest_file(tmp_path, files, dataset="pusht"):
    manifest = {"id": SOURCES[dataset].repo_id, "sha": "a" * 40, "siblings": files}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def entry(name, payload, *, lfs=True):
    value = {"rfilename": name, "size": len(payload)}
    if lfs:
        value["lfs"] = {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    else:
        value["blobId"] = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
    return value


def response(payload, *, status=200, headers=None):
    result = mock.MagicMock(status_code=status, headers=headers or {})
    result.__enter__.return_value = result
    result.iter_content.return_value = [payload]
    return result


def test_complete_snapshot_reuses_only_verified_preview_files(tmp_path):
    payload = b"complete dataset contents"
    file = entry("data/example.parquet", payload)
    preview = tmp_path / "dataset_previews/lerobot--pusht/data/example.parquet"
    preview.parent.mkdir(parents=True)
    preview.write_bytes(payload)
    downloader = DatasetDownloader(tmp_path / "datasets", endpoint="https://hf-mirror.com")
    with mock.patch.object(downloader, "session") as session:
        downloader.run(["pusht"], manifests={"pusht": manifest_file(tmp_path, [file])})
    session.assert_not_called()
    destination = tmp_path / "datasets/lerobot/pusht"
    assert (destination / "data/example.parquet").read_bytes() == payload
    source = json.loads((destination / "dataset_source.json").read_text())
    assert source["complete"]
    assert source["revision"] == "a" * 40
    assert source["files"] == 1


def test_resume_requires_matching_content_range_and_full_hash(tmp_path, monkeypatch):
    payload = b"0123456789"
    file = entry("data/example.parquet", payload)
    downloader = DatasetDownloader(tmp_path / "datasets", endpoint="https://hf-mirror.com")
    dataset = downloader.prepare("pusht", manifest_file=manifest_file(tmp_path, [file]))
    partial = dataset[1] / "data/example.parquet.part"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(payload[:3])
    session = mock.Mock()
    session.get.side_effect = [
        response(payload[3:], status=206, headers={"Content-Range": "bytes 0-6/10"}),
        response(payload[3:], status=206, headers={"Content-Range": "bytes 3-9/10"}),
    ]
    monkeypatch.setattr(downloader, "session", lambda: session)
    monkeypatch.setattr("openpi.datasets.download.time.sleep", lambda _: None)
    downloader.fetch(dataset, file)
    assert (dataset[1] / "data/example.parquet").read_bytes() == payload
    assert session.get.call_args.kwargs["headers"] == {"Range": "bytes=3-"}
    assert not partial.exists()


def test_corrupt_files_never_mark_snapshot_complete(tmp_path, monkeypatch):
    payload = b"valid data"
    file = entry("data/example.parquet", payload)
    downloader = DatasetDownloader(tmp_path / "datasets", endpoint="https://hf-mirror.com", workers=1)
    session = mock.Mock()
    session.get.side_effect = lambda *args, **kwargs: response(b"x" * len(payload))
    monkeypatch.setattr(downloader, "session", lambda: session)
    monkeypatch.setattr("openpi.datasets.download.time.sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="1 files failed"):
        downloader.run(["pusht"], manifests={"pusht": manifest_file(tmp_path, [file])})
    destination = tmp_path / "datasets/lerobot/pusht"
    assert not (destination / "data/example.parquet").exists()
    assert not json.loads((destination / "dataset_source.json").read_text())["complete"]
    assert len(json.loads((tmp_path / "datasets/download_status.json").read_text())["errors"]) == 1


@pytest.mark.parametrize("name", ["../escape", "/tmp/escape", "data/../../escape"])
def test_manifest_cannot_escape_dataset_directory(tmp_path, name):
    downloader = DatasetDownloader(tmp_path / "datasets", endpoint="https://hf-mirror.com")
    with pytest.raises(ValueError, match="Unsafe dataset path"):
        downloader.prepare("pusht", manifest_file=manifest_file(tmp_path, [entry(name, b"data")]))


def test_git_blob_hash_detects_same_size_corruption(tmp_path):
    path = tmp_path / "info.json"
    path.write_bytes(b"abcd")
    file = entry(path.name, b"abcd", lfs=False)
    assert DatasetDownloader.verify(path, file)
    path.write_bytes(b"abce")
    assert not DatasetDownloader.verify(path, file)


def test_parallel_ranges_reuse_streaming_prefix_and_verify_whole_file(tmp_path, monkeypatch):
    prefix = b"x" * (1024 * 1024)
    payload = prefix + b"last chunk"
    file = entry("data/example.parquet", payload)
    downloader = DatasetDownloader(tmp_path / "datasets", endpoint="https://hf-mirror.com", parts=2)
    dataset = downloader.prepare("pusht", manifest_file=manifest_file(tmp_path, [file]))
    target = dataset[1] / file["rfilename"]
    target.parent.mkdir(parents=True)
    partial = target.with_suffix(target.suffix + ".part")
    partial.write_bytes(prefix)
    session = mock.Mock()
    session.get.side_effect = [
        response(b"", status=302, headers={"Location": "https://cdn.example/immutable"}),
        response(
            payload[len(prefix) :],
            status=206,
            headers={"Content-Range": f"bytes {len(prefix)}-{len(payload) - 1}/{len(payload)}"},
        ),
    ]
    monkeypatch.setattr(downloader, "session", lambda: session)
    downloader.fetch_ranges(dataset, file, target)
    assert target.read_bytes() == payload
    assert downloader.count == 1
    assert not partial.exists()
    assert not (target.parent / ("." + target.name + ".ranges")).exists()
    assert session.get.call_args.kwargs["headers"] == {"Range": f"bytes={len(prefix)}-{len(payload) - 1}"}


@pytest.mark.parametrize(("status", "content_range"), [(200, None), (206, "bytes 1-9/10")])
def test_parallel_ranges_reject_ignored_or_incorrect_ranges(tmp_path, monkeypatch, status, content_range):
    file = entry("data/example.parquet", b"0123456789")
    downloader = DatasetDownloader(tmp_path / "datasets", endpoint="https://hf-mirror.com", parts=2)
    dataset = downloader.prepare("pusht", manifest_file=manifest_file(tmp_path, [file]))
    target = dataset[1] / file["rfilename"]
    target.parent.mkdir(parents=True)
    session = mock.Mock()

    def get(*args, **kwargs):
        if "allow_redirects" in kwargs:
            return response(b"", status=302, headers={"Location": "https://cdn.example/immutable"})
        return response(b"0123456789", status=status, headers={"Content-Range": content_range})

    session.get.side_effect = get
    monkeypatch.setattr(downloader, "session", lambda: session)
    monkeypatch.setattr("openpi.datasets.download.time.sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="Range 0-9 failed"):
        downloader.fetch_ranges(dataset, file, target)
    assert not target.exists()
    assert downloader.count == 0


def test_parallel_ranges_never_publish_same_size_corruption(tmp_path, monkeypatch):
    file = entry("data/example.parquet", b"correct")
    downloader = DatasetDownloader(tmp_path / "datasets", endpoint="https://hf-mirror.com", parts=2)
    dataset = downloader.prepare("pusht", manifest_file=manifest_file(tmp_path, [file]))
    target = dataset[1] / file["rfilename"]
    target.parent.mkdir(parents=True)
    session = mock.Mock()
    session.get.side_effect = [
        response(b"", status=302, headers={"Location": "https://cdn.example/immutable"}),
        response(b"corrupt", status=206, headers={"Content-Range": "bytes 0-6/7"}),
    ]
    monkeypatch.setattr(downloader, "session", lambda: session)
    with pytest.raises(ValueError, match="content hash"):
        downloader.fetch_ranges(dataset, file, target)
    assert not target.exists()
    assert downloader.count == 0
    assert not (target.parent / ("." + target.name + ".ranges")).exists()


def test_mirror_preserves_canonical_revision_and_verifies_contents(tmp_path, monkeypatch):
    payload = b"canonical LIBERO contents"
    file = entry("data/example.parquet", payload)
    downloader = DatasetDownloader(tmp_path / "datasets", endpoint="https://hf-mirror.com", libero_mirror=True)
    session = mock.Mock()
    session.get.return_value = response(payload)
    monkeypatch.setattr(downloader, "session", lambda: session)
    downloader.run(["libero"], manifests={"libero": manifest_file(tmp_path, [file], dataset="libero")})
    destination = tmp_path / "datasets/physical-intelligence/libero"
    source = json.loads((destination / "dataset_source.json").read_text())
    assert source["complete"]
    assert source["repo_id"] == "physical-intelligence/libero"
    assert source["revision"] == "a" * 40
    assert source["transport_mirrors"]
    assert session.get.call_args.args[0].startswith("https://modelscope.cn/api/v1/datasets/")
    assert (destination / file["rfilename"]).read_bytes() == payload


def test_unavailable_mirror_falls_back_to_pinned_huggingface_snapshot(tmp_path, monkeypatch):
    payload = b"correct"
    file = entry("data/example.parquet", payload)
    downloader = DatasetDownloader(tmp_path / "datasets", endpoint="https://hf-mirror.com", parts=2, libero_mirror=True)
    dataset = downloader.prepare("libero", manifest_file=manifest_file(tmp_path, [file], dataset="libero"))
    target = dataset[1] / file["rfilename"]
    target.parent.mkdir(parents=True)
    session = mock.Mock()
    session.get.side_effect = [
        response(b"", status=503),
        response(b"", status=302, headers={"Location": "https://cdn.example/immutable"}),
        response(payload, status=206, headers={"Content-Range": "bytes 0-6/7"}),
    ]
    monkeypatch.setattr(downloader, "session", lambda: session)
    downloader.fetch_ranges(dataset, file, target)
    assert target.read_bytes() == payload
    assert "/resolve/" + "a" * 40 + "/" in session.get.call_args_list[1].args[0]


def test_corrupt_streaming_prefix_is_discarded_before_range_retry(tmp_path, monkeypatch):
    payload = b"correct"
    file = entry("data/example.parquet", payload)
    downloader = DatasetDownloader(tmp_path / "datasets", endpoint="https://hf-mirror.com", parts=2)
    dataset = downloader.prepare("pusht", manifest_file=manifest_file(tmp_path, [file]))
    target = dataset[1] / file["rfilename"]
    target.parent.mkdir(parents=True)
    partial = target.with_suffix(target.suffix + ".part")
    partial.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="content hash"):
        downloader.fetch_ranges(dataset, file, target)
    assert not partial.exists()
    session = mock.Mock()
    session.get.side_effect = [
        response(b"", status=302, headers={"Location": "https://cdn.example/immutable"}),
        response(payload, status=206, headers={"Content-Range": "bytes 0-6/7"}),
    ]
    monkeypatch.setattr(downloader, "session", lambda: session)
    downloader.fetch_ranges(dataset, file, target)
    assert target.read_bytes() == payload
