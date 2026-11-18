import pathlib
from unittest import mock

import pytest

import openpi.shared.download as download


@pytest.fixture(scope="session", autouse=True)
def set_openpi_data_home(tmp_path_factory):
    temp_dir = tmp_path_factory.mktemp("openpi_data")
    with pytest.MonkeyPatch().context() as mp:
        mp.setenv("OPENPI_DATA_HOME", str(temp_dir))
        yield


def test_download_local(tmp_path: pathlib.Path):
    local_path = tmp_path / "local"
    local_path.touch()

    result = download.maybe_download(str(local_path))
    assert result == local_path

    with pytest.raises(FileNotFoundError):
        download.maybe_download("bogus")


def test_download_gs_dir():
    remote_path = "gs://openpi-assets/testdata/random"

    local_path = download.maybe_download(remote_path)
    assert local_path.exists()

    new_local_path = download.maybe_download(remote_path)
    assert new_local_path == local_path


def test_download_gs():
    remote_path = "gs://openpi-assets/testdata/random/random_512kb.bin"

    local_path = download.maybe_download(remote_path)
    assert local_path.exists()

    new_local_path = download.maybe_download(remote_path)
    assert new_local_path == local_path


def test_download_fsspec():
    remote_path = "gs://big_vision/paligemma_tokenizer.model"

    local_path = download.maybe_download(remote_path, gs={"token": "anon"})
    assert local_path.exists()

    new_local_path = download.maybe_download(remote_path, gs={"token": "anon"})
    assert new_local_path == local_path


def test_download_failure_is_not_published(monkeypatch):
    filesystem = mock.Mock()
    filesystem.info.return_value = {"name": "checkpoint", "size": 100, "type": "file"}

    def fail_download(url, destination, recursive):
        destination.write_bytes(b"incomplete")
        raise OSError("Download interrupted")

    filesystem.get.side_effect = fail_download
    monkeypatch.setattr(download.fsspec.core, "url_to_fs", lambda url, **kwargs: (filesystem, "checkpoint"))
    monkeypatch.setattr(download.time, "sleep", lambda duration: None)
    with pytest.raises(OSError, match="Download interrupted"):
        download.maybe_download("memory://checkpoint")
    assert not (download.get_cache_dir() / "checkpoint").exists()
