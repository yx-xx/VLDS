"""Download immutable dataset snapshots with resumable files and content verification."""

import concurrent.futures
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
from urllib.parse import quote

import requests

from openpi.datasets.sources import SOURCES


class DatasetDownloader:
    def __init__(self, root: Path, *, endpoint: str, workers: int = 8, parts: int = 1, libero_mirror: bool = False):
        self.root = root.resolve()
        self.endpoint = endpoint.rstrip("/")
        self.workers = workers
        self.parts = parts
        self.libero_mirror = libero_mirror
        self.lock = threading.Lock()
        self.count = 0
        self.completed_bytes = 0
        self.transferred_bytes = 0
        self.started = time.monotonic()
        self.local = threading.local()

    def session(self):
        if not hasattr(self.local, "session"):
            self.local.session = requests.Session()
        return self.local.session

    @staticmethod
    def write_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(path)

    @staticmethod
    def relative_path(value):
        path = Path(value)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError(f"Unsafe dataset path: {value}")
        return path

    @staticmethod
    def expected_size(file):
        return file.get("lfs", {}).get("size", file["size"])

    @staticmethod
    def verify(path, file):
        if not path.is_file() or path.stat().st_size != DatasetDownloader.expected_size(file):
            return False
        lfs = file.get("lfs")
        digest = hashlib.sha256() if lfs else hashlib.sha1()
        if not lfs:
            digest.update(f"blob {path.stat().st_size}\0".encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
        expected = lfs["sha256"] if lfs else file["blobId"]
        return digest.hexdigest() == expected

    def prepare(self, name, *, manifest_file=None):
        source = SOURCES[name]
        destination = self.root / source.repo_id if source.format == "lerobot" else self.root / "raw/libero"
        manifest_path = destination / ".vlds/manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
        elif manifest_file:
            manifest = json.loads(Path(manifest_file).read_text())
        else:
            url = f"{self.endpoint}/api/datasets/{source.repo_id}/revision/{quote(source.revision, safe='')}"
            response = self.session().get(url, params={"blobs": "true"}, timeout=(20, 60))
            response.raise_for_status()
            manifest = response.json()
        if manifest.get("id") != source.repo_id or not manifest.get("sha") or not manifest.get("siblings"):
            raise ValueError(f"Invalid dataset manifest: {name}")
        files = manifest["siblings"]
        for file in files:
            self.relative_path(file["rfilename"])
            self.expected_size(file)
            if not file.get("lfs", {}).get("sha256") and not file.get("blobId"):
                raise ValueError(f"Missing content hash: {file['rfilename']}")
        destination.mkdir(parents=True, exist_ok=True)
        self.write_json(manifest_path, manifest)
        provenance = {
            "repo_id": source.repo_id,
            "requested_revision": source.revision,
            "revision": manifest["sha"],
            "format": source.format,
            "endpoint": self.endpoint,
            "mode": "full_snapshot",
            "complete": False,
            "files": len(files),
            "bytes": sum(self.expected_size(file) for file in files),
        }
        if self.libero_mirror and name == "libero":
            provenance["transport_mirrors"] = [
                "https://modelscope.cn/datasets/buxiangqimingzi/physical-intelligence-libero"
            ]
        self.write_json(destination / "dataset_source.json", provenance)
        return name, destination, manifest, provenance

    def reuse_preview(self, destination, file, repo):
        preview = self.root.parent / "dataset_previews" / repo.replace("/", "--") / file["rfilename"]
        target = destination / self.relative_path(file["rfilename"])
        if not target.exists() and self.verify(preview, file):
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(preview, target)
            except OSError:
                shutil.copyfile(preview, target)

    def file_url(self, repo_id, revision, filename, *, mirror=True):
        if self.libero_mirror and mirror and repo_id == "physical-intelligence/libero":
            return (
                "https://modelscope.cn/api/v1/datasets/buxiangqimingzi/physical-intelligence-libero/repo"
                f"?Source=SDK&Revision=master&FilePath={quote(filename, safe='')}"
            )
        return f"{self.endpoint}/datasets/{repo_id}/resolve/{revision}/{quote(filename, safe='/')}"

    def fetch_ranges(self, dataset, file, target):
        """Resume bounded byte ranges, then publish only after whole-file verification."""
        name, _destination, manifest, provenance = dataset
        size = self.expected_size(file)
        chunk_size = 1024 * 1024
        pieces = target.parent / ("." + target.name + ".ranges")
        pieces.mkdir(parents=True, exist_ok=True)
        url = self.file_url(provenance["repo_id"], manifest["sha"], file["rfilename"])
        fallback_url = self.file_url(provenance["repo_id"], manifest["sha"], file["rfilename"], mirror=False)
        location = [None]
        location_lock = threading.Lock()

        def resolve():
            with location_lock:
                if location[0] is None:
                    with self.session().get(
                        url,
                        params={"download": "true", "vlds_request": str(time.time_ns())},
                        allow_redirects=False,
                        stream=True,
                        timeout=(20, 60),
                    ) as response:
                        if response.status_code >= 400 and fallback_url != url:
                            with self.session().get(
                                fallback_url,
                                params={"download": "true", "vlds_request": str(time.time_ns())},
                                allow_redirects=False,
                                stream=True,
                                timeout=(20, 60),
                            ) as fallback:
                                fallback.raise_for_status()
                                location[0] = fallback.headers.get("Location", fallback_url)
                        else:
                            response.raise_for_status()
                            location[0] = response.headers.get("Location", url)
                return location[0]

        # Reuse the complete prefix of the previous streaming download.
        partial = target.with_suffix(target.suffix + ".part")
        if partial.is_file():
            with partial.open("rb") as stream:
                for start in range(0, min(partial.stat().st_size, size), chunk_size):
                    end = min(start + chunk_size, size) - 1
                    block = stream.read(end - start + 1)
                    piece = pieces / f"{start:012d}-{end:012d}"
                    if len(block) == end - start + 1 and not piece.exists():
                        temporary = piece.with_suffix(".part")
                        temporary.write_bytes(block)
                        temporary.replace(piece)

        def fetch_piece(start):
            end = min(start + chunk_size, size) - 1
            expected = end - start + 1
            piece = pieces / f"{start:012d}-{end:012d}"
            if piece.is_file() and piece.stat().st_size == expected:
                return
            last_error = None
            for attempt in range(12):
                remote = None
                try:
                    remote = resolve()
                    with self.session().get(
                        remote,
                        headers={"Range": f"bytes={start}-{end}"},
                        stream=True,
                        timeout=(20, 90),
                    ) as response:
                        response.raise_for_status()
                        if response.status_code != 206 or response.headers.get("Content-Range") != (
                            f"bytes {start}-{end}/{size}"
                        ):
                            raise ValueError("Server returned an inconsistent download range")
                        temporary = piece.with_suffix(".part")
                        received = 0
                        with temporary.open("wb") as stream:
                            for block in response.iter_content(256 * 1024):
                                received += len(block)
                                if received > expected:
                                    raise ValueError("Server returned an oversized download range")
                                stream.write(block)
                                with self.lock:
                                    self.transferred_bytes += len(block)
                    if received != expected:
                        raise ValueError("Server returned a truncated download range")
                    temporary.replace(piece)
                    return
                except (requests.RequestException, ValueError, OSError) as error:
                    last_error = error
                    status = getattr(getattr(error, "response", None), "status_code", None)
                    if status in (401, 403) and remote is not None:
                        with location_lock:
                            if location[0] == remote:
                                location[0] = None
                    time.sleep(min(2 ** min(attempt, 5), 30))
            raise RuntimeError(f"Range {start}-{end} failed: {type(last_error).__name__}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.parts) as pool:
            list(pool.map(fetch_piece, range(0, size, chunk_size)))
        assembled = target.with_suffix(target.suffix + ".range-assembled")
        with assembled.open("wb") as output:
            for start in range(0, size, chunk_size):
                end = min(start + chunk_size, size) - 1
                with (pieces / f"{start:012d}-{end:012d}").open("rb") as stream:
                    shutil.copyfileobj(stream, output)
        if not self.verify(assembled, file):
            assembled.unlink()
            partial.unlink(missing_ok=True)
            shutil.rmtree(pieces)
            raise ValueError("Assembled ranges failed the dataset content hash check")
        assembled.replace(target)
        partial.unlink(missing_ok=True)
        shutil.rmtree(pieces)
        return self.record(name, file)

    def fetch(self, dataset, file):
        name, destination, manifest, provenance = dataset
        target = destination / self.relative_path(file["rfilename"])
        self.reuse_preview(destination, file, provenance["repo_id"])
        if self.verify(target, file):
            target.with_suffix(target.suffix + ".part").unlink(missing_ok=True)
            return self.record(name, file)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".part")
        size = self.expected_size(file)
        if self.parts > 1 and size >= 8 * 1024 * 1024 and file.get("lfs"):
            return self.fetch_ranges(dataset, file, target)
        last_error = None
        for attempt in range(12):
            try:
                if partial.is_file() and partial.stat().st_size == size:
                    if self.verify(partial, file):
                        partial.replace(target)
                        return self.record(name, file)
                    partial.unlink()
                if partial.is_file() and partial.stat().st_size > size:
                    partial.unlink()
                offset = partial.stat().st_size if partial.exists() else 0
                url = self.file_url(provenance["repo_id"], manifest["sha"], file["rfilename"], mirror=attempt < 2)
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                with self.session().get(
                    url,
                    params={"download": "true", "vlds_request": str(time.time_ns())},
                    headers=headers,
                    stream=True,
                    timeout=(20, 90),
                ) as response:
                    response.raise_for_status()
                    append = response.status_code == 206 and offset > 0
                    if response.status_code == 206:
                        expected_range = offset if append else 0
                        if not response.headers.get("Content-Range", "").startswith(f"bytes {expected_range}-"):
                            raise ValueError("Server returned an inconsistent resume range")
                    with partial.open("ab" if append else "wb") as stream:
                        for block in response.iter_content(1024 * 1024):
                            stream.write(block)
                            with self.lock:
                                self.transferred_bytes += len(block)
                if not self.verify(partial, file):
                    if partial.stat().st_size == size:
                        partial.unlink()
                    raise ValueError("Downloaded file failed its size or content hash check")
                partial.replace(target)
                return self.record(name, file)
            except (requests.RequestException, ValueError, OSError) as error:
                last_error = error
                delay = min(2 ** min(attempt, 5), 30)
                if attempt in {0, 5}:
                    status = getattr(getattr(error, "response", None), "status_code", None)
                    print(f"RETRY {name}/{file['rfilename']} attempt={attempt + 1} status={status}", flush=True)
                time.sleep(delay)
        raise RuntimeError(f"Failed to download {name}/{file['rfilename']}: {type(last_error).__name__}")

    def record(self, name, file):
        with self.lock:
            self.count += 1
            self.completed_bytes += self.expected_size(file)
        return name

    def run(self, names, *, manifests=None):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".download.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("Another dataset download is already running in this directory") from error
            return self._run(names, manifests=manifests)

    def _run(self, names, *, manifests=None):
        datasets = [self.prepare(name, manifest_file=(manifests or {}).get(name)) for name in names]
        total_size = sum(dataset[3]["bytes"] for dataset in datasets)
        if shutil.disk_usage(self.root).free < total_size * 1.05:
            raise ValueError(f"Not enough free disk space for {total_size} bytes")
        work = [(dataset, file) for dataset in datasets for file in dataset[2]["siblings"]]
        # Fetch metadata first, then complete the smaller datasets before the large snapshots.
        work.sort(
            key=lambda pair: (
                not pair[1]["rfilename"].startswith("meta/"),
                pair[0][3]["bytes"],
                self.expected_size(pair[1]),
            )
        )
        # Make one raw demonstration available early for checking recorded-state replay.
        sample = next(
            (item for item in work if item[0][0] == "libero_raw" and item[1]["rfilename"].endswith(".hdf5")), None
        )
        if sample is not None:
            work.remove(sample)
            work.insert(
                next((i for i, item in enumerate(work) if not item[1]["rfilename"].startswith("meta/")), 0), sample
            )
        total = len(work)
        status_path = self.root / "download_status.json"
        errors = []
        counts = dict.fromkeys(names, 0)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {
                pool.submit(self.fetch, dataset, file): (dataset[0], file["rfilename"]) for dataset, file in work
            }
            pending = set(futures)
            while pending:
                done, pending = concurrent.futures.wait(
                    pending, timeout=20, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done:
                    try:
                        counts[future.result()] += 1
                    except Exception as error:
                        errors.append({"dataset": futures[future][0], "file": futures[future][1], "error": str(error)})
                now = time.monotonic()
                status = {
                    "verified_files": self.count,
                    "total_files": total,
                    "verified_bytes": self.completed_bytes,
                    "total_bytes": total_size,
                    "elapsed_seconds": round(now - self.started),
                    "transferred_bytes": self.transferred_bytes,
                    "bytes_per_second": round(self.transferred_bytes / max(now - self.started, 1)),
                    "datasets": counts,
                    "errors": errors,
                    "updated_at": datetime.datetime.now(datetime.UTC).isoformat(),
                }
                self.write_json(status_path, status)
                for dataset in datasets:
                    name, destination, manifest, provenance = dataset
                    if counts[name] == len(manifest["siblings"]) and not provenance["complete"]:
                        provenance.update(complete=True, verified_at=status["updated_at"])
                        self.write_json(destination / "dataset_source.json", provenance)
                        print(f"COMPLETE {name}: {counts[name]} files, {provenance['bytes']} bytes", flush=True)
                if not done or self.count % 50 == 0 or not pending:
                    print(
                        f"PROGRESS {self.count}/{total} files; {self.completed_bytes / 10**9:.3f}/{total_size / 10**9:.3f} GB; errors={len(errors)}",
                        flush=True,
                    )
        if errors:
            raise RuntimeError(f"{len(errors)} files failed; rerun the same command to resume. See {status_path}")
        return datasets
