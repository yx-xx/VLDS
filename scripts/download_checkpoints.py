import argparse
import base64
import dataclasses
import hashlib
import json
import logging
import os
import pathlib
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

import filelock
import google_crc32c

from openpi.shared import download

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class CheckpointObject:
    name: str
    relative_path: pathlib.PurePosixPath
    size: int
    generation: str
    md5: str | None
    crc32c: str

    @property
    def url(self) -> str:
        name = urllib.parse.quote(self.name, safe="/")
        return f"https://storage.googleapis.com/openpi-assets/{name}?generation={self.generation}"

    @property
    def urls(self) -> tuple[str, str]:
        return (
            self.url,
            self.url.replace(
                "https://storage.googleapis.com/openpi-assets/", "https://openpi-assets.storage.googleapis.com/", 1
            ),
        )


def checkpoint_objects(url: str) -> tuple[pathlib.PurePosixPath, list[CheckpointObject]]:
    parsed = urllib.parse.urlsplit(url)
    checkpoint = pathlib.PurePosixPath(parsed.path.lstrip("/"))
    if (
        parsed.scheme != "gs"
        or parsed.netloc != "openpi-assets"
        or parsed.query
        or parsed.fragment
        or ".." in checkpoint.parts
        or len(checkpoint.parts) < 2
        or checkpoint.parts[0] != "checkpoints"
    ):
        raise ValueError("Expected gs://openpi-assets/checkpoints/<checkpoint-name>.")
    prefix = str(checkpoint) + "/"
    parameters = {
        "prefix": prefix,
        "fields": "items(name,size,generation,md5Hash,crc32c),nextPageToken",
        "maxResults": "1000",
    }
    objects = []
    while True:
        api_url = "https://storage.googleapis.com/storage/v1/b/openpi-assets/o?" + urllib.parse.urlencode(parameters)
        with urllib.request.urlopen(api_url, timeout=30) as response:
            page = json.load(response)
        for item in page.get("items", []):
            name = item["name"]
            if name.endswith("/"):
                continue
            relative_path = pathlib.PurePosixPath(name.removeprefix(prefix))
            if not name.startswith(prefix) or ".." in relative_path.parts or relative_path.is_absolute():
                raise ValueError(f"Unsafe checkpoint object path: {name!r}")
            if any(character in name for character in "\r\n"):
                raise ValueError(f"Unsupported checkpoint object path: {name!r}")
            digest = base64.b64decode(item["md5Hash"], validate=True) if "md5Hash" in item else None
            crc32c = base64.b64decode(item["crc32c"], validate=True)
            if (
                (digest is not None and len(digest) != 16)
                or len(crc32c) != 4
                or int(item["size"]) < 0
                or not str(item["generation"]).isdigit()
            ):
                raise ValueError(f"Invalid checkpoint object metadata: {name!r}")
            objects.append(
                CheckpointObject(
                    name,
                    relative_path,
                    int(item["size"]),
                    str(item["generation"]),
                    digest.hex() if digest is not None else None,
                    crc32c.hex(),
                )
            )
        if "nextPageToken" not in page:
            break
        parameters["pageToken"] = page["nextPageToken"]
    if not objects:
        raise FileNotFoundError(f"No checkpoint objects found at {url}")
    return checkpoint, objects


def verify_objects(directory: pathlib.Path, objects: list[CheckpointObject]) -> None:
    for item in objects:
        path = directory / item.relative_path
        if not path.is_file() or path.stat().st_size != item.size:
            raise OSError(f"Missing or incomplete checkpoint file: {path}")
        digest = hashlib.md5(usedforsecurity=False)
        crc32c = google_crc32c.Checksum()
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
                crc32c.update(chunk)
        if crc32c.digest().hex() != item.crc32c or (item.md5 is not None and digest.hexdigest() != item.md5):
            raise OSError(f"Checkpoint checksum mismatch: {path}")


def write_input(path: pathlib.Path, destination: pathlib.Path, objects: list[CheckpointObject]) -> None:
    lines = []
    for item in objects:
        target = destination / item.relative_path
        lines.extend(["\t".join(item.urls), f"  dir={target.parent}", f"  out={target.name}"])
        if item.md5 is not None:
            lines.append(f"  checksum=md5={item.md5}")
    path.write_text("\n".join(lines) + "\n")


def fetch_checkpoint(url: str, aria2c: pathlib.Path, connections: int, concurrent_downloads: int) -> pathlib.Path:
    checkpoint, objects = checkpoint_objects(url)
    destination = download.get_cache_dir() / "openpi-assets" / checkpoint
    destination.parent.mkdir(parents=True, exist_ok=True)
    with filelock.FileLock(destination.with_suffix(".lock")):
        if destination.exists():
            LOGGER.info("Verifying existing checkpoint: %s", destination)
            verify_objects(destination, objects)
            return destination
        staging = destination.with_suffix(".partial")
        staging.mkdir(parents=True, exist_ok=True)
        manifest = {
            item.name: {"size": item.size, "generation": item.generation, "md5": item.md5, "crc32c": item.crc32c}
            for item in objects
        }
        manifest_path = staging / ".download_manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError(f"Upstream checkpoint changed; move aside {staging} before restarting.")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        input_path = destination.with_suffix(".aria2.txt")
        write_input(input_path, staging, objects)
        LOGGER.info(
            "Downloading %s: %.2f GiB. Partial files: %s", url, sum(item.size for item in objects) / 1024**3, staging
        )
        env = os.environ.copy()
        library_dir = aria2c.parent.parent / "lib/x86_64-linux-gnu"
        if library_dir.is_dir():
            env["LD_LIBRARY_PATH"] = str(library_dir) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        command = [
            str(aria2c),
            f"--input-file={input_path}",
            "--continue=true",
            "--check-integrity=true",
            "--allow-overwrite=true",
            "--auto-file-renaming=false",
            "--file-allocation=none",
            "--auto-save-interval=10",
            "--max-tries=0",
            "--retry-wait=10",
            "--connect-timeout=15",
            "--timeout=30",
            "--summary-interval=30",
            "--enable-color=false",
            f"--max-connection-per-server={connections}",
            f"--split={connections * len(objects[0].urls)}",
            "--min-split-size=16M",
            f"--max-concurrent-downloads={concurrent_downloads}",
        ]
        process = subprocess.Popen(command, env=env)
        try:
            return_code = process.wait()
            if return_code:
                raise subprocess.CalledProcessError(return_code, command)
        except BaseException:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
        LOGGER.info("Verifying completed checkpoint: %s", url)
        verify_objects(staging, objects)
        staging.rename(destination)
        input_path.unlink()
        LOGGER.info("Checkpoint ready: %s", destination)
        return destination


def interrupt_download(signum, frame) -> None:
    raise KeyboardInterrupt


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume and verify public openpi checkpoints using aria2.")
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--aria2c", type=pathlib.Path)
    parser.add_argument("--connections", type=int, default=8)
    parser.add_argument("--concurrent-downloads", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.connections <= 16 or args.concurrent_downloads < 1:
        parser.error("Connections must be between 1 and 16; concurrent downloads must be positive.")
    aria2c = args.aria2c or pathlib.Path(shutil.which("aria2c") or ROOT / ".cache/download-tools/usr/bin/aria2c")
    if not aria2c.is_file():
        parser.error("aria2c is missing; install aria2 or specify --aria2c /path/to/aria2c.")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    signal.signal(signal.SIGTERM, interrupt_download)
    for checkpoint in args.checkpoints:
        for attempt in range(5):
            try:
                fetch_checkpoint(checkpoint, aria2c.resolve(), args.connections, args.concurrent_downloads)
                break
            except (TimeoutError, urllib.error.URLError):
                if attempt == 4:
                    raise
                LOGGER.exception("Checkpoint metadata request failed; retrying")
                time.sleep(min(2**attempt, 30))


if __name__ == "__main__":
    main()
