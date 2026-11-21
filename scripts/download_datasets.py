"""Fetch complete, versioned datasets for local browsing, training and evaluation."""

import argparse
from pathlib import Path

from openpi.datasets.download import DatasetDownloader
from openpi.datasets.sources import SOURCES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=SOURCES, default=list(SOURCES))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "data/datasets")
    parser.add_argument(
        "--endpoint", default="https://huggingface.co", choices=("https://huggingface.co", "https://hf-mirror.com")
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--parts", type=int, default=1, help="Parallel byte ranges per large file")
    parser.add_argument("--manifest-dir", type=Path, help="Reuse previously fetched API manifests named <dataset>.json")
    parser.add_argument("--libero-mirror", choices=("modelscope",), help="Optional byte-verified LIBERO mirror")
    args = parser.parse_args()
    if args.workers < 1 or args.parts < 1 or len(set(args.datasets)) != len(args.datasets):
        parser.error("workers and parts must be positive and datasets must be unique")
    manifests = {name: args.manifest_dir / f"{name}.json" for name in args.datasets} if args.manifest_dir else None
    DatasetDownloader(
        args.root,
        endpoint=args.endpoint,
        workers=args.workers,
        parts=args.parts,
        libero_mirror=bool(args.libero_mirror),
    ).run(args.datasets, manifests=manifests)


if __name__ == "__main__":
    main()
