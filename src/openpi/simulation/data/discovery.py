"""Discover shared local datasets and prefer verified full copies over previews."""

import contextlib
import json
from pathlib import Path

from openpi.simulation.data.descriptions import PROFILES
from openpi.simulation.data.descriptions import dataset_source


def provenance(path):
    marker = path / "dataset_source.json"
    if marker.is_file():
        with contextlib.suppress(OSError, ValueError):
            return json.loads(marker.read_text())
    return {}


def discover_datasets(roots, extra=()):
    found = {Path(path).resolve() for path in extra}
    for root in roots:
        if not root.is_dir():
            continue
        for pattern in ("*/meta/info.json", "*/*/meta/info.json"):
            found.update(path.parent.parent.resolve() for path in root.glob(pattern))
        for pattern in ("*/dataset_source.json", "*/*/dataset_source.json"):
            for marker in root.glob(pattern):
                if (
                    provenance(marker.parent).get("format") == "libero_hdf5"
                    and next(marker.parent.rglob("*.hdf5"), None) is not None
                ):
                    found.add(marker.parent.resolve())
    selected, custom = {}, []
    for path in sorted(found):
        source = dataset_source(path)
        if source not in PROFILES:
            custom.append(path)
            continue
        marker = provenance(path)
        # Keep an existing playable preview until its complete replacement is verified.
        rank = 2 if marker.get("complete") else (0 if marker.get("mode") == "full_snapshot" else 1)
        previous = selected.get(source)
        if previous is None or rank > previous[0]:
            selected[source] = rank, path
    return custom + [path for _rank, path in selected.values()]
