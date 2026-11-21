"""Open the desktop scene selector for interactive model validation."""

import argparse
import json

from openpi.simulation import catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog", action="store_true", help="List installed environments and task types without opening a GUI."
    )
    parser.add_argument("--dataset", help="Open a local demonstration dataset directly in the dataset browser.")
    parser.add_argument(
        "--replay", action="store_true", help="Automatically open recorded 3D replay for selected episodes."
    )
    parser.add_argument("--watch", action="store_true", help="Reload the current dataset when its files change.")
    args = parser.parse_args()
    if args.catalog:
        print(json.dumps(catalog.build_catalog(), indent=2, ensure_ascii=False))
        return
    from openpi.simulation.ui import inspector

    inspector.main(dataset_path=args.dataset, replay=args.replay, watch=args.watch)


if __name__ == "__main__":
    main()
