"""Train and validate the directly executable VLDS smoke model.

Example:

    env -u PYTHONPATH .venv/bin/python -m openpi.training.vlds_train \
        --steps 2000 --output data/experiments/vlds_smoke_20261123
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from openpi.models.vlds import StructuredVLDSModel
from openpi.models.vlds import vlds_loss
from openpi.models.vlds_config import VLDSConfig
from openpi.training.vlds_dataset import SyntheticVLDSDataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=Path("data/experiments/vlds_smoke_20261123"))
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu" if name == "auto" else name)


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _evaluate(
    model: StructuredVLDSModel, loader: DataLoader, config: VLDSConfig, device: torch.device
) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for raw_batch in loader:
            batch = _move(raw_batch, device)
            prediction = model(batch["state_history"], batch["execution_context"], batch["language"])
            _, metrics = vlds_loss(prediction, batch, config)
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + value * len(batch["state_history"])
            count += len(batch["state_history"])
    return {key: value / count for key, value in sums.items()}


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.steps < 1 or args.samples < 1 or args.batch_size < 1:
        raise ValueError("steps, samples and batch-size must be positive")
    torch.manual_seed(args.seed)
    device = _device(args.device)
    config = VLDSConfig()
    dataset = SyntheticVLDSDataset(args.samples, config, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    evaluation_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model = StructuredVLDSModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-2)
    iterator = iter(loader)
    history: list[dict[str, float]] = []
    started = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = _move(batch, device)
        prediction = model(batch["state_history"], batch["execution_context"], batch["language"])
        loss, metrics = vlds_loss(prediction, batch, config)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            record = {"step": float(step), **metrics, "seconds": time.time() - started}
            history.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)

    validation = _evaluate(model, evaluation_loader, config, device)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "config": dataclasses.asdict(config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": args.steps,
        "seed": args.seed,
    }
    torch.save(checkpoint, args.output / "checkpoint.pt")
    result = {
        "schema": "vlds.smoke_training.v1",
        "device": str(device),
        "steps": args.steps,
        "samples": args.samples,
        "batch_size": args.batch_size,
        "config": dataclasses.asdict(config),
        "history": history,
        "validation": validation,
        "elapsed_seconds": time.time() - started,
        "checkpoint": str(args.output / "checkpoint.pt"),
        "dataset_scope": "synthetic_causal_structured_smoke_only",
    }
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"validation": validation, "checkpoint": result["checkpoint"]}, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
