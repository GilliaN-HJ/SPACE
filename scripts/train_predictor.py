#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from memory_jepa.data.latent_dataset import PredictorTrainingDataset, collate_predictor
from memory_jepa.losses import jepa_loss
from memory_jepa.utils import apply_overrides, load_config, seed_everything
from memory_jepa.utils.hashing import canonical_hash, state_dict_hash
from memory_jepa.utils.io import atomic_torch_save, write_json
from memory_jepa.utils.modeling import build_predictor
from memory_jepa.utils.parameter_count import count_parameters, expected_predictor_parameters
from memory_jepa.utils.training import autocast_context, cosine_warmup_scheduler


@torch.no_grad()
def evaluate(
    model,
    loader,
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, Any]:
    model.eval()
    horizon_weights = torch.tensor(config["predictor"]["horizon_weights"], device=device)
    full_losses: list[torch.Tensor] = []
    local_losses: list[torch.Tensor] = []
    for batch in loader:
        batch = batch.to(device)
        with autocast_context(device, config["training"]["precision"]):
            prediction = model(
                batch.local,
                batch.memory,
                batch.memory_valid,
                batch.local_age,
                batch.memory_age,
            ).predictions
            full = jepa_loss(
                prediction,
                batch.targets,
                batch.target_valid,
                horizon_weights,
                config["predictor"]["target_loss"],
            )
            no_memory_prediction = model(
                batch.local,
                batch.memory,
                torch.zeros_like(batch.memory_valid),
                batch.local_age,
                batch.memory_age,
            ).predictions
            local = jepa_loss(
                no_memory_prediction,
                batch.targets,
                batch.target_valid,
                horizon_weights,
                config["predictor"]["target_loss"],
            )
        full_losses.append(full.per_horizon.float().cpu())
        local_losses.append(local.per_horizon.float().cpu())
    full_tensor = torch.cat(full_losses)
    local_tensor = torch.cat(local_losses)
    result: dict[str, Any] = {
        "loss": float(full_tensor.mean()),
        "local_only_loss": float(local_tensor.mean()),
        "relative_memory_gain": float(
            (local_tensor.mean() - full_tensor.mean()) / local_tensor.mean().clamp_min(1e-8)
        ),
    }
    for index, horizon in enumerate(config["predictor"]["horizons"]):
        result[f"loss_h{horizon}"] = float(full_tensor[:, index].mean())
        result[f"local_only_loss_h{horizon}"] = float(local_tensor[:, index].mean())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the frozen-latent Temporal JEPA predictor")
    parser.add_argument("--config", default="configs/assembly101.yaml")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=None)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    config = apply_overrides(load_config(args.config), args.set)
    seed_everything(args.seed)
    device = torch.device(args.device)
    manifest = args.manifest or str(Path(config["data"]["root"]) / "manifest.json")
    output_dir = Path(args.output or config["project"]["output_dir"]) / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    d_in = int(json.loads(Path(manifest).read_text(encoding="utf-8"))["d_in"])
    model = build_predictor(config, d_in).to(device)

    actual_parameters = count_parameters(model)
    expected_parameters = expected_predictor_parameters(
        d_in=d_in,
        d_model=int(config["predictor"]["d_model"]),
        d_ff=int(config["predictor"]["d_ff"]),
        n_layers=int(config["predictor"]["n_layers"]),
        horizons=len(config["predictor"]["horizons"]),
    )
    if actual_parameters != expected_parameters:
        raise AssertionError(f"Predictor parameter mismatch: {actual_parameters} != {expected_parameters}")

    dataset_arguments = dict(
        manifest_path=manifest,
        local_context_len=int(config["predictor"]["local_context_len"]),
        memory_slots=int(config["predictor"]["memory_budget"]) + 1,
        horizons=config["predictor"]["horizons"],
        memory_dropout=float(config["predictor"]["memory_dropout"]),
        seed=args.seed,
    )
    train_dataset = PredictorTrainingDataset(split="train", **dataset_arguments)
    val_dataset = PredictorTrainingDataset(split="val", **dataset_arguments)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["training"]["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=int(config["training"]["num_workers"]) > 0,
        collate_fn=collate_predictor,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["training"]["batch_size"]) * 2,
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=int(config["training"]["num_workers"]) > 0,
        collate_fn=collate_predictor,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["predictor_lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    epochs = int(config["training"]["predictor_epochs"])
    scheduler = cosine_warmup_scheduler(
        optimizer, epochs * len(train_loader), float(config["training"]["warmup_ratio"])
    )
    horizon_weights = torch.tensor(config["predictor"]["horizon_weights"], device=device)
    history: list[dict[str, Any]] = []
    best_validation = float("inf")
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running = torch.zeros(len(config["predictor"]["horizons"]), device=device)
        seen = 0
        progress = tqdm(train_loader, desc=f"predictor {epoch}/{epochs}", leave=False)
        for batch in progress:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, config["training"]["precision"]):
                prediction = model(
                    batch.local,
                    batch.memory,
                    batch.memory_valid,
                    batch.local_age,
                    batch.memory_age,
                ).predictions
                losses = jepa_loss(
                    prediction,
                    batch.targets,
                    batch.target_valid,
                    horizon_weights,
                    config["predictor"]["target_loss"],
                )
            losses.mean.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["training"]["grad_clip_norm"])
            )
            optimizer.step()
            scheduler.step()
            batch_size = batch.local.shape[0]
            running += losses.per_horizon.detach().sum(dim=0)
            seen += batch_size
            progress.set_postfix(loss=f"{losses.mean.item():.4f}")
        validation = evaluate(model, val_loader, device, config)
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": float(running.sum() / (seen * len(running))),
            "lr": optimizer.param_groups[0]["lr"],
            **{f"val_{key}": value for key, value in validation.items()},
        }
        for index, horizon in enumerate(config["predictor"]["horizons"]):
            row[f"train_loss_h{horizon}"] = float(running[index] / seen)
        history.append(row)
        print(json.dumps(row, sort_keys=True))
        if validation["loss"] < best_validation:
            best_validation = validation["loss"]
            checkpoint = {
                "format_version": 1,
                "stage": "predictor",
                "seed": args.seed,
                "d_in": d_in,
                "config": config,
                "config_hash": canonical_hash(config),
                "manifest": str(Path(manifest).resolve()),
                "model": model.state_dict(),
                "predictor_hash": state_dict_hash(model),
                "trainable_parameters": actual_parameters,
                "validation": validation,
                "elapsed_sec": time.time() - started,
            }
            atomic_torch_save(checkpoint, output_dir / "predictor_best.pt")
        write_json(history, output_dir / "predictor_history.json")
    print(f"Best checkpoint: {output_dir / 'predictor_best.pt'}")


if __name__ == "__main__":
    main()
