from __future__ import annotations

import copy
from typing import Any

import torch

from memory_jepa.models.temporal_jepa import TemporalJEPAPredictor
from memory_jepa.models.utility_estimator import CausalUtilityEstimator


def build_predictor(config: dict[str, Any], d_in: int) -> TemporalJEPAPredictor:
    model = config["predictor"]
    return TemporalJEPAPredictor(
        d_in=d_in,
        d_model=int(model["d_model"]),
        n_layers=int(model["n_layers"]),
        n_heads=int(model["n_heads"]),
        d_ff=int(model["d_ff"]),
        dropout=float(model["dropout"]),
        memory_slots=int(model["memory_budget"]) + 1,
        local_context_len=int(model["local_context_len"]),
        horizons=model["horizons"],
    )


def build_utility_estimator(
    config: dict[str, Any], predictor: TemporalJEPAPredictor
) -> CausalUtilityEstimator:
    utility = config["utility_estimator"]
    return CausalUtilityEstimator(
        d_model=predictor.d_model,
        d_utility=int(utility["d_utility"]),
        n_layers=int(utility["n_layers"]),
        n_heads=int(utility["n_heads"]),
        d_ff=int(utility["d_ff"]),
        dropout=float(utility["dropout"]),
        num_horizons=predictor.num_horizons,
        horizon_weights=config["predictor"]["horizon_weights"],
        architecture=str(utility.get("architecture", "pooled_set")),
    )


def load_predictor_checkpoint(
    path: str,
    device: torch.device | str = "cpu",
    *,
    memory_budget: int | None = None,
) -> tuple[TemporalJEPAPredictor, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if memory_budget is not None:
        if int(memory_budget) <= 0:
            raise ValueError("memory_budget must be positive")
        config = copy.deepcopy(config)
        config["predictor"]["memory_budget"] = int(memory_budget)
    predictor = build_predictor(config, int(checkpoint["d_in"]))
    predictor.load_state_dict(checkpoint["model"])
    predictor.to(device)
    return predictor, checkpoint


def load_utility_checkpoint(
    path: str,
    predictor: TemporalJEPAPredictor,
    device: torch.device | str = "cpu",
) -> tuple[CausalUtilityEstimator, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    estimator = build_utility_estimator(checkpoint["config"], predictor)
    estimator.load_state_dict(checkpoint["model"])
    estimator.to(device)
    return estimator, checkpoint
