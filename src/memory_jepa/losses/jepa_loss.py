from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional


@dataclass
class JEPALossOutput:
    per_horizon: torch.Tensor
    scalar: torch.Tensor
    mean: torch.Tensor


def latent_distance(
    prediction: torch.Tensor, target: torch.Tensor, loss_type: str = "cosine"
) -> torch.Tensor:
    target = target.detach()
    if loss_type == "cosine":
        prediction = functional.normalize(prediction, dim=-1)
        target = functional.normalize(target, dim=-1)
        return 1.0 - (prediction * target).sum(dim=-1)
    if loss_type == "l1":
        scale = target.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return ((prediction - target).abs() / scale).mean(dim=-1)
    if loss_type == "l2":
        scale = target.square().mean(dim=-1).clamp_min(1e-6)
        return (prediction - target).square().mean(dim=-1) / scale
    raise ValueError(f"Unknown latent loss: {loss_type}")


def jepa_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_valid: torch.Tensor,
    horizon_weights: torch.Tensor,
    loss_type: str = "cosine",
) -> JEPALossOutput:
    per_horizon = latent_distance(prediction, target, loss_type)
    valid = target_valid.to(per_horizon.dtype)
    weights = horizon_weights.to(per_horizon.device, per_horizon.dtype)
    effective = valid * weights
    scalar = (per_horizon * effective).sum(dim=-1) / effective.sum(dim=-1).clamp_min(1e-8)
    mean = scalar.mean()
    return JEPALossOutput(per_horizon=per_horizon, scalar=scalar, mean=mean)

