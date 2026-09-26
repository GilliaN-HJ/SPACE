from __future__ import annotations

from dataclasses import dataclass

import torch

from memory_jepa.losses.jepa_loss import jepa_loss
from memory_jepa.models.temporal_jepa import TemporalJEPAPredictor


@dataclass
class CounterfactualUtility:
    full_loss_per_horizon: torch.Tensor
    leave_one_out_loss_per_horizon: torch.Tensor
    utility_per_horizon: torch.Tensor
    utility_scalar: torch.Tensor
    oracle_evict_index: torch.Tensor


@torch.no_grad()
def exact_counterfactual_utility(
    predictor: TemporalJEPAPredictor,
    local: torch.Tensor,
    candidate_memory: torch.Tensor,
    candidate_valid: torch.Tensor,
    targets: torch.Tensor,
    target_valid: torch.Tensor,
    horizon_weights: torch.Tensor,
    loss_type: str,
    local_age: torch.Tensor | None = None,
    candidate_age: torch.Tensor | None = None,
) -> CounterfactualUtility:
    """Vectorized exact leave-one-slot-out utility using the same predictor."""
    batch, slots, d_in = candidate_memory.shape
    if slots != predictor.memory_slots:
        raise ValueError("Candidate slots must equal predictor physical memory slots")
    if candidate_age is None:
        candidate_age = torch.zeros((batch, slots), device=candidate_memory.device)
    if local_age is None:
        local_age = torch.arange(
            local.shape[1] - 1, -1, -1, device=local.device, dtype=torch.float32
        ).expand(batch, -1)

    full_prediction = predictor(
        local,
        candidate_memory,
        candidate_valid,
        local_age=local_age,
        memory_age=candidate_age,
    ).predictions
    full_loss = jepa_loss(
        full_prediction, targets, target_valid, horizon_weights, loss_type
    ).per_horizon

    expanded_memory = candidate_memory.unsqueeze(1).expand(batch, slots, slots, d_in)
    expanded_valid = candidate_valid.unsqueeze(1).expand(batch, slots, slots).clone()
    diagonal = torch.arange(slots, device=candidate_memory.device)
    expanded_valid[:, diagonal, diagonal] = False
    expanded_age = candidate_age.unsqueeze(1).expand(batch, slots, slots)

    flat_prediction = predictor(
        local.unsqueeze(1).expand(batch, slots, *local.shape[1:]).reshape(
            batch * slots, *local.shape[1:]
        ),
        expanded_memory.reshape(batch * slots, slots, d_in),
        expanded_valid.reshape(batch * slots, slots),
        local_age=local_age.unsqueeze(1).expand(batch, slots, local.shape[1]).reshape(
            batch * slots, local.shape[1]
        ),
        memory_age=expanded_age.reshape(batch * slots, slots),
    ).predictions
    expanded_target = targets.unsqueeze(1).expand(batch, slots, *targets.shape[1:]).reshape(
        batch * slots, *targets.shape[1:]
    )
    expanded_target_valid = target_valid.unsqueeze(1).expand(
        batch, slots, target_valid.shape[1]
    ).reshape(batch * slots, target_valid.shape[1])
    leave_one_out = jepa_loss(
        flat_prediction,
        expanded_target,
        expanded_target_valid,
        horizon_weights,
        loss_type,
    ).per_horizon.reshape(batch, slots, -1)
    utility = leave_one_out - full_loss.unsqueeze(1)
    scalar = (utility * horizon_weights.to(utility)).sum(dim=-1)
    scalar = scalar.masked_fill(~candidate_valid, torch.inf)
    oracle_index = scalar.argmin(dim=-1)
    return CounterfactualUtility(full_loss, leave_one_out, utility, scalar, oracle_index)


@torch.no_grad()
def loop_counterfactual_utility(
    predictor: TemporalJEPAPredictor,
    local: torch.Tensor,
    candidate_memory: torch.Tensor,
    candidate_valid: torch.Tensor,
    targets: torch.Tensor,
    target_valid: torch.Tensor,
    horizon_weights: torch.Tensor,
    loss_type: str,
    local_age: torch.Tensor | None = None,
    candidate_age: torch.Tensor | None = None,
) -> CounterfactualUtility:
    batch, slots, _ = candidate_memory.shape
    if local_age is None:
        local_age = torch.arange(
            local.shape[1] - 1, -1, -1, device=local.device, dtype=torch.float32
        ).expand(batch, -1)
    if candidate_age is None:
        candidate_age = torch.zeros((batch, slots), device=local.device)
    full = predictor(
        local, candidate_memory, candidate_valid, local_age, candidate_age
    ).predictions
    full_loss = jepa_loss(full, targets, target_valid, horizon_weights, loss_type).per_horizon
    losses = []
    for index in range(slots):
        valid = candidate_valid.clone()
        valid[:, index] = False
        prediction = predictor(local, candidate_memory, valid, local_age, candidate_age).predictions
        losses.append(
            jepa_loss(prediction, targets, target_valid, horizon_weights, loss_type).per_horizon
        )
    leave_one_out = torch.stack(losses, dim=1)
    utility = leave_one_out - full_loss.unsqueeze(1)
    scalar = (utility * horizon_weights.to(utility)).sum(dim=-1)
    scalar = scalar.masked_fill(~candidate_valid, torch.inf)
    return CounterfactualUtility(
        full_loss, leave_one_out, utility, scalar, scalar.argmin(dim=-1)
    )

