"""Shared, label-free primitives for training SPACE artifacts."""

from __future__ import annotations

from typing import Any

import torch

from memory_jepa.losses import jepa_loss
from memory_jepa.memory.policies.counter_based import (
    CounterBasedReservoirSamplingPolicy,
    counter_uniform,
    stable_video_seed,
)
from memory_jepa.memory.state import MemoryState
from memory_jepa.models.predictive_selector import build_predictive_action_features
from memory_jepa.utils.training import autocast_context


ROLLOUTS = (8, 32)
REFERENCE_STREAM = 100


def insert_first_free(
    state: MemoryState, event: torch.Tensor, timestamp: int
) -> MemoryState:
    output = state.clone()
    slot = int((~output.valid).nonzero(as_tuple=False)[0].item())
    output.values[slot] = event
    output.timestamps[slot] = timestamp
    output.event_ids[slot] = timestamp
    output.valid[slot] = True
    output.access[slot] = 0
    return output


def replace_compact(
    state: MemoryState, event: torch.Tensor, timestamp: int, action: int
) -> MemoryState:
    active = state.valid.nonzero(as_tuple=False).squeeze(-1)
    values = torch.cat((state.values[active], event.unsqueeze(0)))
    timestamps = torch.cat((
        state.timestamps[active], torch.tensor([timestamp], device=event.device)
    ))
    event_ids = torch.cat((
        state.event_ids[active], torch.tensor([timestamp], device=event.device)
    ))
    keep = torch.ones(state.capacity + 1, dtype=torch.bool, device=event.device)
    keep[int(action)] = False
    return MemoryState(
        values=values[keep].clone(),
        timestamps=timestamps[keep].clone(),
        event_ids=event_ids[keep].clone(),
        valid=torch.ones(state.capacity, dtype=torch.bool, device=event.device),
        access=torch.zeros(state.capacity, dtype=torch.float32, device=event.device),
    )


def candidate_tensors(
    state: MemoryState, event: torch.Tensor, timestamp: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if state.active_count != state.capacity:
        raise ValueError("candidate actions require full memory")
    return (
        torch.cat((state.values, event.unsqueeze(0))),
        torch.cat((state.timestamps, torch.tensor([timestamp], device=event.device))),
        torch.cat((state.event_ids, torch.tensor([timestamp], device=event.device))),
    )


def online_capacity(predictor: Any) -> int:
    capacity = int(predictor.memory_slots) - 1
    if capacity <= 0:
        raise ValueError("predictor must expose at least one online memory slot")
    return capacity


def reservoir_action(
    values: torch.Tensor,
    timestamps: torch.Tensor,
    event_ids: torch.Tensor,
    local: torch.Tensor,
    current_time: int,
    *,
    capacity: int,
    video_id: str,
    policy_seed: int,
) -> int:
    sampler = CounterBasedReservoirSamplingPolicy(
        capacity, stable_video_seed(video_id, policy_seed)
    )
    return int(sampler.select_evict_index(
        values.unsqueeze(0),
        timestamps.unsqueeze(0),
        torch.ones(1, capacity + 1, dtype=torch.bool, device=values.device),
        local.unsqueeze(0),
        torch.tensor([current_time], device=values.device),
        candidate_event_ids=event_ids.unsqueeze(0),
    ).item())


@torch.no_grad()
def predict_state(
    predictor: Any,
    state: MemoryState,
    local: torch.Tensor,
    current_time: int,
    precision: str,
) -> torch.Tensor:
    memory, valid, age = state.padded(predictor.memory_slots, current_time)
    local_age = torch.arange(
        local.shape[0] - 1, -1, -1, dtype=torch.float32, device=local.device
    )
    with autocast_context(local.device, precision):
        return predictor(
            local.unsqueeze(0), memory.unsqueeze(0), valid.unsqueeze(0),
            local_age.unsqueeze(0), age.unsqueeze(0),
        ).predictions.squeeze(0).float()


@torch.no_grad()
def online_features(
    predictor: Any,
    candidate: torch.Tensor,
    age: torch.Tensor,
    local: torch.Tensor,
    device: torch.device,
    precision: str,
) -> torch.Tensor:
    proposals = candidate.shape[0]
    indices = torch.arange(proposals, device=device)
    keep = indices.view(proposals, 1) != indices.view(1, proposals)
    keep_indices = indices.view(1, proposals).expand(proposals, -1)[keep].reshape(
        proposals, proposals - 1
    )
    expanded = candidate.unsqueeze(0).expand(proposals, -1, -1)
    memory = expanded.gather(
        1, keep_indices.unsqueeze(-1).expand(-1, -1, candidate.shape[-1])
    )
    memory = torch.cat((
        memory,
        torch.zeros(proposals, 1, candidate.shape[-1], device=device),
    ), dim=1)
    ages = age.unsqueeze(0).expand(proposals, -1).gather(1, keep_indices)
    memory_age = torch.cat((ages, torch.zeros(proposals, 1, device=device)), dim=1)
    local_batch = local.unsqueeze(0).expand(proposals, -1, -1)
    valid = torch.ones(
        proposals, predictor.memory_slots, dtype=torch.bool, device=device
    )
    local_age = torch.arange(
        local.shape[0] - 1, -1, -1, dtype=torch.float32, device=device
    ).expand(proposals, -1)
    with autocast_context(device, precision):
        predictions = predictor(
            local_batch, memory, valid, local_age, memory_age
        ).predictions
    return build_predictive_action_features(
        predictor,
        predictions.unsqueeze(0),
        candidate.unsqueeze(0),
        local.unsqueeze(0),
        age.unsqueeze(0),
    ).float().squeeze(0)


@torch.no_grad()
def proposal_rollout_losses(
    predictor: Any,
    latents: torch.Tensor,
    current_time: int,
    candidate: torch.Tensor,
    candidate_timestamps: torch.Tensor,
    proposal_indices: torch.Tensor,
    horizons: tuple[int, ...],
    horizon_weights: torch.Tensor,
    loss_type: str,
    local_context_len: int,
    video_seed: int,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    query_count, proposal_count = proposal_indices.shape
    branch_count = query_count * proposal_count
    budget = candidate.shape[0] - 1
    flat_indices = proposal_indices.reshape(-1)
    expanded_values = candidate.unsqueeze(0).expand(branch_count, -1, -1)
    expanded_timestamps = candidate_timestamps.unsqueeze(0).expand(branch_count, -1)
    keep = torch.arange(
        candidate.shape[0], device=candidate.device
    ).unsqueeze(0) != flat_indices.unsqueeze(1)
    branch_values = expanded_values[keep].reshape(
        branch_count, budget, candidate.shape[-1]
    )
    branch_timestamps = expanded_timestamps[keep].reshape(branch_count, budget)
    cumulative = torch.zeros(
        branch_count, len(horizons), device=candidate.device, dtype=torch.float32
    )
    scalar_results: dict[int, torch.Tensor] = {}
    horizon_results: dict[int, torch.Tensor] = {}
    if predictor.memory_slots != budget + 1:
        raise ValueError("predictor physical slots do not match rollout budget")

    for offset in range(max(ROLLOUTS)):
        rollout_time = current_time + offset
        if offset:
            event_time = rollout_time - local_context_len
            new_event = latents[event_time]
            candidate_values = torch.cat((
                branch_values,
                new_event.view(1, 1, -1).expand(branch_count, 1, -1),
            ), dim=1)
            candidate_times = torch.cat((
                branch_timestamps,
                torch.full(
                    (branch_count, 1), event_time, device=candidate.device,
                    dtype=branch_timestamps.dtype,
                ),
            ), dim=1)
            draw = counter_uniform(video_seed, rollout_time, REFERENCE_STREAM)
            evict_index = min(int(draw * (budget + 1)), budget)
            keep_index = (
                torch.arange(budget + 1, device=candidate.device) != evict_index
            )
            branch_values = candidate_values[:, keep_index]
            branch_timestamps = candidate_times[:, keep_index]

        local = latents[
            rollout_time - local_context_len + 1 : rollout_time + 1
        ]
        local_age = torch.arange(
            local_context_len - 1, -1, -1,
            device=candidate.device, dtype=torch.float32,
        )
        memory = torch.cat((
            branch_values,
            torch.zeros(
                branch_count, 1, candidate.shape[-1],
                device=candidate.device, dtype=candidate.dtype,
            ),
        ), dim=1)
        memory_valid = torch.zeros(
            branch_count, predictor.memory_slots,
            device=candidate.device, dtype=torch.bool,
        )
        memory_valid[:, :budget] = True
        memory_age = torch.zeros(
            branch_count, predictor.memory_slots,
            device=candidate.device, dtype=torch.float32,
        )
        memory_age[:, :budget] = rollout_time - branch_timestamps
        target = torch.stack([latents[rollout_time + h] for h in horizons])
        prediction = predictor(
            local.unsqueeze(0).expand(branch_count, -1, -1),
            memory,
            memory_valid,
            local_age.unsqueeze(0).expand(branch_count, -1),
            memory_age,
        ).predictions
        loss = jepa_loss(
            prediction,
            target.unsqueeze(0).expand(branch_count, -1, -1),
            torch.ones(
                branch_count, len(horizons),
                device=candidate.device, dtype=torch.bool,
            ),
            horizon_weights,
            loss_type,
        )
        cumulative += loss.per_horizon.float()
        steps = offset + 1
        if steps in ROLLOUTS:
            per_horizon = (cumulative / steps).reshape(
                query_count, proposal_count, len(horizons)
            )
            horizon_results[steps] = per_horizon
            scalar_results[steps] = (
                per_horizon * horizon_weights.to(per_horizon)
            ).sum(dim=-1)
    return scalar_results, horizon_results
