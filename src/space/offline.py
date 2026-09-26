from __future__ import annotations

import torch
import torch.nn.functional as functional

from .memory import CandidateSet, counter_uniform
from .model import TemporalJEPAPredictor, predict_eviction_candidates


def anchor_is_valid(
    anchor: int,
    video_length: int,
    horizons: tuple[int, ...],
    *,
    rollout_steps: int = 1,
) -> bool:
    """Return whether every prediction target remains inside the video."""
    return int(anchor) + int(rollout_steps) - 1 + max(horizons) < int(video_length)


@torch.no_grad()
def one_step_candidate_costs(
    predictor: TemporalJEPAPredictor,
    candidates: CandidateSet,
    *,
    local_context: torch.Tensor,
    current_time: int,
    future_targets: torch.Tensor,
    horizon_weights: torch.Tensor,
) -> torch.Tensor:
    """Evaluate Eq. (3) for every candidate using observed train-only futures."""
    _, memory_ages = candidates.ages(current_time)
    predictions = predict_eviction_candidates(
        predictor, candidates.memories, memory_ages, local_context
    )
    if future_targets.shape != predictions.shape[1:]:
        raise ValueError("future targets must have shape [horizons,d_in]")
    target = future_targets.unsqueeze(0).expand_as(predictions)
    per_horizon = 1.0 - functional.cosine_similarity(
        predictions.float(), target.float(), dim=-1
    )
    return (per_horizon * horizon_weights.to(per_horizon)).sum(dim=-1)


@torch.no_grad()
def rollout_candidate_costs(
    predictor: TemporalJEPAPredictor,
    candidates: CandidateSet,
    *,
    latents: torch.Tensor,
    current_time: int,
    horizons: tuple[int, ...],
    horizon_weights: torch.Tensor,
    rollout_steps: int = 32,
    base_seed: int = 0,
    random_stream: int = 100,
) -> torch.Tensor:
    """R-step cost under the shared counter-based random continuation."""
    if rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive")
    if not anchor_is_valid(current_time, len(latents), horizons, rollout_steps=rollout_steps):
        raise ValueError("rollout targets cross the video boundary")
    context = predictor.local_context_len
    capacity = candidates.capacity
    branches = candidates.actions
    values = candidates.memories.clone()
    timestamps = candidates.memory_timestamps.clone()
    event_ids = candidates.memory_event_ids.clone()
    cumulative = torch.zeros(
        (branches, len(horizons)), device=latents.device, dtype=torch.float32
    )

    for offset in range(rollout_steps):
        rollout_time = int(current_time) + offset
        if offset:
            event_time = rollout_time - context
            new_event = latents[event_time]
            all_values = torch.cat(
                (values, new_event.view(1, 1, -1).expand(branches, 1, -1)), dim=1
            )
            all_times = torch.cat(
                (
                    timestamps,
                    torch.full(
                        (branches, 1), event_time, device=latents.device, dtype=torch.long
                    ),
                ),
                dim=1,
            )
            all_ids = torch.cat(
                (
                    event_ids,
                    torch.full(
                        (branches, 1), event_time, device=latents.device, dtype=torch.long
                    ),
                ),
                dim=1,
            )
            draw = counter_uniform(base_seed, rollout_time, random_stream)
            evict = min(int(draw * (capacity + 1)), capacity)
            keep = torch.arange(capacity + 1, device=latents.device) != evict
            values = all_values[:, keep]
            timestamps = all_times[:, keep]
            event_ids = all_ids[:, keep]

        local = latents[rollout_time - context + 1 : rollout_time + 1]
        ages = (rollout_time - timestamps).float().clamp_min(0)
        predictions = predict_eviction_candidates(predictor, values, ages, local)
        targets = torch.stack([latents[rollout_time + h] for h in horizons])
        per_horizon = 1.0 - functional.cosine_similarity(
            predictions.float(), targets.unsqueeze(0).expand_as(predictions).float(), dim=-1
        )
        cumulative += per_horizon
    average = cumulative / float(rollout_steps)
    return (average * horizon_weights.to(average)).sum(dim=-1)
