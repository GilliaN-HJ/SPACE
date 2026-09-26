from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as functional

from .base import MemoryPolicy, masked_argmax, masked_argmin, minmax_normalize


class FIFOPolicy(MemoryPolicy):
    name = "fifo"

    def select_evict_index(self, candidate_memory, candidate_timestamps, candidate_valid, local_context, current_time, **metadata):
        del candidate_memory, local_context, current_time, metadata
        return masked_argmin(candidate_timestamps.float(), candidate_valid)


class RandomEvictionPolicy(MemoryPolicy):
    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.generator = torch.Generator().manual_seed(seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.generator.manual_seed(seed)

    def select_evict_index(self, candidate_memory, candidate_timestamps, candidate_valid, local_context, current_time, **metadata):
        del candidate_memory, candidate_timestamps, local_context, current_time, metadata
        scores = torch.rand(candidate_valid.shape, generator=self.generator).to(candidate_valid.device)
        return masked_argmin(scores, candidate_valid)


class ReservoirSamplingPolicy(MemoryPolicy):
    name = "reservoir"

    def __init__(self, capacity: int, seed: int = 0) -> None:
        self.capacity = capacity
        self.generator = torch.Generator().manual_seed(seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.generator.manual_seed(seed)

    def select_evict_index(self, candidate_memory, candidate_timestamps, candidate_valid, local_context, current_time, **metadata):
        del candidate_memory, candidate_timestamps, local_context, current_time
        event_ids = metadata.get("candidate_event_ids")
        if event_ids is None:
            raise ValueError("Reservoir policy requires causal candidate_event_ids")
        outputs = []
        for batch_index in range(candidate_valid.shape[0]):
            valid_indices = candidate_valid[batch_index].nonzero(as_tuple=False).squeeze(-1)
            candidate_index = valid_indices[-1]
            seen = max(int(event_ids[batch_index, candidate_index].item()) + 1, self.capacity + 1)
            keep_probability = self.capacity / float(seen)
            draw = torch.rand((), generator=self.generator).item()
            if draw > keep_probability:
                outputs.append(candidate_index)
            else:
                old = valid_indices[:-1]
                chosen = int(torch.randint(len(old), (), generator=self.generator).item())
                outputs.append(old[chosen])
        return torch.stack(outputs).to(candidate_valid.device)


class SurprisePolicy(MemoryPolicy):
    name = "surprise"

    def select_evict_index(self, candidate_memory, candidate_timestamps, candidate_valid, local_context, current_time, **metadata):
        del candidate_timestamps, current_time, metadata
        reference = functional.normalize(local_context.mean(dim=1), dim=-1)
        candidates = functional.normalize(candidate_memory, dim=-1)
        surprise = 1.0 - (candidates * reference.unsqueeze(1)).sum(dim=-1)
        return masked_argmin(surprise, candidate_valid)


class SimilarityRedundancyPolicy(MemoryPolicy):
    name = "similarity"

    def select_evict_index(self, candidate_memory, candidate_timestamps, candidate_valid, local_context, current_time, **metadata):
        del candidate_timestamps, local_context, current_time, metadata
        candidates = functional.normalize(candidate_memory, dim=-1)
        similarity = candidates @ candidates.transpose(1, 2)
        slots = similarity.shape[-1]
        diagonal = torch.eye(slots, dtype=torch.bool, device=similarity.device).unsqueeze(0)
        pair_valid = candidate_valid.unsqueeze(1) & candidate_valid.unsqueeze(2) & ~diagonal
        redundancy = similarity.masked_fill(~pair_valid, -torch.inf).amax(dim=-1)
        return masked_argmax(redundancy, candidate_valid)


class RecencySurpriseAccessPolicy(MemoryPolicy):
    name = "recency_surprise_access"

    def __init__(self, surprise_weight: float = 0.45, recency_weight: float = 0.25, access_weight: float = 0.30) -> None:
        self.surprise_weight = surprise_weight
        self.recency_weight = recency_weight
        self.access_weight = access_weight

    def select_evict_index(self, candidate_memory, candidate_timestamps, candidate_valid, local_context, current_time, **metadata):
        reference = functional.normalize(local_context.mean(dim=1), dim=-1)
        candidates = functional.normalize(candidate_memory, dim=-1)
        surprise = 1.0 - (candidates * reference.unsqueeze(1)).sum(dim=-1)
        age = (current_time.unsqueeze(1) - candidate_timestamps).float().clamp_min(0)
        recency = 1.0 / (1.0 + age)
        access = metadata.get("candidate_access", torch.zeros_like(recency))
        retention = (
            self.surprise_weight * minmax_normalize(surprise, candidate_valid)
            + self.recency_weight * minmax_normalize(recency, candidate_valid)
            + self.access_weight * minmax_normalize(access, candidate_valid)
        )
        return masked_argmin(retention, candidate_valid)


class AttentionAccessPolicy(MemoryPolicy):
    name = "attention"

    def select_evict_index(self, candidate_memory, candidate_timestamps, candidate_valid, local_context, current_time, **metadata):
        del candidate_memory, candidate_timestamps, local_context, current_time
        attention = metadata.get("attention_score")
        if attention is None:
            raise ValueError("Attention policy requires causal attention_score")
        access = metadata.get("candidate_access", torch.zeros_like(attention))
        return masked_argmin(attention + access, candidate_valid)


class CausalSelfInfluencePolicy(MemoryPolicy):
    """Evict the slot with least effect on the predictor's own causal forecast."""

    name = "self_influence"

    def select_evict_index(self, candidate_memory, candidate_timestamps, candidate_valid, local_context, current_time, **metadata):
        del candidate_memory, candidate_timestamps, local_context, current_time
        influence = metadata.get("self_influence")
        if influence is None:
            raise ValueError("Self-influence policy requires causal self_influence")
        return masked_argmin(influence, candidate_valid)


class FutureSaliencePolicy(MemoryPolicy):
    """Non-causal salience upper baseline, intentionally distinct from utility."""

    name = "future_salience"
    causal = False

    def select_evict_index(self, candidate_memory, candidate_timestamps, candidate_valid, local_context, current_time, **metadata):
        del candidate_timestamps, local_context, current_time
        targets = metadata.get("future_targets")
        if targets is None:
            raise ValueError("Future salience baseline requires training-only future targets")
        candidates = functional.normalize(candidate_memory, dim=-1)
        future = functional.normalize(targets, dim=-1)
        salience = torch.einsum("bsd,bhd->bsh", candidates, future).amax(dim=-1)
        return masked_argmin(salience, candidate_valid)


class OracleCounterfactualPolicy(MemoryPolicy):
    name = "oracle"
    causal = False

    def select_evict_index(self, candidate_memory, candidate_timestamps, candidate_valid, local_context, current_time, **metadata):
        del candidate_memory, candidate_timestamps, local_context, current_time
        utility = metadata.get("oracle_utility")
        if utility is None:
            raise ValueError("Oracle policy requires training-only oracle_utility")
        return masked_argmin(utility, candidate_valid)


class LearnedUtilityPolicy(MemoryPolicy):
    name = "learned_utility"

    def select_evict_index(self, candidate_memory, candidate_timestamps, candidate_valid, local_context, current_time, **metadata):
        del candidate_memory, candidate_timestamps, local_context, current_time
        utility = metadata.get("learned_utility")
        if utility is None:
            raise ValueError("Learned policy requires causal learned_utility")
        return masked_argmin(utility, candidate_valid)


class LearnedFIFOReservoirChooserPolicy(ReservoirSamplingPolicy):
    """Use learned utility to choose between FIFO and reservoir proposals."""

    name = "learned_fifo_reservoir"

    def __init__(self, capacity: int, seed: int = 0, threshold: float = 0.0) -> None:
        super().__init__(capacity, seed)
        if threshold < 0:
            raise ValueError("baseline chooser threshold must be non-negative")
        self.threshold = float(threshold)

    def select_evict_index(self, candidate_memory, candidate_timestamps, candidate_valid, local_context, current_time, **metadata):
        ordering = metadata.get("learned_ordering")
        if ordering is None:
            raise ValueError("Learned baseline chooser requires causal learned_ordering")
        reservoir_index = super().select_evict_index(
            candidate_memory,
            candidate_timestamps,
            candidate_valid,
            local_context,
            current_time,
            **metadata,
        )
        fifo_index = masked_argmin(candidate_timestamps.float(), candidate_valid)
        reservoir_score = ordering.gather(1, reservoir_index.unsqueeze(1)).squeeze(1)
        fifo_score = ordering.gather(1, fifo_index.unsqueeze(1)).squeeze(1)
        choose_fifo = (reservoir_score - fifo_score) >= self.threshold
        return torch.where(choose_fifo, fifo_index, reservoir_index)


class GatedLearnedUtilityReservoirPolicy(ReservoirSamplingPolicy):
    """Use learned utility only when it confidently disagrees with reservoir."""

    name = "learned_utility_reservoir"

    def __init__(self, capacity: int, seed: int = 0, threshold: float = 0.5) -> None:
        super().__init__(capacity, seed)
        if threshold < 0:
            raise ValueError("learned reservoir threshold must be non-negative")
        self.threshold = float(threshold)

    def select_evict_index(self, candidate_memory, candidate_timestamps, candidate_valid, local_context, current_time, **metadata):
        ordering = metadata.get("learned_ordering")
        if ordering is None:
            raise ValueError("Gated learned policy requires causal learned_ordering")
        baseline_index = super().select_evict_index(
            candidate_memory,
            candidate_timestamps,
            candidate_valid,
            local_context,
            current_time,
            **metadata,
        )
        learned_index = masked_argmin(ordering, candidate_valid)
        learned_score = ordering.gather(1, learned_index.unsqueeze(1)).squeeze(1)
        baseline_score = ordering.gather(1, baseline_index.unsqueeze(1)).squeeze(1)
        confident = (baseline_score - learned_score) >= self.threshold
        return torch.where(confident, learned_index, baseline_index)


class GatedLearnedUtilityFIFOPolicy(MemoryPolicy):
    """Use learned utility only when it confidently disagrees with FIFO."""

    name = "learned_utility_fifo"

    def __init__(self, threshold: float = 0.5) -> None:
        if threshold < 0:
            raise ValueError("learned FIFO threshold must be non-negative")
        self.threshold = float(threshold)

    def select_evict_index(self, candidate_memory, candidate_timestamps, candidate_valid, local_context, current_time, **metadata):
        del candidate_memory, local_context, current_time
        ordering = metadata.get("learned_ordering")
        if ordering is None:
            raise ValueError("Gated learned policy requires causal learned_ordering")
        learned_index = masked_argmin(ordering, candidate_valid)
        fifo_index = masked_argmin(candidate_timestamps.float(), candidate_valid)
        learned_score = ordering.gather(1, learned_index.unsqueeze(1)).squeeze(1)
        fifo_score = ordering.gather(1, fifo_index.unsqueeze(1)).squeeze(1)
        confident = (fifo_score - learned_score) >= self.threshold
        return torch.where(confident, learned_index, fifo_index)


def build_policy(
    name: str,
    capacity: int,
    seed: int = 0,
    learned_fifo_threshold: float = 0.5,
) -> MemoryPolicy:
    policies: dict[str, Any] = {
        "fifo": FIFOPolicy,
        "random": lambda: RandomEvictionPolicy(seed),
        "reservoir": lambda: ReservoirSamplingPolicy(capacity, seed),
        "surprise": SurprisePolicy,
        "similarity": SimilarityRedundancyPolicy,
        "recency_surprise_access": RecencySurpriseAccessPolicy,
        "attention": AttentionAccessPolicy,
        "self_influence": CausalSelfInfluencePolicy,
        "future_salience": FutureSaliencePolicy,
        "oracle": OracleCounterfactualPolicy,
        "learned_utility": LearnedUtilityPolicy,
        "learned_utility_fifo": lambda: GatedLearnedUtilityFIFOPolicy(
            learned_fifo_threshold
        ),
        "learned_utility_reservoir": lambda: GatedLearnedUtilityReservoirPolicy(
            capacity, seed, learned_fifo_threshold
        ),
        "learned_fifo_reservoir": lambda: LearnedFIFOReservoirChooserPolicy(
            capacity, seed, learned_fifo_threshold
        ),
    }
    if name not in policies:
        raise KeyError(f"Unknown memory policy: {name}")
    factory = policies[name]
    return factory()

