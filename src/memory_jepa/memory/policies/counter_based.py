from __future__ import annotations

import hashlib

import torch
import torch.nn.functional as functional

from .base import MemoryPolicy, masked_argmax, masked_argmin


UINT64_MASK = (1 << 64) - 1


def stable_video_seed(video_id: str, policy_seed: int) -> int:
    payload = f"{video_id}\0{int(policy_seed)}".encode("utf-8")
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
    return value & ((1 << 63) - 1)


def splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & UINT64_MASK
    return (value ^ (value >> 31)) & UINT64_MASK


def counter_uniform(base_seed: int, time_index: int, stream_id: int) -> float:
    value = int(base_seed) & UINT64_MASK
    value ^= (int(time_index) * 0xD2B74407B1CE6E93) & UINT64_MASK
    value ^= (int(stream_id) * 0xCA5A826395121157) & UINT64_MASK
    random_bits = splitmix64(value)
    return float(random_bits >> 11) / float(1 << 53)


class CounterBasedPolicy(MemoryPolicy):
    def __init__(self, seed: int = 0) -> None:
        self.base_seed = int(seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.base_seed = int(seed)


class CounterBasedRandomEvictionPolicy(CounterBasedPolicy):
    name = "random"

    def select_evict_index(
        self,
        candidate_memory,
        candidate_timestamps,
        candidate_valid,
        local_context,
        current_time,
        **metadata,
    ):
        del candidate_memory, candidate_timestamps, local_context, metadata
        outputs = []
        for batch_index in range(candidate_valid.shape[0]):
            valid = candidate_valid[batch_index].nonzero(as_tuple=False).squeeze(-1)
            time_index = int(current_time[batch_index].item())
            draw = counter_uniform(self.base_seed, time_index, 0)
            rank = min(int(draw * len(valid)), len(valid) - 1)
            outputs.append(valid[rank])
        return torch.stack(outputs).to(candidate_valid.device)


class CounterBasedReservoirSamplingPolicy(CounterBasedPolicy):
    name = "reservoir"

    def __init__(
        self,
        capacity: int,
        seed: int = 0,
        keep_stream: int = 0,
        replacement_stream: int = 1,
    ) -> None:
        super().__init__(seed)
        self.capacity = int(capacity)
        self.keep_stream = int(keep_stream)
        self.replacement_stream = int(replacement_stream)
        if self.keep_stream < 0 or self.replacement_stream < 0:
            raise ValueError("counter RNG stream IDs must be non-negative")
        if self.keep_stream == self.replacement_stream:
            raise ValueError("Reservoir keep and replacement streams must differ")

    def reservoir_proposal(
        self,
        candidate_valid: torch.Tensor,
        current_time: torch.Tensor,
        candidate_event_ids: torch.Tensor,
    ) -> torch.Tensor:
        outputs = []
        for batch_index in range(candidate_valid.shape[0]):
            valid = candidate_valid[batch_index].nonzero(as_tuple=False).squeeze(-1)
            candidate_index = valid[-1]
            seen = max(
                int(candidate_event_ids[batch_index, candidate_index].item()) + 1,
                self.capacity + 1,
            )
            time_index = int(current_time[batch_index].item())
            keep_probability = self.capacity / float(seen)
            if counter_uniform(self.base_seed, time_index, self.keep_stream) > keep_probability:
                outputs.append(candidate_index)
                continue
            old = valid[:-1]
            draw = counter_uniform(self.base_seed, time_index, self.replacement_stream)
            rank = min(int(draw * len(old)), len(old) - 1)
            outputs.append(old[rank])
        return torch.stack(outputs).to(candidate_valid.device)

    def select_evict_index(
        self,
        candidate_memory,
        candidate_timestamps,
        candidate_valid,
        local_context,
        current_time,
        **metadata,
    ):
        del candidate_memory, candidate_timestamps, local_context
        event_ids = metadata.get("candidate_event_ids")
        if event_ids is None:
            raise ValueError("Counter Reservoir requires candidate_event_ids")
        return self.reservoir_proposal(candidate_valid, current_time, event_ids)


class CounterBasedLearnedFIFOReservoirChooserPolicy(CounterBasedReservoirSamplingPolicy):
    name = "learned_fifo_reservoir"

    def __init__(self, capacity: int, seed: int = 0, threshold: float = 0.25) -> None:
        super().__init__(capacity, seed)
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        self.threshold = float(threshold)

    def select_evict_index(
        self,
        candidate_memory,
        candidate_timestamps,
        candidate_valid,
        local_context,
        current_time,
        **metadata,
    ):
        ordering = metadata.get("learned_ordering")
        event_ids = metadata.get("candidate_event_ids")
        if ordering is None or event_ids is None:
            raise ValueError("Counter chooser requires learned_ordering and candidate_event_ids")
        reservoir_index = self.reservoir_proposal(candidate_valid, current_time, event_ids)
        fifo_index = masked_argmin(candidate_timestamps.float(), candidate_valid)
        reservoir_score = ordering.gather(1, reservoir_index.unsqueeze(1)).squeeze(1)
        fifo_score = ordering.gather(1, fifo_index.unsqueeze(1)).squeeze(1)
        choose_fifo = (reservoir_score - fifo_score) >= self.threshold
        return torch.where(choose_fifo, fifo_index, reservoir_index)



class CounterBasedGatedLearnedUtilityReservoirPolicy(CounterBasedReservoirSamplingPolicy):
    """Fall back to counter-based Reservoir unless the full selector is confident."""

    name = "learned_utility_reservoir"

    def __init__(self, capacity: int, seed: int = 0, threshold: float = 0.5) -> None:
        super().__init__(capacity, seed)
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        self.threshold = float(threshold)

    def select_evict_index(
        self, candidate_memory, candidate_timestamps, candidate_valid,
        local_context, current_time, **metadata,
    ):
        ordering = metadata.get("learned_ordering")
        event_ids = metadata.get("candidate_event_ids")
        if ordering is None or event_ids is None:
            raise ValueError("Counter gated selector requires ordering and event ids")
        reservoir_index = self.reservoir_proposal(
            candidate_valid, current_time, event_ids
        )
        learned_index = masked_argmin(ordering, candidate_valid)
        learned_score = ordering.gather(1, learned_index.unsqueeze(1)).squeeze(1)
        reservoir_score = ordering.gather(
            1, reservoir_index.unsqueeze(1)
        ).squeeze(1)
        confident = (reservoir_score - learned_score) >= self.threshold
        return torch.where(confident, learned_index, reservoir_index)

class CounterBasedMultiProposalOraclePolicy(CounterBasedPolicy):
    """Exact teacher over a small, ordered set of causal eviction proposals."""

    causal = False
    valid_proposals = ("random", "fifo", "reservoir", "redundancy", "learned")

    def __init__(self, capacity: int, proposals: tuple[str, ...], seed: int = 0) -> None:
        super().__init__(seed)
        if not proposals:
            raise ValueError("multi-proposal Oracle requires at least one proposal")
        unknown = set(proposals).difference(self.valid_proposals)
        if unknown:
            raise ValueError(f"unknown proposals: {sorted(unknown)}")
        if len(set(proposals)) != len(proposals):
            raise ValueError("proposals must be unique")
        self.capacity = int(capacity)
        self.proposals = tuple(proposals)
        self.name = f"proposal_oracle_{len(proposals)}"
        self.last_selected_proposal = ""
        self.last_unique_proposals = 0

    def _random_proposal(self, candidate_valid, current_time):
        outputs = []
        for batch_index in range(candidate_valid.shape[0]):
            valid = candidate_valid[batch_index].nonzero(as_tuple=False).squeeze(-1)
            time_index = int(current_time[batch_index].item())
            draw = counter_uniform(self.base_seed, time_index, 0)
            rank = min(int(draw * len(valid)), len(valid) - 1)
            outputs.append(valid[rank])
        return torch.stack(outputs).to(candidate_valid.device)

    def _reservoir_proposal(self, candidate_valid, current_time, event_ids):
        outputs = []
        for batch_index in range(candidate_valid.shape[0]):
            valid = candidate_valid[batch_index].nonzero(as_tuple=False).squeeze(-1)
            candidate_index = valid[-1]
            seen = max(
                int(event_ids[batch_index, candidate_index].item()) + 1,
                self.capacity + 1,
            )
            time_index = int(current_time[batch_index].item())
            keep_probability = self.capacity / float(seen)
            if counter_uniform(self.base_seed, time_index, 20) > keep_probability:
                outputs.append(candidate_index)
            else:
                old = valid[:-1]
                draw = counter_uniform(self.base_seed, time_index, 21)
                rank = min(int(draw * len(old)), len(old) - 1)
                outputs.append(old[rank])
        return torch.stack(outputs).to(candidate_valid.device)

    @staticmethod
    def _redundancy_proposal(candidate_memory, candidate_valid):
        candidates = functional.normalize(candidate_memory, dim=-1)
        similarity = candidates @ candidates.transpose(1, 2)
        slots = similarity.shape[-1]
        diagonal = torch.eye(
            slots, dtype=torch.bool, device=similarity.device
        ).unsqueeze(0)
        pair_valid = (
            candidate_valid.unsqueeze(1)
            & candidate_valid.unsqueeze(2)
            & ~diagonal
        )
        redundancy = similarity.masked_fill(~pair_valid, -torch.inf).amax(dim=-1)
        return masked_argmax(redundancy, candidate_valid)

    def proposal_indices(
        self,
        candidate_memory,
        candidate_timestamps,
        candidate_valid,
        current_time,
        **metadata,
    ):
        event_ids = metadata.get("candidate_event_ids")
        learned_ordering = metadata.get("learned_ordering")
        indices = []
        for proposal in self.proposals:
            if proposal == "random":
                index = self._random_proposal(candidate_valid, current_time)
            elif proposal == "fifo":
                index = masked_argmin(candidate_timestamps.float(), candidate_valid)
            elif proposal == "reservoir":
                if event_ids is None:
                    raise ValueError("Reservoir proposal requires candidate_event_ids")
                index = self._reservoir_proposal(
                    candidate_valid, current_time, event_ids
                )
            elif proposal == "redundancy":
                index = self._redundancy_proposal(
                    candidate_memory, candidate_valid
                )
            else:
                if learned_ordering is None:
                    raise ValueError("Learned proposal requires learned_ordering")
                index = masked_argmin(learned_ordering, candidate_valid)
            indices.append(index)
        return torch.stack(indices, dim=1)

    def select_evict_index(
        self,
        candidate_memory,
        candidate_timestamps,
        candidate_valid,
        local_context,
        current_time,
        **metadata,
    ):
        del local_context
        proposal_indices = self.proposal_indices(
            candidate_memory,
            candidate_timestamps,
            candidate_valid,
            current_time,
            **metadata,
        )
        self.last_unique_proposals = int(torch.unique(proposal_indices[0]).numel())
        if len(self.proposals) == 1:
            selected = torch.zeros(
                candidate_valid.shape[0], dtype=torch.long,
                device=candidate_valid.device,
            )
        else:
            utility = metadata.get("oracle_utility")
            if utility is None:
                raise ValueError("Multi-proposal Oracle requires oracle_utility")
            proposal_utility = utility.gather(1, proposal_indices)
            # torch.argmin gives the fixed deterministic earliest-proposal tie rule.
            selected = proposal_utility.argmin(dim=1)
        if candidate_valid.shape[0] == 1:
            self.last_selected_proposal = self.proposals[int(selected.item())]
        return proposal_indices.gather(1, selected.unsqueeze(1)).squeeze(1)


class CounterBasedLearnedMultiProposalPolicy(CounterBasedMultiProposalOraclePolicy):
    """Causally score four structured proposals, with Random as safe default."""

    name = "learned_multi_proposal"
    causal = True

    def __init__(self, capacity: int, seed: int = 0, threshold: float = 0.0) -> None:
        super().__init__(
            capacity, ("random", "fifo", "reservoir", "redundancy"), seed
        )
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        self.name = "learned_multi_proposal"
        self.threshold = float(threshold)

    def select_evict_index(
        self,
        candidate_memory,
        candidate_timestamps,
        candidate_valid,
        local_context,
        current_time,
        **metadata,
    ):
        del local_context
        ordering = metadata.get("learned_ordering")
        if ordering is None:
            raise ValueError("Learned multi-proposal policy requires learned_ordering")
        proposal_indices = self.proposal_indices(
            candidate_memory,
            candidate_timestamps,
            candidate_valid,
            current_time,
            **metadata,
        )
        proposal_scores = ordering.gather(1, proposal_indices)
        learned_choice = proposal_scores.argmin(dim=1)
        random_score = proposal_scores[:, 0]
        learned_score = proposal_scores.gather(
            1, learned_choice.unsqueeze(1)
        ).squeeze(1)
        confident = (random_score - learned_score) >= self.threshold
        selected = torch.where(confident, learned_choice, torch.zeros_like(learned_choice))
        self.last_unique_proposals = int(torch.unique(proposal_indices[0]).numel())
        if candidate_valid.shape[0] == 1:
            self.last_selected_proposal = self.proposals[int(selected.item())]
        return proposal_indices.gather(1, selected.unsqueeze(1)).squeeze(1)


class CounterBasedTwoProposalOraclePolicy(CounterBasedReservoirSamplingPolicy):
    name = "two_proposal_oracle"
    causal = False

    def select_evict_index(
        self,
        candidate_memory,
        candidate_timestamps,
        candidate_valid,
        local_context,
        current_time,
        **metadata,
    ):
        utility = metadata.get("oracle_utility")
        event_ids = metadata.get("candidate_event_ids")
        if utility is None or event_ids is None:
            raise ValueError("Counter two-proposal Oracle requires utility and event ids")
        reservoir_index = self.reservoir_proposal(candidate_valid, current_time, event_ids)
        fifo_index = masked_argmin(candidate_timestamps.float(), candidate_valid)
        reservoir_utility = utility.gather(1, reservoir_index.unsqueeze(1)).squeeze(1)
        fifo_utility = utility.gather(1, fifo_index.unsqueeze(1)).squeeze(1)
        # Fixed deterministic tie rule: FIFO wins exact ties.
        choose_fifo = fifo_utility <= reservoir_utility
        return torch.where(choose_fifo, fifo_index, reservoir_index)
