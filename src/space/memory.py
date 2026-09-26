from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch


@dataclass
class MemoryState:
    values: torch.Tensor
    timestamps: torch.Tensor
    event_ids: torch.Tensor
    valid: torch.Tensor

    @classmethod
    def empty(
        cls, capacity: int, feature_dim: int, device: torch.device | str = "cpu"
    ) -> "MemoryState":
        return cls(
            values=torch.zeros((capacity, feature_dim), device=device),
            timestamps=torch.full((capacity,), -1, dtype=torch.long, device=device),
            event_ids=torch.full((capacity,), -1, dtype=torch.long, device=device),
            valid=torch.zeros(capacity, dtype=torch.bool, device=device),
        )

    @property
    def capacity(self) -> int:
        return int(self.values.shape[0])

    @property
    def active_count(self) -> int:
        return int(self.valid.sum().item())

    @property
    def full(self) -> bool:
        return self.active_count == self.capacity

    def clone(self) -> "MemoryState":
        return MemoryState(
            self.values.clone(),
            self.timestamps.clone(),
            self.event_ids.clone(),
            self.valid.clone(),
        )

    def insert_first_free(
        self, value: torch.Tensor, timestamp: int, event_id: int
    ) -> "MemoryState":
        if self.full:
            raise ValueError("memory is full")
        result = self.clone()
        slot = int((~result.valid).nonzero(as_tuple=False)[0].item())
        result.values[slot] = value
        result.timestamps[slot] = int(timestamp)
        result.event_ids[slot] = int(event_id)
        result.valid[slot] = True
        return result


@dataclass(frozen=True)
class CandidateSet:
    """The K+1 feasible evictions at a full-memory update.

    Action a deletes candidate item a. The final action K deletes the new
    event and therefore keeps the current memory unchanged.
    """

    items: torch.Tensor
    timestamps: torch.Tensor
    event_ids: torch.Tensor
    memories: torch.Tensor
    memory_timestamps: torch.Tensor
    memory_event_ids: torch.Tensor

    @property
    def actions(self) -> int:
        return int(self.items.shape[0])

    @property
    def capacity(self) -> int:
        return self.actions - 1

    @property
    def hold_action(self) -> int:
        return self.capacity

    def ages(self, current_time: int) -> tuple[torch.Tensor, torch.Tensor]:
        evicted = (int(current_time) - self.timestamps).float().clamp_min(0)
        memory = (int(current_time) - self.memory_timestamps).float().clamp_min(0)
        return evicted, memory


def enumerate_evictions(
    state: MemoryState,
    new_value: torch.Tensor,
    new_timestamp: int,
    new_event_id: int,
) -> CandidateSet:
    if not state.full:
        raise ValueError("K+1 actions exist only after memory is full")
    active = state.valid.nonzero(as_tuple=False).squeeze(-1)
    items = torch.cat((state.values[active], new_value.view(1, -1)), dim=0)
    timestamps = torch.cat(
        (
            state.timestamps[active],
            torch.tensor([new_timestamp], dtype=torch.long, device=state.timestamps.device),
        )
    )
    event_ids = torch.cat(
        (
            state.event_ids[active],
            torch.tensor([new_event_id], dtype=torch.long, device=state.event_ids.device),
        )
    )
    actions = len(items)
    keep = ~torch.eye(actions, dtype=torch.bool, device=items.device)
    memories = items.unsqueeze(0).expand(actions, -1, -1)[keep].reshape(
        actions, actions - 1, -1
    )
    memory_timestamps = timestamps.unsqueeze(0).expand(actions, -1)[keep].reshape(
        actions, actions - 1
    )
    memory_event_ids = event_ids.unsqueeze(0).expand(actions, -1)[keep].reshape(
        actions, actions - 1
    )
    return CandidateSet(
        items,
        timestamps,
        event_ids,
        memories,
        memory_timestamps,
        memory_event_ids,
    )


def execute_action(candidates: CandidateSet, action: int) -> MemoryState:
    if not 0 <= action < candidates.actions:
        raise ValueError("action lies outside candidate set")
    capacity = candidates.capacity
    return MemoryState(
        values=candidates.memories[action].clone(),
        timestamps=candidates.memory_timestamps[action].clone(),
        event_ids=candidates.memory_event_ids[action].clone(),
        valid=torch.ones(capacity, dtype=torch.bool, device=candidates.items.device),
    )


UINT64_MASK = (1 << 64) - 1


def stable_video_seed(video_id: str, policy_seed: int) -> int:
    payload = f"{video_id}\0{int(policy_seed)}".encode("utf-8")
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
    return value & ((1 << 63) - 1)


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & UINT64_MASK
    return (value ^ (value >> 31)) & UINT64_MASK


def counter_uniform(base_seed: int, time_index: int, stream_id: int) -> float:
    value = int(base_seed) & UINT64_MASK
    value ^= (int(time_index) * 0xD2B74407B1CE6E93) & UINT64_MASK
    value ^= (int(stream_id) * 0xCA5A826395121157) & UINT64_MASK
    return float(_splitmix64(value) >> 11) / float(1 << 53)


def counter_reservoir_action(
    candidates: CandidateSet,
    *,
    current_time: int,
    base_seed: int,
    keep_stream: int = 0,
    replacement_stream: int = 1,
) -> int:
    """Deterministic counter-based Reservoir action used as SPACE's fallback."""
    capacity = candidates.capacity
    seen = max(int(candidates.event_ids[-1].item()) + 1, capacity + 1)
    keep_probability = capacity / float(seen)
    if counter_uniform(base_seed, current_time, keep_stream) > keep_probability:
        return candidates.hold_action
    draw = counter_uniform(base_seed, current_time, replacement_stream)
    return min(int(draw * capacity), capacity - 1)
