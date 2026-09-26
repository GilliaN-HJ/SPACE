from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MemoryState:
    values: torch.Tensor
    timestamps: torch.Tensor
    event_ids: torch.Tensor
    valid: torch.Tensor
    access: torch.Tensor

    @classmethod
    def empty(
        cls, capacity: int, d_in: int, device: torch.device | str = "cpu"
    ) -> "MemoryState":
        return cls(
            values=torch.zeros((capacity, d_in), device=device),
            timestamps=torch.full((capacity,), -1, dtype=torch.long, device=device),
            event_ids=torch.full((capacity,), -1, dtype=torch.long, device=device),
            valid=torch.zeros(capacity, dtype=torch.bool, device=device),
            access=torch.zeros(capacity, dtype=torch.float32, device=device),
        )

    @property
    def capacity(self) -> int:
        return self.values.shape[0]

    @property
    def active_count(self) -> int:
        return int(self.valid.sum().item())

    def clone(self) -> "MemoryState":
        return MemoryState(**{name: getattr(self, name).clone() for name in self.__dataclass_fields__})

    def validate(self, current_time: int | None = None, local_context_len: int | None = None) -> None:
        if self.active_count > self.capacity:
            raise AssertionError("Active memory exceeds capacity")
        if current_time is not None and local_context_len is not None and self.valid.any():
            maximum_timestamp = int(self.timestamps[self.valid].max().item())
            if maximum_timestamp > current_time - local_context_len:
                raise AssertionError("Long-term memory overlaps the local context")

    def padded(self, physical_slots: int, current_time: int) -> tuple[torch.Tensor, ...]:
        if physical_slots < self.capacity:
            raise ValueError("physical_slots cannot be smaller than online capacity")
        d_in = self.values.shape[-1]
        values = torch.zeros((physical_slots, d_in), device=self.values.device, dtype=self.values.dtype)
        valid = torch.zeros(physical_slots, device=self.valid.device, dtype=torch.bool)
        age = torch.zeros(physical_slots, device=self.values.device, dtype=torch.float32)
        count = self.active_count
        if count:
            active = self.valid.nonzero(as_tuple=False).squeeze(-1)
            values[:count] = self.values[active]
            valid[:count] = True
            age[:count] = current_time - self.timestamps[active]
        return values, valid, age

