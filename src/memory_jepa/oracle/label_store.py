from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from memory_jepa.utils.io import atomic_torch_save


UTILITY_TENSOR_KEYS = (
    "local",
    "candidate_memory",
    "candidate_age",
    "candidate_valid",
    "utility_per_horizon",
    "utility_scalar",
    "oracle_evict_index",
)


class UtilityLabelWriter:
    def __init__(self, extra_keys: tuple[str, ...] = ()) -> None:
        keys = (*UTILITY_TENSOR_KEYS, *extra_keys)
        if len(set(keys)) != len(keys):
            raise ValueError("Utility label keys must be unique")
        self.tensor_keys = keys
        self.rows: dict[str, list[torch.Tensor]] = {key: [] for key in keys}

    def append(self, **values: torch.Tensor) -> None:
        if set(values) != set(self.tensor_keys):
            raise ValueError(f"Expected utility fields {self.tensor_keys}, got {tuple(values)}")
        for key, value in values.items():
            self.rows[key].append(value.detach().cpu())

    def __len__(self) -> int:
        return len(self.rows[UTILITY_TENSOR_KEYS[0]])

    def save(self, path: str | Path, metadata: dict[str, Any]) -> None:
        if not len(self):
            raise ValueError("Cannot save an empty utility label set")
        tensors = {key: torch.stack(values) for key, values in self.rows.items()}
        atomic_torch_save({"metadata": metadata, "tensors": tensors}, path)

