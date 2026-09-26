from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class MemoryPolicy(ABC):
    name = "base"
    causal = True

    def reset(self, seed: int | None = None) -> None:
        del seed

    @abstractmethod
    def select_evict_index(
        self,
        candidate_memory: torch.Tensor,
        candidate_timestamps: torch.Tensor,
        candidate_valid: torch.Tensor,
        local_context: torch.Tensor,
        current_time: torch.Tensor,
        **causal_metadata: Any,
    ) -> torch.Tensor:
        raise NotImplementedError


def masked_argmin(score: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return score.masked_fill(~valid, torch.inf).argmin(dim=-1)


def masked_argmax(score: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return score.masked_fill(~valid, -torch.inf).argmax(dim=-1)


def minmax_normalize(score: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    lower = score.masked_fill(~valid, torch.inf).amin(dim=-1, keepdim=True)
    upper = score.masked_fill(~valid, -torch.inf).amax(dim=-1, keepdim=True)
    result = (score - lower) / (upper - lower).clamp_min(1e-6)
    return result.masked_fill(~valid, 0.0)

