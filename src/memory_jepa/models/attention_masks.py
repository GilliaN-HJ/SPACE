from __future__ import annotations

import torch


def build_predictor_attention_mask(
    memory_slots: int, local_tokens: int, horizon_queries: int, device: torch.device | None = None
) -> torch.Tensor:
    """Return boolean [N,N] mask where True means attention is forbidden."""
    context = memory_slots + local_tokens
    total = context + horizon_queries
    mask = torch.zeros((total, total), dtype=torch.bool, device=device)
    mask[:context, context:] = True
    if horizon_queries:
        mask[context:, context:] = True
        diagonal = torch.arange(horizon_queries, device=device)
        mask[context + diagonal, context + diagonal] = False
    return mask


def build_key_padding_mask(memory_valid: torch.Tensor, local_tokens: int, horizons: int) -> torch.Tensor:
    batch = memory_valid.shape[0]
    valid_tail = torch.zeros(
        (batch, local_tokens + horizons), dtype=torch.bool, device=memory_valid.device
    )
    return torch.cat((~memory_valid.bool(), valid_tail), dim=1)

