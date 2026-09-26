from __future__ import annotations

import math

import torch


def sinusoidal_encoding(values: torch.Tensor, dimension: int) -> torch.Tensor:
    """Parameter-free sinusoidal encoding for non-negative ages or horizons."""
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    values = values.to(dtype=torch.float32).clamp_min(0).unsqueeze(-1)
    half = dimension // 2
    if half == 0:
        return values.new_zeros((*values.shape[:-1], dimension))
    scales = torch.exp(
        torch.arange(half, device=values.device, dtype=torch.float32)
        * (-math.log(10_000.0) / max(half - 1, 1))
    )
    angles = values * scales
    result = torch.cat((angles.sin(), angles.cos()), dim=-1)
    if result.shape[-1] < dimension:
        result = torch.nn.functional.pad(result, (0, dimension - result.shape[-1]))
    return result

