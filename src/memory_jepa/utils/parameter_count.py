from __future__ import annotations

from collections.abc import Iterable

import torch.nn as nn


def count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    parameters: Iterable = module.parameters()
    if trainable_only:
        parameters = (parameter for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)


def expected_predictor_parameters(
    d_in: int, d_model: int, d_ff: int, n_layers: int, horizons: int
) -> int:
    input_projection = d_in * d_model + d_model
    layer = 4 * d_model**2 + 2 * d_model * d_ff + d_ff + 9 * d_model
    learned_tokens = horizons * d_model + 3 * d_model + 2 * d_model
    output_head = d_model * d_in + d_in
    return input_projection + n_layers * layer + learned_tokens + output_head


def expected_utility_parameters(
    d_model: int,
    d_utility: int,
    d_ff: int,
    n_layers: int,
    horizons: int,
    architecture: str = "pooled_set",
) -> int:
    condition = 2 * (d_model * d_utility + d_utility)
    layer = 4 * d_utility**2 + 2 * d_utility * d_ff + d_ff + 9 * d_utility
    final_norm = 2 * d_utility
    head = d_utility**2 + d_utility + d_utility * horizons + horizons
    type_embeddings = 2 * d_utility if architecture == "joint_set" else 0
    return condition + n_layers * layer + final_norm + head + type_embeddings

