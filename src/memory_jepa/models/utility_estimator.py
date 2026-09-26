from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .time_encoding import sinusoidal_encoding
from .transformer import PreNormTransformerLayer


@dataclass
class UtilityOutput:
    standardized_per_horizon: torch.Tensor
    utility_per_horizon: torch.Tensor
    utility_scalar: torch.Tensor
    ordering_scalar: torch.Tensor
    ordering_scale: torch.Tensor


class CausalUtilityEstimator(nn.Module):
    """Set-conditioned causal estimator; it never produces a future latent."""

    produces_future_latent = False

    def __init__(
        self,
        d_model: int,
        d_utility: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        num_horizons: int,
        horizon_weights: list[float] | torch.Tensor,
        architecture: str = "pooled_set",
    ) -> None:
        super().__init__()
        if d_utility % n_heads:
            raise ValueError("d_utility must be divisible by n_heads")
        if architecture not in {"pooled_set", "joint_set"}:
            raise ValueError(f"Unknown utility architecture: {architecture}")
        self.architecture = architecture
        self.memory_projection = nn.Linear(d_model, d_utility)
        self.context_projection = nn.Linear(d_model, d_utility)
        if architecture == "joint_set":
            self.type_embeddings = nn.Parameter(torch.empty(2, d_utility))
            nn.init.normal_(self.type_embeddings, std=0.02)
        else:
            self.register_parameter("type_embeddings", None)
        self.layers = nn.ModuleList(
            PreNormTransformerLayer(d_utility, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        )
        self.final_norm = nn.LayerNorm(d_utility)
        self.head = nn.Sequential(
            nn.Linear(d_utility, d_utility), nn.GELU(), nn.Linear(d_utility, num_horizons)
        )
        self.register_buffer("utility_mean", torch.zeros(num_horizons))
        self.register_buffer("utility_std", torch.ones(num_horizons))
        self.register_buffer(
            "horizon_weights", torch.as_tensor(horizon_weights, dtype=torch.float32)
        )

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if mean.shape != self.utility_mean.shape or std.shape != self.utility_std.shape:
            raise ValueError("Utility normalization statistics have incompatible shapes")
        self.utility_mean.copy_(mean)
        self.utility_std.copy_(std.clamp_min(1e-6))

    def standardize_targets(self, utility: torch.Tensor) -> torch.Tensor:
        return (utility - self.utility_mean) / self.utility_std.clamp_min(1e-6)

    def forward(
        self,
        memory_features: torch.Tensor,
        local_features: torch.Tensor,
        candidate_age: torch.Tensor,
        candidate_valid: torch.Tensor,
    ) -> UtilityOutput:
        valid_local = torch.ones(
            local_features.shape[:2], dtype=torch.bool, device=local_features.device
        )
        memory_hidden = (
            self.memory_projection(memory_features)
            + sinusoidal_encoding(candidate_age, self.memory_projection.out_features).to(
                memory_features.dtype
            )
        )
        if self.architecture == "joint_set":
            local_age = torch.arange(
                local_features.shape[1] - 1,
                -1,
                -1,
                device=local_features.device,
                dtype=torch.float32,
            ).expand(local_features.shape[0], -1)
            local_hidden = (
                self.context_projection(local_features)
                + sinusoidal_encoding(local_age, self.memory_projection.out_features).to(
                    local_features.dtype
                )
            )
            memory_hidden = memory_hidden + self.type_embeddings[0]
            local_hidden = local_hidden + self.type_embeddings[1]
            hidden = torch.cat((memory_hidden, local_hidden), dim=1)
            padding_mask = torch.cat((~candidate_valid.bool(), ~valid_local), dim=1)
        else:
            local_summary = (
                local_features * valid_local.unsqueeze(-1)
            ).sum(dim=1) / valid_local.sum(dim=1, keepdim=True).clamp_min(1)
            hidden = memory_hidden + self.context_projection(local_summary).unsqueeze(1)
            padding_mask = ~candidate_valid.bool()
        for layer in self.layers:
            hidden, _ = layer(hidden, key_padding_mask=padding_mask)
        candidate_hidden = hidden[:, : memory_features.shape[1]]
        standardized = self.head(self.final_norm(candidate_hidden))
        utility = standardized * self.utility_std + self.utility_mean
        utility = utility.masked_fill(~candidate_valid.unsqueeze(-1), 0.0)
        scalar = (utility * self.horizon_weights).sum(dim=-1)
        scalar = scalar.masked_fill(~candidate_valid, torch.inf)

        # Ranking and eviction need logits of order one. Using the raw scalar
        # here makes their gradients vanish because counterfactual utilities
        # are typically around 1e-4. Centering is slot-independent, and the
        # positive scale preserves exactly the same within-state ordering.
        centered_weights = self.utility_std * self.horizon_weights
        ordering_scale = centered_weights.square().sum().sqrt().clamp_min(1e-6)
        ordering_scalar = (standardized * centered_weights).sum(dim=-1) / ordering_scale
        ordering_scalar = ordering_scalar.masked_fill(~candidate_valid, torch.inf)
        return UtilityOutput(standardized, utility, scalar, ordering_scalar, ordering_scale)


class CausalUtilityEnsemble(nn.Module):
    """Average independently trained causal utility estimators at inference time."""

    produces_future_latent = False

    def __init__(self, estimators: list[CausalUtilityEstimator]) -> None:
        super().__init__()
        if not estimators:
            raise ValueError("A utility ensemble requires at least one estimator")
        self.estimators = nn.ModuleList(estimators)

    def forward(
        self,
        memory_features: torch.Tensor,
        local_features: torch.Tensor,
        candidate_age: torch.Tensor,
        candidate_valid: torch.Tensor,
    ) -> UtilityOutput:
        outputs = [
            estimator(memory_features, local_features, candidate_age, candidate_valid)
            for estimator in self.estimators
        ]
        return UtilityOutput(
            standardized_per_horizon=torch.stack(
                [output.standardized_per_horizon for output in outputs]
            ).mean(dim=0),
            utility_per_horizon=torch.stack(
                [output.utility_per_horizon for output in outputs]
            ).mean(dim=0),
            utility_scalar=torch.stack(
                [output.utility_scalar for output in outputs]
            ).mean(dim=0),
            ordering_scalar=torch.stack(
                [output.ordering_scalar for output in outputs]
            ).mean(dim=0),
            ordering_scale=torch.stack(
                [output.ordering_scale for output in outputs]
            ).mean(dim=0),
        )

