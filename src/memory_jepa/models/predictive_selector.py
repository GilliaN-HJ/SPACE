from __future__ import annotations

"""Predictor-grounded causal memory selectors.

The selector consumes candidate-specific outputs of a frozen JEPA predictor.
Those outputs are generated from the observed prefix and a counterfactual
memory *candidate* only; no realized future is passed to this module.  Keeping
the action head separate from the predictor makes the causal boundary
explicit and gives the selector a representation aligned with the downstream
future-latent prediction task.
"""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class PredictiveSelectorOutput:
    scores: torch.Tensor


class PredictiveResidualSelector(nn.Module):
    """Score every eviction candidate from frozen-predictor features.

    ``action_features`` has shape ``[B, P, F]`` where ``P=K+1`` is the number
    of possible evictions.  The head predicts a cost/residual, so lower scores
    are preferred.  The model is intentionally small relative to the JEPA
    predictor; the predictor supplies the causal belief representation.
    """

    produces_future_latent = False

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        num_proposals: int = 17,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or hidden_dim <= 0:
            raise ValueError("feature_dim and hidden_dim must be positive")
        if num_proposals <= 1:
            raise ValueError("num_proposals must be greater than one")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_proposals = int(num_proposals)
        self.norm = nn.LayerNorm(feature_dim)
        self.head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, action_features: torch.Tensor) -> PredictiveSelectorOutput:
        if action_features.ndim != 3:
            raise ValueError("action_features must have shape [B,P,F]")
        if action_features.shape[-1] != self.feature_dim:
            raise ValueError("action feature dimension does not match selector")
        if action_features.shape[1] != self.num_proposals:
            raise ValueError("proposal count does not match selector")
        scores = self.head(self.norm(action_features)).squeeze(-1)
        return PredictiveSelectorOutput(scores=scores)


def predictive_feature_dim(predictor_d_model: int, num_horizons: int) -> int:
    """Dimension used by :func:`build_predictive_action_features`."""

    # Flattened projected predictions + projected candidate + projected local
    # summary + log-age.  Keeping this helper in the model module ensures the
    # training and streaming evaluators cannot silently disagree.
    return int(num_horizons * predictor_d_model + 2 * predictor_d_model + 1)


def build_predictive_action_features(
    predictor,
    predictions: torch.Tensor,
    candidate_features: torch.Tensor,
    local_features: torch.Tensor,
    candidate_age: torch.Tensor,
) -> torch.Tensor:
    """Compose action features from frozen predictor outputs.

    Args:
        predictor: frozen ``TemporalJEPAPredictor``.
        predictions: candidate-specific predictions ``[B,P,H,D_in]``.
        candidate_features: evicted candidate latent ``[B,P,D_in]``.
        local_features: current local context ``[B,T,D_in]``.
        candidate_age: age of each candidate ``[B,P]``.
    """
    if predictions.ndim != 4:
        raise ValueError("predictions must have shape [B,P,H,D]")
    batch, proposals, horizons, _ = predictions.shape
    if candidate_features.shape[:2] != (batch, proposals):
        raise ValueError("candidate feature shape is incompatible with predictions")
    if candidate_age.shape != (batch, proposals):
        raise ValueError("candidate_age must have shape [B,P]")
    projected_predictions = predictor.project_inputs(
        predictions.float().reshape(batch * proposals, horizons, -1), detach=True
    ).reshape(batch, proposals, -1)
    projected_candidate = predictor.project_inputs(
        candidate_features.float().reshape(batch * proposals, -1), detach=True
    ).reshape(batch, proposals, -1)
    projected_local = predictor.project_inputs(local_features.float(), detach=True).mean(dim=1)
    projected_local = projected_local.unsqueeze(1).expand(-1, proposals, -1)
    age = torch.log1p(candidate_age.float()).unsqueeze(-1)
    return torch.cat(
        (projected_predictions, projected_candidate, projected_local, age), dim=-1
    )
