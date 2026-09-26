from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from .model import TemporalJEPAPredictor


def action_feature_dimension(d_model: int, num_horizons: int) -> int:
    return int(num_horizons * d_model + 2 * d_model + 1)


@torch.no_grad()
def build_action_features(
    predictor: TemporalJEPAPredictor,
    predictions: torch.Tensor,
    evicted_items: torch.Tensor,
    local: torch.Tensor,
    evicted_ages: torch.Tensor,
) -> torch.Tensor:
    """Implement phi_t(a) from the paper for all K+1 actions.

    Args:
        predictions: [actions,horizons,d_in]
        evicted_items: [actions,d_in]
        local: [context,d_in]
        evicted_ages: [actions]
    """
    if predictions.ndim != 3:
        raise ValueError("predictions must be [actions,horizons,d_in]")
    actions, horizons, _ = predictions.shape
    if evicted_items.shape != (actions, predictor.d_in):
        raise ValueError("evicted items do not align with actions")
    if local.ndim != 2 or local.shape[-1] != predictor.d_in:
        raise ValueError("local context has an incompatible shape")
    if evicted_ages.shape != (actions,):
        raise ValueError("evicted ages do not align with actions")
    projected_predictions = predictor.project_inputs(
        predictions.float().reshape(actions * horizons, -1), detach=True
    ).reshape(actions, -1)
    projected_evicted = predictor.project_inputs(evicted_items.float(), detach=True)
    projected_context = predictor.project_inputs(local.float(), detach=True).mean(dim=0)
    projected_context = projected_context.unsqueeze(0).expand(actions, -1)
    age = torch.log1p(evicted_ages.float()).unsqueeze(-1)
    return torch.cat(
        (projected_predictions, projected_evicted, projected_context, age), dim=-1
    )


def multi_horizon_cosine_cost(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    horizon_weights: torch.Tensor,
) -> torch.Tensor:
    """Weighted cosine cost u_t(a); leading dimensions are preserved."""
    if predictions.shape != targets.shape or predictions.ndim < 2:
        raise ValueError("predictions and targets must have matching [...,H,D] shapes")
    if horizon_weights.shape != (predictions.shape[-2],):
        raise ValueError("horizon weights do not align with predictions")
    per_horizon = 1.0 - functional.cosine_similarity(
        predictions.float(), targets.float(), dim=-1
    )
    return (per_horizon * horizon_weights.to(per_horizon)).sum(dim=-1)


def standardized_relative_targets(
    one_step_costs: torch.Tensor,
    rollout_costs: torch.Tensor,
    reservoir_actions: torch.Tensor,
    *,
    one_step_weight: float = 0.5,
    minimum_scale: float = 1e-8,
) -> tuple[torch.Tensor, float, float]:
    """Construct the offline relative target Delta C_t(a,a_R).

    Scales are estimated over the supplied train-only bank. Negative targets
    favor the candidate; retrieval converts them to a positive advantage.
    """
    if one_step_costs.shape != rollout_costs.shape or one_step_costs.ndim != 2:
        raise ValueError("cost tensors must have shape [states,actions]")
    if reservoir_actions.shape != (len(one_step_costs),):
        raise ValueError("one Reservoir action is required per state")
    if not 0.0 <= one_step_weight <= 1.0:
        raise ValueError("one_step_weight must lie in [0,1]")
    reference = reservoir_actions.long().unsqueeze(1)
    short_delta = one_step_costs - one_step_costs.gather(1, reference)
    rollout_delta = rollout_costs - rollout_costs.gather(1, reference)
    short_scale = float(short_delta.reshape(-1).std(unbiased=True).clamp_min(minimum_scale))
    rollout_scale = float(
        rollout_delta.reshape(-1).std(unbiased=True).clamp_min(minimum_scale)
    )
    targets = (
        float(one_step_weight) * short_delta / short_scale
        + (1.0 - float(one_step_weight)) * rollout_delta / rollout_scale
    )
    return targets, short_scale, rollout_scale


@dataclass(frozen=True)
class RetrievalDistribution:
    mean_cost: torch.Tensor
    standard_error: torch.Tensor
    mean_similarity: torch.Tensor

    def conservative_advantage(
        self,
        reference_action: int,
        alternative_action: int,
        *,
        uncertainty_multiplier: float,
    ) -> tuple[float, float, float]:
        """Mean advantage, uncertainty penalty, and lower confidence bound."""
        if uncertainty_multiplier < 0:
            raise ValueError("uncertainty_multiplier must be non-negative")
        mean = float(
            self.mean_cost[reference_action] - self.mean_cost[alternative_action]
        )
        penalty = float(uncertainty_multiplier) * float(
            self.standard_error[reference_action]
            + self.standard_error[alternative_action]
        )
        return mean, penalty, mean - penalty


@dataclass(frozen=True)
class UtilityBank:
    """Train-only FIFO/Reservoir counterfactual feature and target bank."""

    features: torch.Tensor
    relative_costs: torch.Tensor

    def __post_init__(self) -> None:
        if self.features.ndim != 3:
            raise ValueError("features must have shape [states,actions,feature_dim]")
        if self.relative_costs.shape != self.features.shape[:2]:
            raise ValueError("relative costs must align with bank features")
        if len(self.features) == 0:
            raise ValueError("utility bank cannot be empty")

    @staticmethod
    def _normalize(values: torch.Tensor) -> torch.Tensor:
        return values.float() / values.float().norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)

    def retrieve(
        self,
        query_features: torch.Tensor,
        *,
        state_neighbors: int = 32,
        action_neighbors: int = 32,
    ) -> RetrievalDistribution:
        """Two-stage cosine retrieval over states and then candidate actions."""
        if query_features.ndim != 2:
            raise ValueError("query_features must be [actions,feature_dim]")
        if query_features.shape[-1] != self.features.shape[-1]:
            raise ValueError("query and bank feature dimensions differ")
        if state_neighbors <= 0 or action_neighbors <= 0:
            raise ValueError("neighbor counts must be positive")
        query = self._normalize(query_features)
        bank = self._normalize(self.features.to(query_features.device))
        query_state = functional.normalize(query.mean(dim=0), dim=0)
        bank_states = functional.normalize(bank.mean(dim=1), dim=-1)
        state_count = min(int(state_neighbors), len(bank_states))
        selected = (query_state @ bank_states.T).topk(state_count).indices
        candidates = bank[selected].reshape(-1, bank.shape[-1])
        candidate_costs = self.relative_costs.to(query_features.device)[selected].reshape(-1)
        action_count = min(int(action_neighbors), len(candidates))
        nearest = (query @ candidates.T).topk(action_count, dim=-1)
        costs = candidate_costs[nearest.indices]
        if action_count > 1:
            standard_error = costs.std(dim=-1, unbiased=True) / action_count**0.5
        else:
            standard_error = torch.zeros_like(costs.mean(dim=-1))
        return RetrievalDistribution(
            mean_cost=costs.mean(dim=-1),
            standard_error=standard_error,
            mean_similarity=nearest.values.mean(dim=-1),
        )
