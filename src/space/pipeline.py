from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import SpaceConfig
from .controller import SpaceController, SpaceDecision
from .memory import (
    MemoryState,
    counter_reservoir_action,
    enumerate_evictions,
    execute_action,
    stable_video_seed,
)
from .model import TemporalJEPAPredictor, predict_eviction_candidates
from .slow_basis import (
    SlowPredictiveBasis,
    predictive_state_signatures,
)
from .utility import UtilityBank, build_action_features


@dataclass(frozen=True)
class SpaceStep:
    memory: MemoryState
    decision: SpaceDecision | None
    predictions: torch.Tensor | None
    signatures: torch.Tensor | None


class SPACE:
    """Dataset-independent online SPACE memory controller.

    The encoder, predictor, bases, and bank are frozen. Only memory and the
    causal basin state change at deployment.
    """

    def __init__(
        self,
        *,
        config: SpaceConfig,
        predictor: TemporalJEPAPredictor,
        slow_predictive_basis: SlowPredictiveBasis,
        utility_bank: UtilityBank,
        policy_seed: int = 0,
        device: torch.device | str = "cpu",
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.predictor = predictor.to(self.device).eval()
        for parameter in self.predictor.parameters():
            parameter.requires_grad_(False)
        if predictor.memory_slots not in (config.capacity, config.capacity + 1):
            raise ValueError("predictor physical slots must equal K or K+1")
        if predictor.horizons != config.horizons:
            raise ValueError("predictor horizons and SPACE configuration differ")
        if slow_predictive_basis.input_dimension != predictor.d_model:
            raise ValueError("slow predictive basis must operate in JEPA hidden space")
        self.slow_predictive_basis = slow_predictive_basis
        self.utility_bank = utility_bank
        self.policy_seed = int(policy_seed)
        self.controller = SpaceController(config.controller)
        self.memory = MemoryState.empty(config.capacity, predictor.d_in, self.device)
        self.seen_events = 0

    @torch.no_grad()
    def step(
        self,
        event: torch.Tensor,
        *,
        event_time: int,
        current_time: int,
        local_context: torch.Tensor,
        video_id: str = "stream",
    ) -> SpaceStep:
        """Consume one eligible event and update fixed-capacity memory."""
        event = event.to(self.device).float()
        local_context = local_context.to(self.device).float()
        if event.shape != (self.predictor.d_in,):
            raise ValueError("event has an incompatible feature dimension")
        if local_context.shape != (
            self.predictor.local_context_len,
            self.predictor.d_in,
        ):
            raise ValueError("local_context has an incompatible shape")
        event_id = self.seen_events
        self.seen_events += 1
        if not self.memory.full:
            self.memory = self.memory.insert_first_free(event, event_time, event_id)
            return SpaceStep(self.memory.clone(), None, None, None)

        candidates = enumerate_evictions(self.memory, event, event_time, event_id)
        evicted_ages, memory_ages = candidates.ages(current_time)
        reservoir_action = counter_reservoir_action(
            candidates,
            current_time=current_time,
            base_seed=stable_video_seed(video_id, self.policy_seed),
        )
        predictions = predict_eviction_candidates(
            self.predictor,
            candidates.memories,
            memory_ages,
            local_context,
        )
        features = build_action_features(
            self.predictor,
            predictions,
            candidates.items,
            local_context,
            evicted_ages,
        )
        distribution = self.utility_bank.retrieve(
            features,
            state_neighbors=self.config.utility.state_neighbors,
            action_neighbors=self.config.utility.action_neighbors,
        )
        projected_predictions = self.predictor.project_inputs(
            predictions.float(), detach=True
        )
        signatures = predictive_state_signatures(
            projected_predictions,
            self.slow_predictive_basis,
        )

        decision = self.controller.decide(
            signatures,
            distribution,
            reservoir_action=reservoir_action,
            uncertainty_multiplier=self.config.utility.uncertainty_multiplier,
        )
        self.memory = execute_action(candidates, decision.action)
        self.controller.commit(signatures, decision)
        return SpaceStep(
            self.memory.clone(),
            decision,
            predictions.detach().clone(),
            signatures.detach().clone(),
        )
