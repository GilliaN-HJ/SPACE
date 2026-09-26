from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as functional

from .config import ControllerConfig
from .utility import RetrievalDistribution


@dataclass
class BasinState:
    prototype: torch.Tensor | None = None
    realized_distance_history: list[float] = field(default_factory=list)
    restoration_history: list[float] = field(default_factory=list)

    def clone(self) -> "BasinState":
        return BasinState(
            None if self.prototype is None else self.prototype.detach().clone(),
            list(self.realized_distance_history),
            list(self.restoration_history),
        )

    @staticmethod
    def robust_z(value: float, history: list[float], minimum_scale: float) -> float:
        if not history:
            return 0.0
        values = torch.as_tensor(history, dtype=torch.float64)
        median = float(values.median())
        mad = float((values - median).abs().median())
        scale = max(1.4826 * mad, float(minimum_scale))
        return (float(value) - median) / scale

    def distances(self, signatures: torch.Tensor) -> torch.Tensor:
        if signatures.ndim != 2:
            raise ValueError("signatures must be [actions,state_dim]")
        if self.prototype is None:
            return torch.zeros(len(signatures), device=signatures.device)
        signatures = functional.normalize(signatures.float(), dim=-1)
        prototype = functional.normalize(self.prototype.to(signatures.device).float(), dim=0)
        return (1.0 - signatures @ prototype).clamp(0.0, 2.0)

    def update_prototype(self, signature: torch.Tensor, rate: float) -> None:
        value = functional.normalize(signature.detach().float(), dim=0)
        if self.prototype is None:
            self.prototype = value.clone()
            return
        self.prototype = functional.normalize(
            (1.0 - float(rate)) * self.prototype.to(value.device) + float(rate) * value,
            dim=0,
        ).detach()


@dataclass(frozen=True)
class SpaceDecision:
    action: int
    reservoir_action: int
    projected_action: int
    override: bool
    boundary_release: bool
    departure_gate: float
    restoration_score: float
    restoration_z: float
    reachability_z: float
    utility_lcb: float
    retrieval_support: float
    admissible_actions: tuple[int, ...]
    distances: torch.Tensor


class SpaceController:
    """Utility, departure, and reachability controller from main.tex."""

    def __init__(self, config: ControllerConfig) -> None:
        self.config = config
        self.basin = BasinState()

    @property
    def warmed_up(self) -> bool:
        return (
            self.basin.prototype is not None
            and len(self.basin.realized_distance_history) >= self.config.warmup
            and len(self.basin.restoration_history) >= self.config.warmup
        )

    def decide(
        self,
        signatures: torch.Tensor,
        distribution: RetrievalDistribution,
        *,
        reservoir_action: int,
        uncertainty_multiplier: float,
    ) -> SpaceDecision:
        actions = len(signatures)
        if distribution.mean_cost.shape != (actions,):
            raise ValueError("retrieval distribution and signatures do not align")
        if not 0 <= reservoir_action < actions:
            raise ValueError("Reservoir action lies outside action set")

        distances = self.basin.distances(signatures)
        restoration = max(
            0.0,
            float(distances[reservoir_action] - distances.min()),
        )
        restoration_z = self.basin.robust_z(
            restoration,
            self.basin.restoration_history,
            self.config.minimum_scale,
        )
        if self.warmed_up:
            logit = (restoration_z - self.config.tau_departure) / max(
                self.config.departure_temperature, 1e-12
            )
            departure_gate = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, logit))))
        else:
            departure_gate = 0.0
        admissible = [int(reservoir_action)]
        diagnostics: dict[int, tuple[float, float]] = {
            int(reservoir_action): (0.0, float(distribution.mean_similarity[reservoir_action]))
        }
        for action in range(actions):
            if action == reservoir_action:
                continue
            _, _, lower = distribution.conservative_advantage(
                reservoir_action,
                action,
                uncertainty_multiplier=uncertainty_multiplier,
            )
            support = float(
                min(
                    distribution.mean_similarity[reservoir_action],
                    distribution.mean_similarity[action],
                )
            )
            diagnostics[action] = (lower, support)
            if lower >= self.config.tau_utility and support >= self.config.tau_support:
                admissible.append(action)

        projected = min(admissible, key=lambda a: (float(distances[a]), int(a)))
        reachability_z = self.basin.robust_z(
            float(distances[projected]),
            self.basin.realized_distance_history,
            self.config.minimum_scale,
        )
        boundary = bool(
            self.warmed_up and reachability_z > self.config.tau_reachability
        )
        override = bool(
            self.warmed_up
            and departure_gate >= self.config.tau_gate
            and not boundary
            and projected != reservoir_action
        )
        action = projected if override else int(reservoir_action)
        lcb, support = diagnostics[projected]
        return SpaceDecision(
            action=action,
            reservoir_action=int(reservoir_action),
            projected_action=int(projected),
            override=override,
            boundary_release=boundary,
            departure_gate=float(departure_gate),
            restoration_score=float(restoration),
            restoration_z=float(restoration_z),
            reachability_z=float(reachability_z),
            utility_lcb=float(lcb),
            retrieval_support=float(support),
            admissible_actions=tuple(admissible),
            distances=distances.detach().clone(),
        )

    def commit(self, signatures: torch.Tensor, decision: SpaceDecision) -> None:
        """Update causal statistics only after the action has been fixed."""
        selected_distance = float(decision.distances[decision.action])
        if not decision.boundary_release:
            self.basin.realized_distance_history.append(selected_distance)
            self.basin.restoration_history.append(decision.restoration_score)
            self.basin.realized_distance_history = self.basin.realized_distance_history[
                -self.config.history_window :
            ]
            self.basin.restoration_history = self.basin.restoration_history[
                -self.config.history_window :
            ]
        rate = (
            self.config.alpha_boundary
            if decision.boundary_release
            else self.config.alpha_slow
        )
        self.basin.update_prototype(signatures[decision.action], rate)
