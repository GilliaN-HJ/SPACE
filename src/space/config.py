from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PredictorConfig:
    d_in: int = 2048
    d_model: int = 512
    n_layers: int = 4
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    local_context_len: int = 8


@dataclass(frozen=True)
class SlowBasisConfig:
    rank: int = 32
    short_lag: int = 1
    long_lag: int = 32
    regularization: float = 1e-3


@dataclass(frozen=True)
class UtilityConfig:
    rollout_steps: int = 32
    one_step_weight: float = 0.5
    state_neighbors: int = 32
    action_neighbors: int = 32
    uncertainty_multiplier: float = 1.0


@dataclass(frozen=True)
class ControllerConfig:
    alpha_slow: float = 0.05
    alpha_boundary: float = 1.0
    tau_utility: float = 0.05
    tau_support: float = 0.5
    tau_departure: float = 1.0
    departure_temperature: float = 0.5
    tau_gate: float = 0.5
    tau_reachability: float = 3.0
    history_window: int = 32
    warmup: int = 16
    minimum_scale: float = 1e-5


@dataclass(frozen=True)
class SpaceConfig:
    capacity: int = 16
    horizons: tuple[int, ...] = (1, 4, 16, 64)
    horizon_weights: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    slow_basis: SlowBasisConfig = field(default_factory=SlowBasisConfig)
    utility: UtilityConfig = field(default_factory=UtilityConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if not self.horizons or any(h <= 0 for h in self.horizons):
            raise ValueError("prediction horizons must be positive")
        if len(self.horizons) != len(self.horizon_weights):
            raise ValueError("horizons and weights must have equal length")
        if abs(sum(self.horizon_weights) - 1.0) > 1e-6:
            raise ValueError("horizon weights must sum to one")
        if self.predictor.n_heads <= 0 or self.predictor.d_model % self.predictor.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0.0 <= self.utility.one_step_weight <= 1.0:
            raise ValueError("one_step_weight must lie in [0,1]")


def _construct(data: dict[str, Any]) -> SpaceConfig:
    return SpaceConfig(
        capacity=int(data["capacity"]),
        horizons=tuple(int(x) for x in data["horizons"]),
        horizon_weights=tuple(float(x) for x in data["horizon_weights"]),
        predictor=PredictorConfig(**data["predictor"]),
        slow_basis=SlowBasisConfig(**data["slow_basis"]),
        utility=UtilityConfig(**data["utility"]),
        controller=ControllerConfig(**data["controller"]),
    )


def load_config(path: str | Path) -> SpaceConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("configuration must be a mapping")
    return _construct(payload)
