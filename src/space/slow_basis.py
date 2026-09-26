from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional


@dataclass(frozen=True)
class SlowPredictiveBasis:
    basis: torch.Tensor
    eigenvalues: torch.Tensor
    short_lag: int
    long_lag: int
    regularization: float

    @property
    def input_dimension(self) -> int:
        return int(self.basis.shape[0])

    @property
    def rank(self) -> int:
        return int(self.basis.shape[1])

    def project(self, values: torch.Tensor) -> torch.Tensor:
        """Project normalized JEPA features through the shared slow basis."""
        if values.shape[-1] != self.input_dimension:
            raise ValueError("input and slow-basis dimensions differ")
        projected = functional.normalize(values.float(), dim=-1) @ self.basis.to(values.device)
        return functional.normalize(projected, dim=-1)


def _participant_covariance(
    sequences_by_participant: list[list[torch.Tensor]], lag: int
) -> torch.Tensor:
    if not sequences_by_participant or not sequences_by_participant[0]:
        raise ValueError("at least one participant sequence is required")
    first = sequences_by_participant[0][0]
    dimension, device = int(first.shape[-1]), first.device
    population = torch.zeros(dimension, dimension, dtype=torch.float64, device=device)
    participants = 0
    for sequences in sequences_by_participant:
        participant = torch.zeros_like(population)
        pairs = 0
        for sequence in sequences:
            if sequence.ndim != 2 or sequence.shape[-1] != dimension:
                raise ValueError("all sequences must have shape [time,dimension]")
            if sequence.device != device:
                raise ValueError("all sequences must be on the same device")
            if len(sequence) <= lag:
                continue
            normalized = functional.normalize(sequence.double(), dim=-1)
            difference = normalized[lag:] - normalized[:-lag]
            participant += difference.T @ difference
            pairs += len(difference)
        if pairs:
            population += participant / pairs
            participants += 1
    if not participants:
        raise ValueError(f"no participant contains a valid lag-{lag} pair")
    return population / participants


def fit_slow_predictive_basis(
    sequences_by_participant: list[list[torch.Tensor]],
    *,
    short_lag: int = 1,
    long_lag: int = 32,
    rank: int = 32,
    regularization: float = 1e-3,
) -> SlowPredictiveBasis:
    """Fit the decreasing-eigenvalue generalized slow predictive basis.

    Each participant receives equal covariance mass. To share one basis across
    horizons, pass each participant's per-horizon traces as separate [T,D]
    sequences in that participant's inner list.
    """
    if long_lag <= short_lag:
        raise ValueError("long_lag must exceed short_lag")
    if regularization <= 0:
        raise ValueError("regularization must be positive")
    dimension = int(sequences_by_participant[0][0].shape[-1])
    if not 0 < rank <= dimension:
        raise ValueError("rank must lie in the input dimension")
    short_covariance = _participant_covariance(sequences_by_participant, short_lag)
    long_covariance = _participant_covariance(sequences_by_participant, long_lag)

    # Reference experiments use a scale-aware covariance ridge.
    scale = short_covariance.diagonal().mean().clamp_min(1e-12)
    metric = short_covariance + regularization * scale * torch.eye(
        dimension, dtype=torch.float64, device=short_covariance.device
    )
    metric_values, metric_vectors = torch.linalg.eigh(metric)
    inverse_sqrt = metric_vectors @ torch.diag(
        metric_values.clamp_min(1e-12).rsqrt()
    ) @ metric_vectors.T
    whitened = inverse_sqrt @ long_covariance @ inverse_sqrt
    eigenvalues, eigenvectors = torch.linalg.eigh(whitened)
    ordering = torch.argsort(eigenvalues, descending=True)[:rank]
    generalized = inverse_sqrt @ eigenvectors[:, ordering]
    # Orthonormalization leaves the selected subspace unchanged and makes
    # cosine geometry numerically stable.
    basis, _ = torch.linalg.qr(generalized, mode="reduced")
    return SlowPredictiveBasis(
        basis=basis.float(),
        eigenvalues=eigenvalues[ordering].float(),
        short_lag=int(short_lag),
        long_lag=int(long_lag),
        regularization=float(regularization),
    )


def predictive_state_signatures(
    projected_predictions: torch.Tensor,
    basis: SlowPredictiveBasis,
) -> torch.Tensor:
    """Return normalized action-conditioned states [actions,H*rank]."""
    if projected_predictions.ndim != 3:
        raise ValueError("projected predictions must be [actions,horizons,d_model]")
    projected = basis.project(projected_predictions)
    return functional.normalize(projected.flatten(1), dim=-1)
