from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional


@dataclass(frozen=True)
class MetastableSubspace:
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
        if values.shape[-1] != self.input_dimension:
            raise ValueError(
                f"Expected last dimension {self.input_dimension}, got {values.shape[-1]}"
            )
        normalized = functional.normalize(values.float(), dim=-1)
        projected = normalized @ self.basis.to(normalized.device)
        return functional.normalize(projected, dim=-1)


def _weighted_difference_covariance(
    sequences: list[torch.Tensor], lag: int
) -> torch.Tensor:
    if lag <= 0:
        raise ValueError("lag must be positive")
    dimension = int(sequences[0].shape[-1])
    device = sequences[0].device
    covariance = torch.zeros(
        dimension, dimension, dtype=torch.float64, device=device
    )
    contributors = 0
    for sequence in sequences:
        if sequence.ndim != 2 or sequence.shape[-1] != dimension:
            raise ValueError("All sequences must be [time, dimension]")
        if sequence.device != device:
            raise ValueError("All sequences must be on the same device")
        if len(sequence) <= lag:
            continue
        normalized = functional.normalize(sequence.double(), dim=-1)
        differences = normalized[lag:] - normalized[:-lag]
        covariance += differences.T @ differences / len(differences)
        contributors += 1
    if not contributors:
        raise ValueError(f"No sequence contains a valid lag-{lag} pair")
    return covariance / contributors


def _participant_difference_covariance(
    sequences_by_participant: list[list[torch.Tensor]], lag: int
) -> torch.Tensor:
    if not sequences_by_participant or not sequences_by_participant[0]:
        raise ValueError("At least one participant sequence is required")
    first = sequences_by_participant[0][0]
    dimension, device = int(first.shape[-1]), first.device
    population = torch.zeros(
        dimension, dimension, dtype=torch.float64, device=device
    )
    participants = 0
    for sequences in sequences_by_participant:
        participant = torch.zeros_like(population)
        pairs = 0
        for sequence in sequences:
            if sequence.device != device or sequence.shape[-1] != dimension:
                raise ValueError("All sequences must share device and dimension")
            if len(sequence) <= lag:
                continue
            normalized = functional.normalize(sequence.double(), dim=-1)
            differences = normalized[lag:] - normalized[:-lag]
            participant += differences.T @ differences
            pairs += len(differences)
        if pairs:
            population += participant / pairs
            participants += 1
    if not participants:
        raise ValueError(f"No participant contains a valid lag-{lag} pair")
    return population / participants


def fit_metastable_subspace(
    sequences: list[torch.Tensor] | list[list[torch.Tensor]],
    *,
    short_lag: int = 1,
    long_lag: int = 32,
    rank: int = 32,
    regularization: float = 1e-3,
) -> MetastableSubspace:
    """Fit directions with high long-lag and low short-lag change energy."""
    if not sequences:
        raise ValueError("At least one train sequence is required")
    if long_lag <= short_lag:
        raise ValueError("long_lag must exceed short_lag")
    first = sequences[0][0] if isinstance(sequences[0], list) else sequences[0]
    if rank <= 0 or rank > first.shape[-1]:
        raise ValueError("rank must be within the input dimension")
    if regularization <= 0:
        raise ValueError("regularization must be positive")
    participant_grouped = isinstance(sequences[0], list)
    covariance = (
        _participant_difference_covariance if participant_grouped
        else _weighted_difference_covariance
    )
    short_covariance = covariance(sequences, short_lag)
    long_covariance = covariance(sequences, long_lag)
    dimension = short_covariance.shape[0]
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
    basis, _ = torch.linalg.qr(generalized, mode="reduced")
    return MetastableSubspace(
        basis=basis.float(),
        eigenvalues=eigenvalues[ordering].float(),
        short_lag=int(short_lag),
        long_lag=int(long_lag),
        regularization=float(regularization),
    )


def persistent_innovation_score(
    subspace: MetastableSubspace,
    candidate: torch.Tensor,
    evidence: torch.Tensor,
    memory: torch.Tensor,
    memory_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Causal evidence that a delayed candidate explains a new persistent state."""
    if candidate.ndim == 1:
        candidate = candidate.unsqueeze(0)
    if evidence.ndim == 2:
        evidence = evidence.unsqueeze(0)
    if memory.ndim == 2:
        memory = memory.unsqueeze(0)
    if candidate.ndim != 2 or evidence.ndim != 3 or memory.ndim != 3:
        raise ValueError("Expected candidate [B,D], evidence [B,L,D], memory [B,K,D]")
    batch = candidate.shape[0]
    if evidence.shape[0] != batch or memory.shape[0] != batch:
        raise ValueError("Batch dimensions must match")
    if memory_valid is None:
        memory_valid = torch.ones(
            memory.shape[:2], dtype=torch.bool, device=memory.device
        )
    if memory_valid.shape != memory.shape[:2]:
        raise ValueError("memory_valid has incompatible shape")
    candidate_projected = subspace.project(candidate)
    evidence_projected = subspace.project(evidence)
    memory_projected = subspace.project(memory)
    candidate_persistence = torch.einsum(
        "bd,bld->bl", candidate_projected, evidence_projected
    ).median(dim=-1).values
    explanation = torch.einsum(
        "bkd,bld->bkl", memory_projected, evidence_projected
    ).median(dim=-1).values
    explanation = explanation.masked_fill(~memory_valid, -torch.inf).amax(dim=-1)
    explanation = torch.where(
        memory_valid.any(dim=-1), explanation, torch.full_like(explanation, -1.0)
    )
    return candidate_persistence - explanation


def cmc_evict_index(
    persistent_innovation: torch.Tensor,
    estimated_action_loss: torch.Tensor,
    *,
    gate_threshold: float,
    hold_relative_margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return action index and intervention mask; final action is discard-new."""
    if estimated_action_loss.ndim != 2:
        raise ValueError("estimated_action_loss must be [batch, K+1]")
    if persistent_innovation.shape != estimated_action_loss.shape[:1]:
        raise ValueError("persistent_innovation has incompatible shape")
    hold = estimated_action_loss.shape[1] - 1
    best = estimated_action_loss.argmin(dim=-1)
    row = torch.arange(len(best), device=best.device)
    beats_hold = (
        estimated_action_loss[row, best] + float(hold_relative_margin)
        < estimated_action_loss[:, hold]
    )
    gate_open = persistent_innovation > float(gate_threshold)
    replace = gate_open & beats_hold & (best != hold)
    action = torch.where(replace, best, torch.full_like(best, hold))
    return action, replace
