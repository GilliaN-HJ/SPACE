from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


def sinusoidal_encoding(values: torch.Tensor, dimension: int) -> torch.Tensor:
    """Parameter-free encoding used for event ages and prediction horizons."""
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    values = values.float().clamp_min(0).unsqueeze(-1)
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


def predictor_attention_mask(
    memory_slots: int,
    local_tokens: int,
    horizon_queries: int,
    device: torch.device,
) -> torch.Tensor:
    """Boolean attention mask; True entries are forbidden."""
    context = memory_slots + local_tokens
    total = context + horizon_queries
    mask = torch.zeros((total, total), dtype=torch.bool, device=device)
    mask[:context, context:] = True
    if horizon_queries:
        mask[context:, context:] = True
        diagonal = torch.arange(horizon_queries, device=device)
        mask[context + diagonal, context + diagonal] = False
    return mask


class PreNormTransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True, bias=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        # Attribute names intentionally match the released checkpoint.
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout_ff = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.norm1(hidden)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=attention_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        hidden = hidden + self.dropout1(attended)
        normalized = self.norm2(hidden)
        feedforward = self.linear2(
            self.dropout_ff(self.activation(self.linear1(normalized)))
        )
        return hidden + self.dropout2(feedforward)


@dataclass(frozen=True)
class PredictorOutput:
    predictions: torch.Tensor


class TemporalJEPAPredictor(nn.Module):
    """Causal multi-horizon Transformer JEPA used by SPACE.

    Frozen visual features, memory tokens, recent context, token types, and
    sinusoidal ages are the only inputs. The model returns one future latent
    prediction per configured horizon.
    """

    def __init__(
        self,
        *,
        d_in: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        memory_slots: int,
        local_context_len: int,
        horizons: tuple[int, ...] | list[int],
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_in = int(d_in)
        self.d_model = int(d_model)
        self.memory_slots = int(memory_slots)
        self.local_context_len = int(local_context_len)
        self.horizons = tuple(int(x) for x in horizons)
        self.input_projection = nn.Linear(d_in, d_model)
        self.type_embeddings = nn.Parameter(torch.empty(3, d_model))
        self.horizon_queries = nn.Parameter(torch.empty(len(self.horizons), d_model))
        self.layers = nn.ModuleList(
            PreNormTransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.prediction_head = nn.Linear(d_model, d_in)
        nn.init.normal_(self.type_embeddings, std=0.02)
        nn.init.normal_(self.horizon_queries, std=0.02)

    @property
    def num_horizons(self) -> int:
        return len(self.horizons)

    def project_inputs(self, values: torch.Tensor, *, detach: bool = False) -> torch.Tensor:
        projected = self.input_projection(values)
        return projected.detach() if detach else projected

    def forward(
        self,
        local: torch.Tensor,
        memory: torch.Tensor,
        memory_valid: torch.Tensor,
        local_age: torch.Tensor | None = None,
        memory_age: torch.Tensor | None = None,
    ) -> PredictorOutput:
        batch, local_length, d_in = local.shape
        if (local_length, d_in) != (self.local_context_len, self.d_in):
            raise ValueError("local context has an incompatible shape")
        if memory.shape != (batch, self.memory_slots, self.d_in):
            raise ValueError("memory has an incompatible shape")
        if memory_valid.shape != (batch, self.memory_slots):
            raise ValueError("memory_valid has an incompatible shape")
        if local_age is None:
            local_age = torch.arange(
                local_length - 1, -1, -1, device=local.device, dtype=torch.float32
            ).expand(batch, -1)
        if memory_age is None:
            memory_age = torch.zeros(
                (batch, self.memory_slots), device=memory.device, dtype=torch.float32
            )

        memory_hidden = (
            self.input_projection(memory)
            + self.type_embeddings[0]
            + sinusoidal_encoding(memory_age, self.d_model).to(memory.dtype)
        )
        local_hidden = (
            self.input_projection(local)
            + self.type_embeddings[1]
            + sinusoidal_encoding(local_age, self.d_model).to(local.dtype)
        )
        horizon_values = torch.tensor(
            self.horizons, device=local.device, dtype=torch.float32
        )
        query_hidden = (
            self.horizon_queries
            + self.type_embeddings[2]
            + sinusoidal_encoding(horizon_values, self.d_model).to(local.dtype)
        ).unsqueeze(0).expand(batch, -1, -1)
        hidden = torch.cat((memory_hidden, local_hidden, query_hidden), dim=1)

        mask = predictor_attention_mask(
            self.memory_slots, local_length, self.num_horizons, hidden.device
        )
        valid_tail = torch.zeros(
            (batch, local_length + self.num_horizons),
            dtype=torch.bool,
            device=memory_valid.device,
        )
        padding = torch.cat((~memory_valid.bool(), valid_tail), dim=1)
        for layer in self.layers:
            hidden = layer(hidden, attention_mask=mask, key_padding_mask=padding)
        queries = self.final_norm(hidden[:, -self.num_horizons :])
        return PredictorOutput(self.prediction_head(queries))


@torch.no_grad()
def predict_eviction_candidates(
    predictor: TemporalJEPAPredictor,
    candidate_memories: torch.Tensor,
    candidate_ages: torch.Tensor,
    local: torch.Tensor,
    local_age: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate all K+1 candidate memories in one JEPA batch."""
    actions, effective_slots, feature_dim = candidate_memories.shape
    if actions != effective_slots + 1:
        raise ValueError("expected K+1 candidate memories with K valid slots each")
    if predictor.memory_slots not in (effective_slots, effective_slots + 1):
        raise ValueError("predictor physical slots must be K or K+1")
    if local.ndim != 2:
        raise ValueError("local must have shape [context,d_in]")
    local_batch = local.unsqueeze(0).expand(actions, -1, -1)
    if local_age is not None:
        local_age = local_age.unsqueeze(0).expand(actions, -1)
    if predictor.memory_slots == effective_slots + 1:
        # Released checkpoints use K+1 physical inputs so the same JEPA can
        # evaluate the full K+1 set and leave-one-out masks. Deployment still
        # supplies exactly K valid tokens; the final slot is padding.
        candidate_memories = torch.cat(
            (
                candidate_memories,
                torch.zeros(
                    actions,
                    1,
                    feature_dim,
                    dtype=candidate_memories.dtype,
                    device=candidate_memories.device,
                ),
            ),
            dim=1,
        )
        candidate_ages = torch.cat(
            (
                candidate_ages,
                torch.zeros(
                    actions,
                    1,
                    dtype=candidate_ages.dtype,
                    device=candidate_ages.device,
                ),
            ),
            dim=1,
        )
    valid = torch.zeros(
        (actions, predictor.memory_slots),
        dtype=torch.bool,
        device=candidate_memories.device,
    )
    valid[:, :effective_slots] = True
    return predictor(
        local_batch,
        candidate_memories,
        valid,
        local_age=local_age,
        memory_age=candidate_ages,
    ).predictions


def load_predictor_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[TemporalJEPAPredictor, dict[str, Any]]:
    """Load the original SPACE JEPA checkpoint without key conversion."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "config" not in checkpoint or "model" not in checkpoint:
        raise ValueError("checkpoint must contain config and model entries")
    predictor_config = checkpoint["config"]["predictor"]
    predictor = TemporalJEPAPredictor(
        d_in=int(checkpoint["d_in"]),
        d_model=int(predictor_config["d_model"]),
        n_layers=int(predictor_config["n_layers"]),
        n_heads=int(predictor_config["n_heads"]),
        d_ff=int(predictor_config["d_ff"]),
        dropout=float(predictor_config["dropout"]),
        memory_slots=int(predictor_config["memory_budget"]) + 1,
        local_context_len=int(predictor_config["local_context_len"]),
        horizons=tuple(int(x) for x in predictor_config["horizons"]),
    )
    predictor.load_state_dict(checkpoint["model"], strict=True)
    predictor.to(device).eval()
    for parameter in predictor.parameters():
        parameter.requires_grad_(False)
    return predictor, checkpoint
