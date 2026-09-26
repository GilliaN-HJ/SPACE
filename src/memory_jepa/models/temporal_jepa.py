from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .attention_masks import build_key_padding_mask, build_predictor_attention_mask
from .time_encoding import sinusoidal_encoding
from .transformer import PreNormTransformerLayer


@dataclass
class PredictorOutput:
    predictions: torch.Tensor
    memory_attention: torch.Tensor | None = None


class TemporalJEPAPredictor(nn.Module):
    """The system's only module that predicts future latent representations."""

    produces_future_latent = True

    def __init__(
        self,
        d_in: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        memory_slots: int,
        local_context_len: int,
        horizons: list[int] | tuple[int, ...],
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_in = int(d_in)
        self.d_model = int(d_model)
        self.memory_slots = int(memory_slots)
        self.local_context_len = int(local_context_len)
        self.horizons = tuple(int(value) for value in horizons)
        self.input_projection = nn.Linear(d_in, d_model)
        self.type_embeddings = nn.Parameter(torch.empty(3, d_model))
        self.horizon_queries = nn.Parameter(torch.empty(len(self.horizons), d_model))
        self.layers = nn.ModuleList(
            PreNormTransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.prediction_head = nn.Linear(d_model, d_in)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.type_embeddings, std=0.02)
        nn.init.normal_(self.horizon_queries, std=0.02)

    @property
    def num_horizons(self) -> int:
        return len(self.horizons)

    def project_inputs(self, values: torch.Tensor, detach: bool = False) -> torch.Tensor:
        projected = self.input_projection(values)
        return projected.detach() if detach else projected

    def forward(
        self,
        local: torch.Tensor,
        memory: torch.Tensor,
        memory_valid: torch.Tensor,
        local_age: torch.Tensor | None = None,
        memory_age: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> PredictorOutput:
        batch, local_length, d_in = local.shape
        if local_length != self.local_context_len or d_in != self.d_in:
            raise ValueError(
                f"Expected local [B,{self.local_context_len},{self.d_in}], got {tuple(local.shape)}"
            )
        if memory.shape != (batch, self.memory_slots, self.d_in):
            raise ValueError(
                f"Expected memory {(batch, self.memory_slots, self.d_in)}, got {tuple(memory.shape)}"
            )
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
        horizon_values = torch.tensor(self.horizons, device=local.device, dtype=torch.float32)
        query_hidden = (
            self.horizon_queries
            + self.type_embeddings[2]
            + sinusoidal_encoding(horizon_values, self.d_model).to(local.dtype)
        ).unsqueeze(0).expand(batch, -1, -1)
        hidden = torch.cat((memory_hidden, local_hidden, query_hidden), dim=1)

        attention_mask = build_predictor_attention_mask(
            self.memory_slots, local_length, self.num_horizons, hidden.device
        )
        padding_mask = build_key_padding_mask(memory_valid, local_length, self.num_horizons)
        final_attention = None
        for layer_index, layer in enumerate(self.layers):
            hidden, attention = layer(
                hidden,
                attention_mask=attention_mask,
                key_padding_mask=padding_mask,
                need_weights=return_attention and layer_index == len(self.layers) - 1,
            )
            if attention is not None:
                final_attention = attention

        query_states = self.final_norm(hidden[:, -self.num_horizons :])
        predictions = self.prediction_head(query_states)
        memory_attention = None
        if final_attention is not None:
            query_attention = final_attention[:, :, -self.num_horizons :, : self.memory_slots]
            memory_attention = query_attention.mean(dim=(1, 2))
            memory_attention = memory_attention.masked_fill(~memory_valid, 0.0)
        return PredictorOutput(predictions=predictions, memory_attention=memory_attention)

