from __future__ import annotations

import torch
from torch import nn


class PreNormTransformerLayer(nn.Module):
    """Small explicit encoder layer that can expose final attention weights."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True, bias=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout_ff = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        normalized = self.norm1(hidden)
        attended, weights = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=attention_mask,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        hidden = hidden + self.dropout1(attended)
        normalized = self.norm2(hidden)
        feedforward = self.linear2(self.dropout_ff(self.activation(self.linear1(normalized))))
        hidden = hidden + self.dropout2(feedforward)
        return hidden, weights if need_weights else None

