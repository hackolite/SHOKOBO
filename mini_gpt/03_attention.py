"""Chapitre 3 — Attention, causal mask et multi-head attention.

Formule centrale:
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def describe_shape(name: str, tensor: torch.Tensor, meaning: str) -> None:
    print(f"{name}: shape={list(tensor.shape)} | {meaning}")


def softmax_manual(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_stable = x - x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x_stable)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def causal_mask(sequence_length: int, device: Optional[str] = None) -> torch.Tensor:
    mask = torch.triu(torch.ones(sequence_length, sequence_length, device=device), diagonal=1)
    return mask.bool()


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1)
    scaled_scores = scores / math.sqrt(d_k)
    if mask is not None:
        scaled_scores = scaled_scores.masked_fill(mask, float('-inf'))
    weights = torch.softmax(scaled_scores, dim=-1)
    output = weights @ v
    return output, weights, scaled_scores


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self, embedding_dim: int, num_heads: int, dropout: float = 0.0,
        position_encoding: str = 'learned', attention_backend: str = 'manual',
    ):
        super().__init__()
        for name, value in (('embedding_dim', embedding_dim), ('num_heads', num_heads)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f'{name} must be a positive integer.')
        if embedding_dim % num_heads != 0:
            raise ValueError('embedding_dim doit être divisible par num_heads.')
        if position_encoding not in ('learned', 'rope'):
            raise ValueError("position_encoding must be 'learned' or 'rope'.")
        if attention_backend not in ('manual', 'sdpa'):
            raise ValueError("attention_backend must be 'manual' or 'sdpa'.")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError('dropout must be between 0 and 1.')
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.position_encoding = position_encoding
        self.attention_backend = attention_backend
        if position_encoding == 'rope' and self.head_dim % 2:
            raise ValueError('RoPE requires an even head_dim.')

        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def _validate_past_key_value(self, past_key_value, batch_size, device, dtype=None):
        if not isinstance(past_key_value, (tuple, list)) or len(past_key_value) != 2:
            raise ValueError('Each cache entry must be a (key, value) pair.')
        k, v = past_key_value
        for tensor in (k, v):
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
                raise ValueError('Cached keys and values must be rank-4 tensors.')
            if (tensor.shape[0] != batch_size or tensor.shape[1] != self.num_heads
                    or tensor.shape[3] != self.head_dim):
                raise ValueError('Cache batch size, number of heads or head dimension is invalid.')
            if tensor.device != device or not tensor.is_floating_point():
                raise ValueError('Cache device or dtype is invalid.')
            if dtype is not None and tensor.dtype != dtype:
                raise ValueError('Cache dtype must match projected keys and values.')
        if k.shape != v.shape or k.dtype != v.dtype:
            raise ValueError('Cached keys and values must have matching shapes and dtypes.')
        return k.shape[2]

    def _apply_rope(self, tensor: torch.Tensor, offset: int) -> torch.Tensor:
        # Adjacent feature pairs rotate at different frequencies; cached keys are already rotated.
        frequency = 10000.0 ** (
            -torch.arange(0, self.head_dim, 2, device=tensor.device, dtype=torch.float32)
            / self.head_dim
        )
        positions = torch.arange(
            offset, offset + tensor.shape[2], device=tensor.device, dtype=torch.float32,
        )
        angles = positions[:, None] * frequency[None, :]
        cos, sin = angles.cos().to(tensor.dtype), angles.sin().to(tensor.dtype)
        even, odd = tensor[..., 0::2], tensor[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False,
        past_key_value=None, use_cache: bool = False,
    ):
        """Cache tensors have shape [batch, heads, past_tokens, head_dim]."""
        if self.training and (use_cache or past_key_value is not None):
            raise ValueError('KV caching is inference-only; call eval() first.')
        batch_size, sequence_length, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)

        past_length = 0
        if past_key_value is not None:
            past_length = self._validate_past_key_value(
                past_key_value, batch_size, k.device, k.dtype,
            )
        if self.position_encoding == 'rope':
            q = self._apply_rope(q, past_length)
            k = self._apply_rope(k, past_length)
        if past_key_value is not None:
            k = torch.cat((past_key_value[0], k), dim=2)
            v = torch.cat((past_key_value[1], v), dim=2)

        mask = None
        if past_length:
            query_positions = torch.arange(sequence_length, device=x.device) + past_length
            key_positions = torch.arange(k.shape[2], device=x.device)
            mask = key_positions[None, :] > query_positions[:, None]
        output = None
        weights = None
        if (self.attention_backend == 'sdpa' and not return_attention
                and hasattr(F, 'scaled_dot_product_attention')):
            try:
                # SDPA dispatches to its CPU math implementation when fused kernels are unavailable.
                # Dropout remains on the merged output, not on attention probabilities.
                output = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=None if mask is None else ~mask,
                    dropout_p=0.0, is_causal=mask is None,
                )
            except (RuntimeError, NotImplementedError):
                if x.device.type != 'cpu':
                    raise
        if output is None:
            if mask is None:
                mask = causal_mask(sequence_length, device=x.device)
            output, weights, _ = scaled_dot_product_attention(q, k, v, mask=mask)
        output = output.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.embedding_dim)
        output = self.out_proj(self.dropout(output))

        if use_cache:
            present_key_value = (k, v)
            if return_attention:
                return output, weights, present_key_value
            return output, present_key_value
        if return_attention:
            return output, weights
        return output


def tiny_numeric_example() -> None:
    print()
    print('=== 1. Exemple numérique minuscule ===')
    q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    k = torch.tensor([[[1.0, 1.0], [1.0, 0.0]]])
    v = torch.tensor([[[10.0, 0.0], [0.0, 5.0]]])

    output, weights, scaled_scores = scaled_dot_product_attention(q, k, v)
    print('Q =', q)
    print('K =', k)
    print('V =', v)
    print('QK^T / sqrt(d_k) =', scaled_scores)
    print('softmax(scores) =', weights)
    print('attention output =', output)


def pytorch_attention_example() -> None:
    print()
    print('=== 2. Exemple PyTorch avec batch=1, sequence_length=4, embedding_dim=8 ===')
    x = torch.arange(32, dtype=torch.float32, device=DEVICE).view(1, 4, 8) / 10.0
    wq = torch.randn(8, 8, device=DEVICE) * 0.2
    wk = torch.randn(8, 8, device=DEVICE) * 0.2
    wv = torch.randn(8, 8, device=DEVICE) * 0.2

    q = x @ wq
    k = x @ wk
    v = x @ wv
    mask = causal_mask(sequence_length=4, device=DEVICE).unsqueeze(0)
    output, weights, scaled_scores = scaled_dot_product_attention(q, k, v, mask=mask)

    describe_shape('X', x, '[batch, tokens, embedding_dim]')
    describe_shape('Q', q, '[batch, tokens, d_k]')
    describe_shape('K', k, '[batch, tokens, d_k]')
    describe_shape('V', v, '[batch, tokens, d_v]')
    describe_shape('scores', scaled_scores, '[batch, tokens, tokens]')
    describe_shape('attention_weights', weights, '[batch, tokens, tokens]')
    describe_shape('output', output, '[batch, tokens, d_v]')

    print('Masque causal (True = case interdite):')
    print(mask[0].int())
    print('Pourquoi -inf avant le softmax ? Parce que softmax(-inf) = 0, donc un token futur reçoit un poids nul.')


def multi_head_example() -> None:
    print()
    print('=== 3. Multi-head attention ===')
    x = torch.randn(2, 6, 8, device=DEVICE)
    mha = MultiHeadSelfAttention(embedding_dim=8, num_heads=4, dropout=0.0).to(DEVICE)
    output, weights = mha(x, return_attention=True)

    describe_shape('input x', x, '[batch, sequence_length, embedding_dim]')
    describe_shape('attention weights', weights, '[batch, heads, query_positions, key_positions]')
    describe_shape('output', output, '[batch, sequence_length, embedding_dim]')
    print('Intuition: chaque tête peut apprendre une relation différente entre les tokens.')
    print('Exercice: compare num_heads=1 et num_heads=4. La contrainte est embedding_dim % num_heads == 0.')


def main() -> None:
    print(f'Using device: {DEVICE}')
    tiny_numeric_example()
    pytorch_attention_example()
    multi_head_example()


if __name__ == '__main__':
    main()
