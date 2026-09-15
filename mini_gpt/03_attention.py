"""Chapitre 3 — Attention, causal mask et multi-head attention.

Formule centrale:
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

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
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError('embedding_dim doit être divisible par num_heads.')
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        batch_size, sequence_length, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)

        mask = causal_mask(sequence_length, device=x.device).unsqueeze(0).unsqueeze(0)
        output, weights, _ = scaled_dot_product_attention(q, k, v, mask=mask)
        output = output.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.embedding_dim)
        output = self.out_proj(self.dropout(output))

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
