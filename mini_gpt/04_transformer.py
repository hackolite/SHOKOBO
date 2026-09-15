"""Chapitre 4 — Transformer Block.

x
↓
Self Attention
↓
Residual Connection + LayerNorm
↓
Feed Forward
↓
Residual Connection + LayerNorm
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import torch.nn as nn

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE_DIR = Path(__file__).resolve().parent


def load_local_module(filename: str, module_name: str):
    path = BASE_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


attention_module = load_local_module('03_attention.py', 'mini_gpt_attention')
MultiHeadSelfAttention = attention_module.MultiHeadSelfAttention


def describe_shape(name: str, tensor: torch.Tensor, meaning: str) -> None:
    print(f"{name}: shape={list(tensor.shape)} | {meaning}")


class FeedForward(nn.Module):
    def __init__(self, embedding_dim: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError('embedding_dim doit être divisible par num_heads.')
        self.attention = MultiHeadSelfAttention(embedding_dim, num_heads, dropout=dropout)
        self.norm_1 = nn.LayerNorm(embedding_dim)
        self.feed_forward = FeedForward(embedding_dim, ffn_dim, dropout=dropout)
        self.norm_2 = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        attn_output = self.attention(x)
        x = self.norm_1(x + attn_output)
        if verbose:
            describe_shape('after attention + residual + norm', x, '[batch, tokens, embedding_dim]')
        ffn_output = self.feed_forward(x)
        x = self.norm_2(x + ffn_output)
        if verbose:
            describe_shape('after ffn + residual + norm', x, '[batch, tokens, embedding_dim]')
        return x


def main() -> None:
    print(f'Using device: {DEVICE}')
    embedding_dim = 128
    num_heads = 4
    head_dim = embedding_dim // num_heads
    ffn_dim = 512

    print('d_model = embedding_dim =', embedding_dim)
    print('num_heads =', num_heads)
    print('head_dim = d_model / num_heads =', head_dim)
    print('ffn_dim =', ffn_dim)

    x = torch.randn(2, 16, embedding_dim, device=DEVICE)
    block = TransformerBlock(embedding_dim, num_heads, ffn_dim, dropout=0.1).to(DEVICE)

    describe_shape('x', x, '[batch_size, sequence_length, embedding_dim]')
    y = block(x, verbose=True)
    describe_shape('y', y, '[batch_size, sequence_length, embedding_dim]')

    print('Pourquoi les connexions résiduelles ? Elles aident le gradient à circuler dans les réseaux profonds.')
    print('Pourquoi LayerNorm ? Elle stabilise les activations token par token.')
    print('Exercice: passe embedding_dim=64 et num_heads=8. Correction: 64 % 8 == 0, donc cela fonctionne.')


if __name__ == '__main__':
    main()
