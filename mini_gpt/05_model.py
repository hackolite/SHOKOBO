"""Chapitre 5 — MiniGPT complet.

Token IDs
↓
Token Embedding + Position Embedding
↓
Transformer Blocks
↓
Final LayerNorm
↓
LM Head
↓
Logits
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE_DIR = Path(__file__).resolve().parent
CORPUS_PATH = BASE_DIR / 'data' / 'tiny_corpus.txt'


def load_local_module(filename: str, module_name: str):
    path = BASE_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


transformer_module = load_local_module('04_transformer.py', 'mini_gpt_transformer')
tokenizer_module = load_local_module('01_tokenizer.py', 'mini_gpt_tokenizer')
TransformerBlock = transformer_module.TransformerBlock
SimpleTokenizer = tokenizer_module.SimpleTokenizer


def describe_shape(name: str, tensor: torch.Tensor, meaning: str) -> None:
    print(f"{name}: shape={list(tensor.shape)} | {meaning}")


class MiniGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        embedding_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        position_encoding: str = 'learned',
        attention_backend: str = 'manual',
    ):
        super().__init__()
        for name, value in (
            ('vocab_size', vocab_size), ('context_length', context_length),
            ('embedding_dim', embedding_dim), ('num_heads', num_heads),
            ('num_layers', num_layers), ('ffn_dim', ffn_dim),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f'{name} must be a positive integer.')
        if embedding_dim % num_heads:
            raise ValueError('embedding_dim doit être divisible par num_heads.')
        if position_encoding not in ('learned', 'rope'):
            raise ValueError("position_encoding must be 'learned' or 'rope'.")
        if attention_backend not in ('manual', 'sdpa'):
            raise ValueError("attention_backend must be 'manual' or 'sdpa'.")
        if position_encoding == 'rope' and (embedding_dim // num_heads) % 2:
            raise ValueError('RoPE requires an even head_dim.')
        if not 0.0 <= dropout <= 1.0:
            raise ValueError('dropout must be between 0 and 1.')
        self.context_length = context_length
        self.position_encoding = position_encoding
        self.attention_backend = attention_backend
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = (
            nn.Embedding(context_length, embedding_dim) if position_encoding == 'learned' else None
        )
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                    position_encoding=position_encoding,
                    attention_backend=attention_backend,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        verbose: bool = False,
        past_key_values=None,
        use_cache: bool = False,
    ):
        """Return (logits, loss), plus per-layer (key, value) tuples when use_cache=True.

        Caches may only be used in eval mode without targets. Pass only new tokens
        with a cache; cached and new tokens together must fit context_length.
        """
        if (use_cache or past_key_values is not None) and (self.training or targets is not None):
            raise ValueError('KV caching requires eval() mode and no targets.')
        if x.ndim != 2 or x.shape[1] == 0:
            raise ValueError('x must have shape [batch, nonempty sequence].')
        batch_size, sequence_length = x.shape
        past_length = 0
        if past_key_values is not None:
            if (not isinstance(past_key_values, (tuple, list))
                    or len(past_key_values) != len(self.blocks)):
                raise ValueError('past_key_values must contain one (key, value) pair per layer.')
            lengths = [
                block.attention._validate_past_key_value(cache, batch_size, x.device)
                for block, cache in zip(self.blocks, past_key_values)
            ]
            if len(set(lengths)) != 1:
                raise ValueError('All cache layers must have the same sequence length.')
            past_length = lengths[0]
        if past_length + sequence_length > self.context_length:
            raise ValueError('Cached plus new sequence_length dépasse context_length.')

        tok_emb = self.token_embedding(x)
        pos_emb = None
        if self.position_embedding is not None:
            positions = torch.arange(
                past_length, past_length + sequence_length, device=x.device,
            ).unsqueeze(0)
            pos_emb = self.position_embedding(positions)
        h = self.dropout(tok_emb if pos_emb is None else tok_emb + pos_emb)

        if verbose:
            describe_shape('token ids', x, '[batch_size, sequence_length]')
            describe_shape('token embeddings', tok_emb, '[batch_size, sequence_length, embedding_dim]')
            if pos_emb is not None:
                describe_shape('position embeddings', pos_emb, '[1, sequence_length, embedding_dim]')
            describe_shape('combined embeddings', h, '[batch_size, sequence_length, embedding_dim]')

        present_key_values = []
        for i, block in enumerate(self.blocks, start=1):
            past = None if past_key_values is None else past_key_values[i - 1]
            h = block(h, past_key_value=past, use_cache=use_cache)
            if use_cache:
                h, present = h
                present_key_values.append(present)
            if verbose:
                describe_shape(f'after transformer block {i}', h, '[batch_size, sequence_length, embedding_dim]')

        logits = self.lm_head(self.final_norm(h))
        if verbose:
            describe_shape('logits', logits, '[batch_size, sequence_length, vocab_size]')
            print('Les logits sont des scores bruts: plus un logit est élevé, plus le token est jugé probable.')

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        if use_cache:
            return logits, loss, tuple(present_key_values)
        return logits, loss

    def count_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters())


def main() -> None:
    print(f'Using device: {DEVICE}')
    text = CORPUS_PATH.read_text(encoding='utf-8')
    tokenizer = SimpleTokenizer.from_text(text)
    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        context_length=16,
        embedding_dim=128,
        num_heads=4,
        num_layers=2,
        ffn_dim=512,
        dropout=0.1,
    ).to(DEVICE)

    sample_ids = tokenizer.encode('bonjour le monde')[:16]
    x = torch.tensor([sample_ids], dtype=torch.long, device=DEVICE)
    logits, _ = model(x, verbose=True)

    print(f'Nombre total de paramètres: {model.count_parameters():,}')
    print('Initialisation des poids: loi normale simple N(0, 0.02), suffisante pour ce mini-projet pédagogique.')
    print('Exercice: augmente num_layers de 2 à 4. Le modèle devient plus profond et contient plus de paramètres.')
    describe_shape('final logits', logits, '[1, sequence_length, vocab_size]')


if __name__ == '__main__':
    main()
