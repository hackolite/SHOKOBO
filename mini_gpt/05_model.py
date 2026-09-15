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
from typing import Optional, Tuple

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
    ):
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(context_length, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, sequence_length = x.shape
        if sequence_length > self.context_length:
            raise ValueError('sequence_length dépasse context_length.')

        positions = torch.arange(sequence_length, device=x.device).unsqueeze(0)
        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)
        h = self.dropout(tok_emb + pos_emb)

        if verbose:
            describe_shape('token ids', x, '[batch_size, sequence_length]')
            describe_shape('token embeddings', tok_emb, '[batch_size, sequence_length, embedding_dim]')
            describe_shape('position embeddings', pos_emb, '[1, sequence_length, embedding_dim]')
            describe_shape('combined embeddings', h, '[batch_size, sequence_length, embedding_dim]')

        for i, block in enumerate(self.blocks, start=1):
            h = block(h)
            if verbose:
                describe_shape(f'after transformer block {i}', h, '[batch_size, sequence_length, embedding_dim]')

        logits = self.lm_head(self.final_norm(h))
        if verbose:
            describe_shape('logits', logits, '[batch_size, sequence_length, vocab_size]')
            print('Les logits sont des scores bruts: plus un logit est élevé, plus le token est jugé probable.')

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
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
