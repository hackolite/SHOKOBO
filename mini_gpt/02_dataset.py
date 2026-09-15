"""Chapitre 2 — Dataset autoregressif.

INPUT
ids = [t0, t1, t2, ..., tN]
↓
TRANSFORMATION
fenêtre glissante de taille context_length
↓
OUTPUT
X = [t0, t1, ..., t{T-1}]
Y = [t1, t2, ..., tT]

Idée clé : X[t] -> prédire X[t+1].
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset

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


tokenizer_module = load_local_module('01_tokenizer.py', 'mini_gpt_tokenizer')
SimpleTokenizer = tokenizer_module.SimpleTokenizer


class NextTokenDataset(Dataset):
    def __init__(self, token_ids: list[int], context_length: int):
        if len(token_ids) <= context_length:
            raise ValueError('Le corpus doit être plus long que context_length.')
        self.token_ids = token_ids
        self.context_length = context_length

    def __len__(self) -> int:
        return len(self.token_ids) - self.context_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.token_ids[idx : idx + self.context_length]
        y = self.token_ids[idx + 1 : idx + self.context_length + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def visualize_window(tokenizer: SimpleTokenizer, x: torch.Tensor, y: torch.Tensor) -> None:
    print('Fenêtre glissante:')
    print(f'X ids: {x.tolist()}')
    print(f'Y ids: {y.tolist()}')
    print(f'X texte: {tokenizer.decode(x.tolist())!r}')
    print(f'Y texte: {tokenizer.decode(y.tolist())!r}')
    print('Lecture: chaque position de X doit prédire le token situé juste à droite dans Y.')


def describe_shape(name: str, tensor: torch.Tensor, meaning: str) -> None:
    print(f"{name}: shape={list(tensor.shape)} | {meaning}")


def main() -> None:
    print(f'Using device: {DEVICE}')
    text = CORPUS_PATH.read_text(encoding='utf-8')
    tokenizer = SimpleTokenizer.from_text(text)
    token_ids = tokenizer.encode(text)
    context_length = 16
    batch_size = 4

    dataset = NextTokenDataset(token_ids, context_length=context_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    x0, y0 = dataset[0]
    batch_x, batch_y = next(iter(dataloader))

    print()
    print('=== 1. Construction des exemples ===')
    visualize_window(tokenizer, x0, y0)

    print()
    print('=== 2. Shapes ===')
    describe_shape('x0', x0, '[context_length] = une entrée unique')
    describe_shape('y0', y0, '[context_length] = les cibles décalées de +1')
    describe_shape('batch_x', batch_x, '[batch_size, context_length] = plusieurs fenêtres')
    describe_shape('batch_y', batch_y, '[batch_size, context_length] = plusieurs cibles')

    print()
    print('=== 3. Définitions ===')
    print(f'batch_size = {batch_size}')
    print(f'context_length = {context_length}')
    print(f'nombre de fenêtres = {len(dataset)}')
    print('sequence_length et context_length désignent ici la même longueur T utilisée par le modèle.')

    print()
    print('=== 4. Petit exercice ===')
    print('Exercice: passe context_length de 16 à 8. Que se passe-t-il ?')
    print('Correction: chaque exemple contient moins de contexte, mais le nombre total de fenêtres augmente.')


if __name__ == '__main__':
    main()
