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
import json
import math
import operator
import os
from pathlib import Path
from typing import Tuple

import numpy as np
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
train_tokenizer = tokenizer_module.train_tokenizer
tokenizer_from_state = tokenizer_module.tokenizer_from_state
TEXT_CHUNK_SIZE = 4096


def _text_chunks(path: Path, start: int = 0, stop: int | None = None):
    """Read bounded Unicode chunks; offsets count characters, not UTF-8 bytes."""
    with path.open('r', encoding='utf-8', newline='') as source:
        position = 0
        while stop is None or position < stop:
            chunk = source.read(TEXT_CHUNK_SIZE if stop is None else min(TEXT_CHUNK_SIZE, stop - position))
            if not chunk:
                break
            end = position + len(chunk)
            if end > start:
                yield chunk[max(0, start - position):]
            position = end


def prepare_corpus(
    corpus_path,
    output_dir,
    tokenizer_kind: str = 'char',
    vocab_size: int = 2000,
    validation_fraction: float = 0.1,
):
    """Return (tokenizer, train.bin Path, val.bin Path), fitting on train only.

    Three bounded streaming passes count raw Unicode characters, fit on the
    contiguous training prefix, then encode the two disjoint splits. The train
    length is floor(N * (1 - validation_fraction)); no newline normalization is
    performed. Each split needs at least two raw characters AND encoded tokens.

    Native numpy int64 IDs are persisted without a header; tokenizer.json stores
    to_state() and can be restored with tokenizer_from_state(json.load(...)).
    Encoding uses chunks of at most TEXT_CHUNK_SIZE characters; BPE cannot merge
    across chunks, so IDs may differ from encoding the whole split at once.
    Decoding concatenated BPE IDs still reconstructs the original split.
    BPE fitting uses a bounded training prefix (see train_tokenizer). Memory is
    bounded by chunk size, Unicode alphabet, and that BPE training budget, not
    corpus size or line length. The source must remain unchanged during passes.
    """
    if not math.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
        raise ValueError('validation_fraction must be finite and strictly between 0 and 1.')
    if tokenizer_kind not in ('char', 'bpe'):
        raise ValueError("tokenizer_kind must be 'char' or 'bpe'.")
    corpus_path = Path(corpus_path)
    output_dir = Path(output_dir)
    paths = [output_dir / name for name in ('train.bin', 'val.bin', 'tokenizer.json')]
    staged = [path.with_name(path.name + '.partial') for path in paths]
    if any(path.resolve() == corpus_path.resolve() for path in paths + staged):
        raise ValueError('The corpus path must not be an output artifact path.')
    total = sum(len(chunk) for chunk in _text_chunks(corpus_path))
    train_length = int(total * (1 - validation_fraction))
    if train_length < 2 or total - train_length < 2:
        raise ValueError(
            'Training and validation splits must each contain at least two characters; '
            'use a larger corpus or adjust validation_fraction.'
        )
    tokenizer = train_tokenizer(
        _text_chunks(corpus_path, stop=train_length), kind=tokenizer_kind, vocab_size=vocab_size,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Encode in a single pass, cutting a chunk exactly at the raw split boundary.
        counts = [0, 0]
        with staged[0].open('wb') as train_file, staged[1].open('wb') as val_file:
            position = 0
            for chunk in _text_chunks(corpus_path):
                cut = max(0, min(len(chunk), train_length - position))
                for index, (text, target) in enumerate(
                    ((chunk[:cut], train_file), (chunk[cut:], val_file))
                ):
                    if text:
                        ids = np.asarray(tokenizer.encode(text), dtype=np.int64)
                        ids.tofile(target)
                        counts[index] += len(ids)
                position += len(chunk)
        for name, count in zip(('Training', 'Validation'), counts):
            if count < 2:
                raise ValueError(f'{name} split has only {count} tokens; at least two are required.')
        staged[2].write_text(json.dumps(tokenizer.to_state(), ensure_ascii=False), encoding='utf-8')
        for source, target in zip(staged, paths):
            source.replace(target)
    finally:
        for path in staged:
            path.unlink(missing_ok=True)
    return tokenizer, paths[0], paths[1]


class MemmapTokenDataset(Dataset):
    """Lazy read-only int64 mapping; window starts are computed, never materialized.

    Each process opens its own mapping on first access. Returned tensors copy
    only one window and may be safely mutated. Files must not change in use.
    """

    def __init__(self, path, context_length: int, stride: int = 1):
        for name, value in (('context_length', context_length), ('stride', stride)):
            if isinstance(value, bool):
                raise ValueError(f'{name} must be a positive integer.')
            try:
                value = operator.index(value)
            except TypeError as exc:
                raise ValueError(f'{name} must be a positive integer.') from exc
            if value < 1:
                raise ValueError(f'{name} must be a positive integer.')
            setattr(self, name, value)
        self.path = Path(path).resolve()
        size = self.path.stat().st_size
        if size % np.dtype(np.int64).itemsize:
            raise ValueError('Token file size must be a multiple of eight bytes (numpy int64).')
        self.token_count = size // np.dtype(np.int64).itemsize
        if self.token_count <= self.context_length:
            raise ValueError('Token file must contain at least context_length + 1 tokens.')
        self._tokens = None
        self._pid = None

    def __len__(self) -> int:
        return (self.token_count - self.context_length - 1) // self.stride + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = operator.index(idx)
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError('Token window index out of range.')
        if self._tokens is None or self._pid != os.getpid():
            self._tokens = np.memmap(self.path, dtype=np.int64, mode='r')
            self._pid = os.getpid()
        start = idx * self.stride
        x = self._tokens[start:start + self.context_length].copy()
        y = self._tokens[start + 1:start + self.context_length + 1].copy()
        return torch.from_numpy(x), torch.from_numpy(y)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_tokens'] = None
        state['_pid'] = None
        return state


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
