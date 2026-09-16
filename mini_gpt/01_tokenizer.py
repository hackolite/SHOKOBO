"""Chapitre 1 — Tokenizer.

INPUT
texte brut : str
↓
TRANSFORMATION
caractères -> vocabulaire -> ids
↓
OUTPUT
token_ids : List[int]

Exercices:
1. Modifie le texte d'exemple et observe les nouveaux ids.
2. Ajoute un nouveau caractère dans le corpus. Que devient vocab_size ?

Corrections:
1. Un caractère inconnu devient <unk>.
2. vocab_size augmente si le caractère n'était pas déjà dans le vocabulaire.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import torch

BPE_TRAINING_CHAR_LIMIT = 1_000_000
BPE_CHUNK_SIZE = 4096

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE_DIR = Path(__file__).resolve().parent
CORPUS_PATH = BASE_DIR / 'data' / 'tiny_corpus.txt'


class SimpleTokenizer:
    """Tokenizer caractère par caractère, volontairement simple et transparent."""

    def __init__(self, vocab: Iterable[str]):
        unique_tokens = sorted(set(vocab) - {'<unk>'})
        self.special_tokens = ['<unk>']
        self.itos = self.special_tokens + unique_tokens
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}
        self.unk_id = self.stoi['<unk>']

    @classmethod
    def from_text(cls, text: str) -> 'SimpleTokenizer':
        return cls(vocab=text)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> List[int]:
        return [self.stoi.get(char, self.unk_id) for char in text]

    def decode(self, ids: Iterable[int]) -> str:
        tokens = [self.itos[idx] if 0 <= idx < self.vocab_size else '<unk>' for idx in ids]
        return ''.join('?' if token == '<unk>' else token for token in tokens)

    def token_to_id(self, token: str) -> int:
        return self.stoi.get(token, self.unk_id)

    def id_to_token(self, idx: int) -> str:
        return self.itos[idx] if 0 <= idx < self.vocab_size else '<unk>'

    def to_state(self) -> dict:
        return {'kind': 'char', 'vocab': self.itos[1:]}


def _bpe_library():
    try:
        import tokenizers
    except ImportError as exc:
        raise ImportError(
            'BPE requires the optional dependency: '
            'pip install -r mini_gpt/requirements-bpe.txt'
        ) from exc
    return tokenizers


class BPETokenizer:
    """Byte-level BPE backed by Hugging Face tokenizers, with lossless UTF-8 decoding."""

    def __init__(self, backend):
        self.backend = backend

    @property
    def vocab_size(self) -> int:
        return self.backend.get_vocab_size()

    def encode(self, text: str) -> List[int]:
        return self.backend.encode(text, add_special_tokens=False).ids

    def decode(self, ids: Iterable[int]) -> str:
        return self.backend.decode(list(ids), skip_special_tokens=False)

    def to_state(self) -> dict:
        return {'kind': 'bpe', 'tokenizer_json': self.backend.to_str()}


def tokenizer_from_state(state: dict):
    """Restore without fitting; BPE state contains the library's complete JSON model."""
    if state.get('kind') == 'char':
        return SimpleTokenizer(state['vocab'])
    if state.get('kind') == 'bpe':
        library = _bpe_library()
        return BPETokenizer(library.Tokenizer.from_str(state['tokenizer_json']))
    raise ValueError("Tokenizer state must have kind 'char' or 'bpe'.")


def _bpe_training_chunks(text_iterator: Iterable[str]):
    remaining = BPE_TRAINING_CHAR_LIMIT
    for text in text_iterator:
        for start in range(0, min(len(text), remaining), BPE_CHUNK_SIZE):
            chunk = text[start:start + min(BPE_CHUNK_SIZE, remaining)]
            remaining -= len(chunk)
            yield chunk
            if remaining == 0:
                return


def train_tokenizer(text_iterator: Iterable[str], kind: str = 'char', vocab_size: int = 2000):
    """Fit on training text only, accepting a one-pass iterable of strings.

    Char mode collects the Unicode alphabet and ignores vocab_size. BPE uses
    at most the first 1,000,000 training characters, in pieces of at most 4096,
    bounding the library trainer's word-frequency table even for giant lines.
    Chunk boundaries prohibit cross-chunk merges. Byte-level BPE keeps all 256
    bytes plus <unk>, so its vocabulary can exceed a requested size below 257.
    No normalization or prefix space is added.
    """
    if kind == 'char':
        alphabet = set()
        for text in text_iterator:
            alphabet.update(text)
        if not alphabet:
            raise ValueError('Cannot train a tokenizer on empty training text.')
        return SimpleTokenizer(alphabet)
    if kind != 'bpe':
        raise ValueError("tokenizer kind must be 'char' or 'bpe'.")
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size < 1:
        raise ValueError('vocab_size must be a positive integer.')
    library = _bpe_library()
    backend = library.Tokenizer(library.models.BPE(unk_token='<unk>'))
    backend.pre_tokenizer = library.pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = library.decoders.ByteLevel()
    trainer = library.trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=['<unk>'],
        initial_alphabet=library.pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    chunks = iter(_bpe_training_chunks(text_iterator))
    first = next(chunks, None)
    if first is None:
        raise ValueError('Cannot train a tokenizer on empty training text.')
    from itertools import chain
    backend.train_from_iterator(chain([first], chunks), trainer=trainer)
    return BPETokenizer(backend)


def describe_shape(name: str, tensor: torch.Tensor, meaning: str) -> None:
    print(f"{name}: shape={list(tensor.shape)} | {meaning}")


def load_corpus() -> str:
    return CORPUS_PATH.read_text(encoding='utf-8')


def main() -> None:
    print(f'Using device: {DEVICE}')
    corpus = load_corpus()
    tokenizer = SimpleTokenizer.from_text(corpus)

    print()
    print('=== 1. Texte -> tokens -> ids ===')
    text = 'bonjour'
    token_ids = tokenizer.encode(text)
    reconstructed = tokenizer.decode(token_ids)

    print(f'Texte brut: {text!r}')
    print(f'Tokens (caractères): {list(text)}')
    print(f'Token IDs: {token_ids}')
    print(f'Texte reconstruit: {reconstructed!r}')

    print()
    print('=== 2. Vocabulaire ===')
    print(f'vocab_size = {tokenizer.vocab_size}')
    for token in ['b', 'o', 'n', ' ', '.', 'z']:
        print(f'token -> id : {token!r} -> {tokenizer.token_to_id(token)}')
    for idx in range(min(10, tokenizer.vocab_size)):
        print(f'id -> token : {idx} -> {tokenizer.id_to_token(idx)!r}')

    print()
    print('=== 3. Shapes importantes ===')
    ids_tensor = torch.tensor(token_ids, dtype=torch.long)
    batch_ids = ids_tensor.unsqueeze(0)
    describe_shape('ids_tensor', ids_tensor, '[sequence_length] = une phrase tokenisée')
    describe_shape('batch_ids', batch_ids, '[batch_size, sequence_length] = un mini-batch de 1 phrase')

    print()
    print('=== 4. Intuition ===')
    print('Un tokenizer transforme un texte lisible en nombres entiers manipulables par un réseau.')
    print('Un vrai LLM utilise souvent BPE ou SentencePiece pour compresser des sous-mots fréquents.')
    print('Ici, le niveau caractère est moins performant, mais parfait pour comprendre.')

    print()
    print('=== 5. Petit exercice ===')
    exercise_text = 'mini gpt'
    print(f'Exercice: encode puis decode {exercise_text!r}.')
    print(f'Correction ids: {tokenizer.encode(exercise_text)}')
    print(f'Correction texte: {tokenizer.decode(tokenizer.encode(exercise_text))!r}')


if __name__ == '__main__':
    main()
