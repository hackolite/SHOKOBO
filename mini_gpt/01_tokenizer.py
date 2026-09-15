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

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE_DIR = Path(__file__).resolve().parent
CORPUS_PATH = BASE_DIR / 'data' / 'tiny_corpus.txt'


class SimpleTokenizer:
    """Tokenizer caractère par caractère, volontairement simple et transparent."""

    def __init__(self, vocab: Iterable[str]):
        unique_tokens = sorted(set(vocab))
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
