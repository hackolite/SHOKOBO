"""Chapitre 7 — Génération autoregressive.

prompt -> tokenize -> model -> logits du dernier token -> probabilités -> prochain token
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Optional

import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE_DIR = Path(__file__).resolve().parent
CORPUS_PATH = BASE_DIR / 'data' / 'tiny_corpus.txt'
CHECKPOINT_PATH = BASE_DIR / 'checkpoints' / 'mini_gpt.pt'


def load_local_module(filename: str, module_name: str):
    path = BASE_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


tokenizer_module = load_local_module('01_tokenizer.py', 'mini_gpt_tokenizer')
model_module = load_local_module('05_model.py', 'mini_gpt_model')
SimpleTokenizer = tokenizer_module.SimpleTokenizer
MiniGPT = model_module.MiniGPT


def top_k_logits(logits: torch.Tensor, top_k: Optional[int]) -> torch.Tensor:
    if top_k is None or top_k <= 0 or top_k >= logits.size(-1):
        return logits
    values, _ = torch.topk(logits, k=top_k)
    threshold = values[:, [-1]]
    return torch.where(logits < threshold, torch.full_like(logits, float('-inf')), logits)


@torch.no_grad()
def generate(
    model: MiniGPT,
    tokenizer: SimpleTokenizer,
    prompt: str,
    max_new_tokens: int = 40,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    sample: bool = True,
) -> str:
    model.eval()
    token_ids = tokenizer.encode(prompt)
    idx = torch.tensor([token_ids], dtype=torch.long, device=DEVICE)

    print(f'Prompt: {prompt!r}')
    print(f'Prompt ids: {token_ids}')

    for step in range(max_new_tokens):
        idx_cond = idx[:, -model.context_length :]
        logits, _ = model(idx_cond)
        next_token_logits = logits[:, -1, :]

        if temperature <= 0:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            scaled_logits = next_token_logits / temperature
            filtered_logits = top_k_logits(scaled_logits, top_k=top_k)
            probs = torch.softmax(filtered_logits, dim=-1)
            next_token = (
                torch.multinomial(probs, num_samples=1)
                if sample
                else torch.argmax(probs, dim=-1, keepdim=True)
            )

        idx = torch.cat([idx, next_token], dim=1)
        partial_text = tokenizer.decode(idx[0].tolist())
        print(f'step={step + 1:02d} next_token_id={next_token.item()} texte={partial_text!r}')

    return tokenizer.decode(idx[0].tolist())


def load_model_and_tokenizer() -> tuple[MiniGPT, SimpleTokenizer]:
    if CHECKPOINT_PATH.exists():
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        tokenizer = SimpleTokenizer(checkpoint['vocab'][1:])
        tokenizer.itos = checkpoint['vocab']
        tokenizer.stoi = {token: idx for idx, token in enumerate(tokenizer.itos)}
        tokenizer.unk_id = tokenizer.stoi['<unk>']
        config = checkpoint['config']
        model = MiniGPT(
            vocab_size=len(tokenizer.itos),
            context_length=config['context_length'],
            embedding_dim=config['embedding_dim'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            ffn_dim=config['ffn_dim'],
            dropout=config['dropout'],
        ).to(DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, tokenizer

    text = CORPUS_PATH.read_text(encoding='utf-8')
    tokenizer = SimpleTokenizer.from_text(text)
    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        context_length=32,
        embedding_dim=64,
        num_heads=4,
        num_layers=2,
        ffn_dim=256,
        dropout=0.1,
    ).to(DEVICE)
    return model, tokenizer


def main() -> None:
    print(f'Using device: {DEVICE}')
    model, tokenizer = load_model_and_tokenizer()
    if CHECKPOINT_PATH.exists():
        print(f'Checkpoint chargé depuis {CHECKPOINT_PATH}')
    else:
        print('Aucun checkpoint trouvé: la génération utilisera un modèle aléatoire, donc le texte sera peu cohérent.')

    print()
    print('=== 1. Argmax (température 0) ===')
    generate(model, tokenizer, prompt='bonjour ', max_new_tokens=20, temperature=0.0, top_k=None, sample=False)

    print()
    print('=== 2. Sampling pédagogique ===')
    print('temperature petite -> distribution plus pointue; temperature grande -> plus de diversité.')
    generate(model, tokenizer, prompt='mini ', max_new_tokens=20, temperature=0.7, top_k=5, sample=True)
    print('Exercice: compare temperature=0.2, 0.7 et 1.2, puis top_k=None vs top_k=3.')


if __name__ == '__main__':
    main()
