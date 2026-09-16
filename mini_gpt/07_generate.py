"""Chapitre 7 — Génération autoregressive.

prompt -> tokenize -> model -> logits du dernier token -> probabilités -> prochain token
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import time
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
    use_cache: bool = False,
    verbose: bool = True,
) -> str:
    if max_new_tokens < 0:
        raise ValueError('max_new_tokens must be non-negative.')
    if not math.isfinite(temperature):
        raise ValueError('temperature must be finite.')
    model.eval()
    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        raise ValueError('Le prompt doit contenir au moins un token.')
    device = next(model.parameters()).device
    idx = torch.tensor([token_ids], dtype=torch.long, device=device)
    past_key_values = None

    if verbose:
        print(f'Prompt: {prompt!r}')
        print(f'Prompt ids: {token_ids}')

    for step in range(max_new_tokens):
        idx_cond = idx[:, -model.context_length :]
        if use_cache:
            # Rebuild a full sliding window on overflow: its positions restart at zero.
            if past_key_values is not None and past_key_values[0][0].size(-2) < model.context_length:
                idx_cond = idx[:, -1:]
            else:
                past_key_values = None
            logits, _, past_key_values = model(
                idx_cond, past_key_values=past_key_values, use_cache=True,
            )
        else:
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
        if verbose:
            partial_text = tokenizer.decode(idx[0].tolist())
            print(f'step={step + 1:02d} next_token_id={next_token.item()} texte={partial_text!r}')

    return tokenizer.decode(idx[0].tolist())


def load_model_and_tokenizer(
    checkpoint_path: Path = CHECKPOINT_PATH,
    device: str = DEVICE,
    attention_backend: Optional[str] = None,
    precision: str = 'float32',
) -> tuple[MiniGPT, SimpleTokenizer]:
    if precision not in ('float32', 'bfloat16'):
        raise ValueError('precision must be float32 or bfloat16.')
    dtype = torch.bfloat16 if precision == 'bfloat16' else torch.float32
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if 'tokenizer' in checkpoint:
            tokenizer = tokenizer_module.tokenizer_from_state(checkpoint['tokenizer'])
        else:
            tokenizer = SimpleTokenizer(checkpoint['vocab'][1:])
            tokenizer.itos = checkpoint['vocab']
            tokenizer.stoi = {token: idx for idx, token in enumerate(tokenizer.itos)}
            tokenizer.unk_id = tokenizer.stoi['<unk>']
        config = checkpoint['config']
        model = MiniGPT(
            vocab_size=tokenizer.vocab_size,
            context_length=config['context_length'],
            embedding_dim=config['embedding_dim'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            ffn_dim=config['ffn_dim'],
            dropout=config['dropout'],
            position_encoding=config.get('position_encoding', 'learned'),
            attention_backend=attention_backend or config.get('attention_backend', 'manual'),
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device=device, dtype=dtype)
        model.eval()
        return model, tokenizer

    if checkpoint_path != CHECKPOINT_PATH:
        raise FileNotFoundError(checkpoint_path)
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
        attention_backend=attention_backend or 'manual',
    ).to(device=device, dtype=dtype)
    model.eval()
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT_PATH)
    parser.add_argument('--device', default=DEVICE)
    parser.add_argument('--prompt', default='bonjour ')
    parser.add_argument('--max-new-tokens', type=int, default=20)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--greedy', action='store_true')
    parser.add_argument('--kv-cache', action='store_true')
    parser.add_argument('--attention-backend', choices=('manual', 'sdpa'))
    parser.add_argument('--precision', choices=('float32', 'bfloat16'), default='float32')
    parser.add_argument('--benchmark', action='store_true', help='Compare greedy generation with/without cache.')
    args = parser.parse_args()
    model, tokenizer = load_model_and_tokenizer(
        args.checkpoint, args.device, args.attention_backend, args.precision,
    )
    if args.checkpoint.exists():
        print(f'Checkpoint chargé depuis {args.checkpoint}')
    else:
        print('Aucun checkpoint trouvé: la génération utilisera un modèle aléatoire, donc le texte sera peu cohérent.')
    if not args.benchmark:
        print(generate(
            model, tokenizer, args.prompt, args.max_new_tokens,
            args.temperature, args.top_k, sample=not args.greedy, use_cache=args.kv_cache,
        ))
        return
    device = next(model.parameters()).device
    outputs = []
    for cache in (False, True):
        generate(model, tokenizer, args.prompt, 1, temperature=0, use_cache=cache, verbose=False)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        outputs.append(generate(
            model, tokenizer, args.prompt, args.max_new_tokens,
            temperature=0, use_cache=cache, verbose=False,
        ))
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
        print(f'cache={cache} seconds={elapsed:.4f} tokens/s={args.max_new_tokens / elapsed:.2f} peak_cuda_bytes={peak}')
    print(f'Greedy outputs identical: {outputs[0] == outputs[1]}')
    print(outputs[-1])


if __name__ == '__main__':
    main()
