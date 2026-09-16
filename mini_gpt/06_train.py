"""Chapitre 6 — Entraînement.

forward -> logits -> loss -> backward() -> optimizer.step() -> zero_grad()
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE_DIR = Path(__file__).resolve().parent
CORPUS_PATH = BASE_DIR / 'data' / 'tiny_corpus.txt'
CHECKPOINT_DIR = BASE_DIR / 'checkpoints'
CHECKPOINT_PATH = CHECKPOINT_DIR / 'mini_gpt.pt'
LOSS_PLOT_PATH = CHECKPOINT_DIR / 'loss_curve.png'


def load_local_module(filename: str, module_name: str):
    path = BASE_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


tokenizer_module = load_local_module('01_tokenizer.py', 'mini_gpt_tokenizer')
dataset_module = load_local_module('02_dataset.py', 'mini_gpt_dataset')
model_module = load_local_module('05_model.py', 'mini_gpt_model')

SimpleTokenizer = tokenizer_module.SimpleTokenizer
NextTokenDataset = dataset_module.NextTokenDataset
MiniGPT = model_module.MiniGPT
MODEL_PRESETS = {
    'tiny': dict(context_length=32, embedding_dim=64, num_heads=4, num_layers=2, ffn_dim=256),
    'small': dict(context_length=128, embedding_dim=128, num_heads=4, num_layers=4, ffn_dim=512),
    'medium': dict(context_length=256, embedding_dim=256, num_heads=8, num_layers=6, ffn_dim=1024),
}
MODEL_KEYS = tuple(MODEL_PRESETS['tiny']) + ('dropout', 'position_encoding', 'attention_backend')


def create_dataloaders(token_ids: list[int], context_length: int, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    split_idx = int(0.9 * len(token_ids))
    train_dataset = NextTokenDataset(token_ids[:split_idx], context_length=context_length)
    val_dataset = NextTokenDataset(token_ids[split_idx:], context_length=context_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


@torch.no_grad()
def evaluate(
    model: MiniGPT,
    dataloader: DataLoader,
    criterion: nn.Module,
    precision: str = 'float32',
) -> float:
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == 'bfloat16'):
            logits, _ = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
    model.train(was_training)
    return total_loss / max(total_tokens, 1)


def train(
    corpus_path: Path = CORPUS_PATH,
    checkpoint_dir: Path = CHECKPOINT_DIR,
    preset: str = 'tiny',
    tokenizer_kind: str = 'char',
    vocab_size: int = 2000,
    validation_fraction: float = 0.1,
    device: str = DEVICE,
    **overrides,
) -> Dict[str, object]:
    if preset not in MODEL_PRESETS:
        raise ValueError(f'Unknown preset: {preset}')
    config = {
        **MODEL_PRESETS[preset],
        'dropout': 0.1,
        'batch_size': 8,
        'epochs': 5,
        'learning_rate': 3e-4,
        'position_encoding': 'learned',
        'attention_backend': 'manual',
        'stride': 1,
        'max_steps': None,
        'precision': 'float32',
        'seed': 42,
    }
    unknown = overrides.keys() - config.keys()
    if unknown:
        raise ValueError(f'Unknown options: {sorted(unknown)}')
    config.update({key: value for key, value in overrides.items() if value is not None})
    for key in ('context_length', 'embedding_dim', 'num_heads', 'num_layers', 'ffn_dim',
                'batch_size', 'epochs', 'stride'):
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f'{key} must be a positive integer.')
    if not 0 <= config['dropout'] < 1 or not 0 < config['learning_rate'] < float('inf'):
        raise ValueError('Invalid dropout or learning_rate.')
    if config['max_steps'] is not None and config['max_steps'] <= 0:
        raise ValueError('max_steps must be positive.')
    if config['precision'] not in ('float32', 'bfloat16'):
        raise ValueError('precision must be float32 or bfloat16.')
    device = torch.device(device)
    torch.manual_seed(config['seed'])
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / 'mini_gpt.pt'
    loss_plot_path = checkpoint_dir / 'loss_curve.png'
    print(f'Using device: {device}')
    tokenizer, train_path, val_path = dataset_module.prepare_corpus(
        Path(corpus_path), checkpoint_dir / 'tokens', tokenizer_kind=tokenizer_kind,
        vocab_size=vocab_size, validation_fraction=validation_fraction,
    )
    datasets = [
        dataset_module.MemmapTokenDataset(path, config['context_length'], stride=config['stride'])
        for path in (train_path, val_path)
    ]
    # Replacement sampling avoids a corpus-sized shuffled index array.
    train_loader = DataLoader(
        datasets[0], batch_size=config['batch_size'],
        sampler=RandomSampler(datasets[0], replacement=True, num_samples=len(datasets[0])),
    )
    val_loader = DataLoader(datasets[1], batch_size=config['batch_size'], shuffle=False)
    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        **{key: config[key] for key in MODEL_KEYS},
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    train_losses = []
    val_losses = []
    global_step = 0
    tokens_seen = 0
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start_time = time.perf_counter()
    print(f'Parameters: {model.count_parameters():,}; tokenizer: {tokenizer_kind}')

    print()
    print('=== Shapes pour la loss ===')
    sample_x, sample_y = next(iter(train_loader))
    print(f'X: shape={list(sample_x.shape)} = [B, T]')
    print(f'Y: shape={list(sample_y.shape)} = [B, T]')
    with torch.no_grad():
        sample_logits, _ = model(sample_x.to(device))
    print(f'logits: shape={list(sample_logits.shape)} = [B, T, vocab_size]')
    print('Pour CrossEntropyLoss: logits -> [B*T, vocab_size], targets -> [B*T]')

    for epoch in range(1, config['epochs'] + 1):
        epoch_loss = 0.0
        epoch_tokens = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=config['precision'] == 'bfloat16'):
                logits, _ = model(x)
                loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * y.numel()
            epoch_tokens += y.numel()
            tokens_seen += y.numel()
            global_step += 1
            print(
                f"epoch={epoch:02d} step={global_step:03d} "
                f"loss={loss.item():.4f} lr={optimizer.param_groups[0]['lr']:.6f}"
            )
            if config['max_steps'] is not None and global_step >= config['max_steps']:
                break

        train_loss = epoch_loss / epoch_tokens
        val_loss = evaluate(model, val_loader, criterion, config['precision'])
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f'--> fin epoch {epoch}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f}')
        if config['max_steps'] is not None and global_step >= config['max_steps']:
            break

    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_time
    metrics = {
        'elapsed_seconds': elapsed,
        'training_tokens_per_second_including_validation': tokens_seen / elapsed,
        'parameters': model.count_parameters(),
        'parameter_bytes': sum(p.numel() * p.element_size() for p in model.parameters()),
        'peak_cuda_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
        'train_loss': train_losses,
        'validation_loss': val_losses,
    }
    print(f'Metrics: {metrics}')

    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, marker='o', label='train')
    plt.plot(val_losses, marker='s', label='validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Courbe de loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_plot_path)
    plt.close()

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'tokenizer': tokenizer.to_state(),
        'tokenizer_kind': tokenizer_kind,
        'metrics': metrics,
    }
    if tokenizer_kind == 'char':
        checkpoint['vocab'] = tokenizer.itos
    torch.save(checkpoint, checkpoint_path)
    print(f'Checkpoint sauvegardé: {checkpoint_path}')
    print(f'Courbe sauvegardée: {loss_plot_path}')
    print('Exercice: augmente learning_rate à 1e-3. Observe si l apprentissage devient plus instable ou plus rapide.')
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, default=CORPUS_PATH)
    parser.add_argument('--checkpoint-dir', type=Path, default=CHECKPOINT_DIR)
    parser.add_argument('--preset', choices=MODEL_PRESETS, default='tiny')
    parser.add_argument('--tokenizer', dest='tokenizer_kind', choices=('char', 'bpe'), default='char')
    parser.add_argument('--vocab-size', type=int, default=2000)
    parser.add_argument('--validation-fraction', type=float, default=0.1)
    parser.add_argument('--device', default=DEVICE)
    for key in ('context_length', 'embedding_dim', 'num_heads', 'num_layers', 'ffn_dim',
                'batch_size', 'epochs', 'stride', 'max_steps', 'seed'):
        parser.add_argument('--' + key.replace('_', '-'), type=int)
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--position-encoding', choices=('learned', 'rope'))
    parser.add_argument('--attention-backend', choices=('manual', 'sdpa'))
    parser.add_argument('--precision', choices=('float32', 'bfloat16'))
    args = vars(parser.parse_args())
    args['corpus_path'] = args.pop('corpus')
    train(**args)


if __name__ == '__main__':
    main()
