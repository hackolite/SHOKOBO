"""Chapitre 6 — Entraînement.

forward -> logits -> loss -> backward() -> optimizer.step() -> zero_grad()
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

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


def create_dataloaders(token_ids: list[int], context_length: int, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    dataset = NextTokenDataset(token_ids, context_length=context_length)
    split_idx = max(1, int(0.9 * len(dataset)))
    train_indices = list(range(split_idx))
    val_indices = list(range(split_idx, len(dataset))) or [len(dataset) - 1]
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model: MiniGPT, dataloader: DataLoader, criterion: nn.Module) -> float:
    model.eval()
    losses = []
    for x, y in dataloader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        logits, _ = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(len(losses), 1)


def train() -> Dict[str, object]:
    print(f'Using device: {DEVICE}')
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    config = {
        'context_length': 32,
        'embedding_dim': 64,
        'num_heads': 4,
        'num_layers': 2,
        'ffn_dim': 256,
        'dropout': 0.1,
        'batch_size': 8,
        'epochs': 5,
        'learning_rate': 3e-4,
    }

    text = CORPUS_PATH.read_text(encoding='utf-8')
    tokenizer = SimpleTokenizer.from_text(text)
    token_ids = tokenizer.encode(text)
    train_loader, val_loader = create_dataloaders(
        token_ids,
        context_length=config['context_length'],
        batch_size=config['batch_size'],
    )

    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        context_length=config['context_length'],
        embedding_dim=config['embedding_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        ffn_dim=config['ffn_dim'],
        dropout=config['dropout'],
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    train_losses = []
    val_losses = []
    global_step = 0

    print()
    print('=== Shapes pour la loss ===')
    sample_x, sample_y = next(iter(train_loader))
    print(f'X: shape={list(sample_x.shape)} = [B, T]')
    print(f'Y: shape={list(sample_y.shape)} = [B, T]')
    sample_logits, _ = model(sample_x.to(DEVICE))
    print(f'logits: shape={list(sample_logits.shape)} = [B, T, vocab_size]')
    print('Pour CrossEntropyLoss: logits -> [B*T, vocab_size], targets -> [B*T]')

    for epoch in range(1, config['epochs'] + 1):
        epoch_losses = []
        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits, _ = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            global_step += 1
            print(
                f"epoch={epoch:02d} step={global_step:03d} "
                f"loss={loss.item():.4f} lr={optimizer.param_groups[0]['lr']:.6f}"
            )

        train_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        val_loss = evaluate(model, val_loader, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f'--> fin epoch {epoch}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f}')

    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, marker='o', label='train')
    plt.plot(val_losses, marker='s', label='validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Courbe de loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH)
    plt.close()

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'vocab': tokenizer.itos,
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    print(f'Checkpoint sauvegardé: {CHECKPOINT_PATH}')
    print(f'Courbe sauvegardée: {LOSS_PLOT_PATH}')
    print('Exercice: augmente learning_rate à 1e-3. Observe si l apprentissage devient plus instable ou plus rapide.')
    return checkpoint


if __name__ == '__main__':
    train()
