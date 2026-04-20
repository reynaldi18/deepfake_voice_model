"""Training utilities shared across notebooks."""
import random
import numpy as np
import torch
import torch.nn as nn


def get_device() -> torch.device:
    """Pilih device terbaik: MPS → CUDA → CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int = 42):
    """Set semua random seed untuk reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, criterion, optimizer=None, device=None):
    """
    Jalankan satu epoch train atau eval.

    Args:
        model: model PyTorch
        loader: DataLoader
        criterion: loss function
        optimizer: jika None → mode eval
        device: jika None → diambil dari parameter model

    Returns:
        (loss, accuracy) tuple
    """
    if device is None:
        device = next(model.parameters()).device

    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item() * len(labels)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += len(labels)

    return total_loss / total, correct / total
