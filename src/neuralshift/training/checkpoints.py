"""Atomic PyTorch checkpoints; load only artifacts from trusted local runs."""

import os
from pathlib import Path

import torch


def save_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_suffix(".tmp.pt")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path, device="cpu"):
    if not Path(path).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device, weights_only=False)
