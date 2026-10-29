"""Resolved, validated YAML configurations; relative paths use the project cwd."""

from copy import deepcopy
from pathlib import Path

import yaml

DEFAULT = {
    "profile": "development", "dataset": "fashion_mnist", "model": "cnn", "seed": 42,
    "data": {"root": "data", "download": False, "validation_fraction": 0.1,
             "split_seed": 2026, "train_limit": None, "val_limit": None, "test_limit": None},
    "model_options": {"dropout": 0.25, "batch_norm": True},
    "training": {"epochs": 15, "batch_size": 128, "lr": 0.001, "weight_decay": 0.0001,
                 "patience": 5, "device": "cpu", "workers": 0, "threads": 4,
                 "deterministic": True},
    "evaluation": {"batch_size": 128, "mc_passes": 30, "bins": 15,
                   "corruptions": ["gaussian_noise", "blur", "brightness", "contrast"],
                   "severities": [1, 2, 3, 4, 5], "corruption_seed": 1729},
    "output": "results",
}


def merge(base, overrides):
    result = deepcopy(base)
    for key, value in overrides.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def load_config(path):
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a YAML mapping")
    if "extends" in raw:
        base = load_config(path.parent / raw.pop("extends"))
    else:
        base = DEFAULT
    config = merge(base, raw)
    validate_config(config)
    return config


def validate_config(c):
    if c["dataset"] not in {"fashion_mnist", "cifar10"}:
        raise ValueError("dataset must be fashion_mnist or cifar10")
    if c["model"] not in {"cnn", "resnet18"}:
        raise ValueError("model must be cnn or resnet18")
    if not 0 < c["data"]["validation_fraction"] < 1:
        raise ValueError("validation_fraction must be between zero and one")
    for key in ("train_limit", "val_limit", "test_limit"):
        if c["data"][key] is not None and c["data"][key] < 10:
            raise ValueError(f"{key} must be null or at least 10")
    if not 0 < c["model_options"]["dropout"] < 1:
        raise ValueError("dropout must be strictly between 0 and 1 for MC dropout")
    for key in ("epochs", "batch_size", "patience", "threads"):
        if c["training"][key] < 1:
            raise ValueError(f"training.{key} must be positive")
    if c["training"]["lr"] <= 0 or c["training"]["workers"] < 0:
        raise ValueError("Invalid learning rate or worker count")
    if c["training"]["device"] not in {"cpu", "cuda", "auto"}:
        raise ValueError("device must be cpu, cuda, or auto")
    if c["evaluation"]["mc_passes"] < 2 or c["evaluation"]["bins"] < 1:
        raise ValueError("MC dropout needs at least two passes; ECE bins must be positive")
    if any(s not in range(1, 6) for s in c["evaluation"]["severities"]):
        raise ValueError("Corruption severities must be integers 1 through 5")
    if len(set(c["evaluation"]["severities"])) != len(c["evaluation"]["severities"]):
        raise ValueError("Duplicate severities are not allowed")
    if not set(c["evaluation"]["corruptions"]) <= {"gaussian_noise", "blur", "brightness", "contrast"}:
        raise ValueError("Unknown corruption family")
