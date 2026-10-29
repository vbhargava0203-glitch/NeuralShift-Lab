"""Seeding, hardware discovery, and execution provenance."""

import importlib
import importlib.metadata
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import torch


def seed_everything(seed, deterministic=True, threads=4):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def select_device(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; select training.device: cpu")
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu" if name == "auto" else name)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def environment():
    packages = ["torch", "torchvision", "numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "seaborn", "PyYAML"]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    modules = {"torch": "torch", "torchvision": "torchvision", "numpy": "numpy", "pandas": "pandas", "scipy": "scipy", "scikit-learn": "sklearn", "matplotlib": "matplotlib", "seaborn": "seaborn", "PyYAML": "yaml", "pillow": "PIL"}
    versions = {name: importlib.import_module(module).__version__ for name, module in modules.items()}
    return {"python": sys.version, "platform": platform.platform(), "processor": platform.processor(),
            "cpu_count": os.cpu_count(), "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "versions": versions, "distribution_metadata_versions": {p: importlib.metadata.version(p) for p in packages}, "git_commit": commit,
            "limitations": "Exact bitwise reproducibility across hardware, OS, or library versions is not guaranteed."}
