"""Software-level repeatability checks; no scientific runs are simulated."""
import random
from datetime import datetime

import numpy as np
import pytest
import torch

from neuralshift.reproducibility import seed_everything, seed_worker, select_device, timestamp


def test_seed_controls_python_numpy_torch():
    def draw(seed):
        seed_everything(seed, deterministic=True, threads=1)
        return random.random(), np.random.rand(3), torch.rand(3)

    first = draw(81)
    again = draw(81)
    other = draw(82)
    assert first[0] == again[0] and first[0] != other[0]
    assert np.array_equal(first[1], again[1])
    assert torch.equal(first[2], again[2])
    assert not torch.equal(first[2], other[2])
    assert torch.are_deterministic_algorithms_enabled()


def test_worker_seeds_python_and_numpy_from_torch_seed():
    torch.manual_seed(90)
    seed_worker(0)
    first = random.random(), np.random.rand()
    torch.manual_seed(90)
    seed_worker(0)
    assert first == (random.random(), np.random.rand())


def test_unavailable_cuda_fails_clearly(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert str(select_device("auto")) == "cpu"
    assert str(select_device("cpu")) == "cpu"
    with pytest.raises(ValueError, match="unavailable"):
        select_device("cuda")


def test_timestamp_is_timezone_aware():
    assert datetime.fromisoformat(timestamp()).utcoffset().total_seconds() == 0
