"""Dataset contract tests use synthetic fixtures only, never research evidence."""
from copy import deepcopy

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from neuralshift.config import DEFAULT
from neuralshift.datasets import loaders
from neuralshift.datasets.splits import make_splits, subset_indices
from neuralshift.datasets.transforms import STATS, normalize


class SyntheticImages(Dataset):
    """Offline software-test fixture; deliberately not a real benchmark."""

    classes = [f"fixture_class_{i}" for i in range(10)]

    def __init__(self, n=100, channels=1, size=28):
        self.targets = torch.arange(n) % 10
        self.images = torch.rand(n, channels, size, size,
                                 generator=torch.Generator().manual_seed(123))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.images[index], int(self.targets[index])


@pytest.fixture
def config(tmp_path):
    result = deepcopy(DEFAULT)
    result["data"].update(root=str(tmp_path / "data"), validation_fraction=0.2,
                          train_limit=40, val_limit=20, test_limit=20, download=False)
    result["training"].update(batch_size=10, workers=0)
    return result



def test_stratified_splits_are_reproducible_disjoint_and_complete():
    labels = np.arange(100) % 10
    train, val = make_splits(labels, 0.2, seed=3)
    again_train, again_val = make_splits(labels, 0.2, seed=3)
    assert np.array_equal(train, again_train)
    assert np.array_equal(val, again_val)
    assert not set(train) & set(val)
    assert set(train) | set(val) == set(range(100))
    assert np.bincount(labels[val], minlength=10).tolist() == [2] * 10
    assert not np.array_equal(val, make_splits(labels, 0.2, seed=4)[1])
    assert np.array_equal(subset_indices(train, 20, 9), subset_indices(train, 20, 9))
    assert len(set(subset_indices(train, 20, 9))) == 20


@pytest.mark.parametrize("dataset,channels,size", [("fashion_mnist", 1, 28), ("cifar10", 3, 32)])
def test_dataset_dimensions_labels_normalization_and_metadata(monkeypatch, config, dataset, channels, size):
    config["dataset"] = dataset
    train_source = SyntheticImages(100, channels, size)
    test_source = SyntheticImages(30, channels, size)
    monkeypatch.setattr(loaders, "load_source", lambda config, train=True: train_source if train else test_source)
    train, val = loaders.training_data(config)
    test = loaders.test_data(config)
    image, label, index = train[0]
    assert image.shape == (channels, size, size)
    assert 0 <= label < 10 and label == train_source.targets[index]
    assert torch.allclose(image, normalize(train_source.images[index], dataset))
    raw_image, _, test_index = test[0]
    assert torch.equal(raw_image, test_source.images[test_index])
    assert 0 <= raw_image.min() <= raw_image.max() <= 1
    assert len(train) == 40 and len(val) == 20 and len(test) == 20
    train_ids = {("official_train", int(i)) for i in train.indices}
    val_ids = {("official_train", int(i)) for i in val.indices}
    test_ids = {("official_test", int(i)) for i in test.indices}
    assert not train_ids & val_ids and not train_ids & test_ids and not val_ids & test_ids
    meta = loaders.metadata(config, train, val)
    assert meta["name"] == dataset
    assert meta["official_train_n"] == len(train_source)
    assert meta["normalization"] == STATS[dataset]
    assert meta["split_seed"] == config["data"]["split_seed"]
    assert meta["train"]["identifier_namespace"] == "official_train"
    assert sum(meta["train"]["class_distribution"]) == len(train)
    assert sum(meta["validation"]["class_distribution"]) == len(val)
    assert meta["source_sha256"] == {}


def test_normalization_known_values():
    values = torch.tensor([[[0.286, 0.639]]])
    assert torch.allclose(normalize(values, "fashion_mnist"), torch.tensor([[[0.0, 1.0]]]), atol=1e-6)
    batched = values.repeat(3, 1, 1, 1)
    assert normalize(batched, "fashion_mnist").shape == batched.shape


def test_seeded_loader_shuffling(monkeypatch, config):
    source = SyntheticImages()
    monkeypatch.setattr(loaders, "load_source", lambda config, train=True: source)
    train, _ = loaders.training_data(config)

    def order(seed):
        generator = torch.Generator().manual_seed(seed)
        loader = loaders.make_loader(train, config, shuffle=True, generator=generator)
        return torch.cat([ids for _, _, ids in loader])

    assert torch.equal(order(5), order(5))
    assert not torch.equal(order(5), order(6))
