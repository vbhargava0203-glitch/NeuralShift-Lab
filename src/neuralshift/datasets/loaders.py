"""Official torchvision datasets with persisted sample identities and source metadata."""

from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from neuralshift.datasets.splits import make_splits, subset_indices
from neuralshift.datasets.transforms import STATS, normalize
from neuralshift.reproducibility import seed_worker
from neuralshift.utils.artifacts import sha256


class IndexedDataset(Dataset):
    def __init__(self, source, indices, dataset, partition, normalized=True):
        self.source, self.indices, self.dataset = source, np.asarray(indices), dataset
        self.partition, self.normalized = partition, normalized

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        source_index = int(self.indices[index])
        image, label = self.source[source_index]
        if self.normalized:
            image = normalize(image, self.dataset)
        return image, int(label), source_index


def load_source(config, train=True):
    name, root = config["dataset"], config["data"]["root"]
    cls = datasets.FashionMNIST if name == "fashion_mnist" else datasets.CIFAR10
    if name == "fashion_mnist":
        # Official maintainer mirror, same MD5-verified files as torchvision.
        cls.mirrors = ["https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"]
    try:
        return cls(root=root, train=train, download=config["data"]["download"], transform=transforms.ToTensor())
    except RuntimeError as exc:
        raise RuntimeError(f"Cannot load {name}. Run download-data with this config first. {exc}") from exc


def training_data(config):
    source = load_source(config, train=True)
    labels = np.asarray(source.targets)
    data = config["data"]
    train, val = make_splits(labels, data["validation_fraction"], data["split_seed"])
    train = subset_indices(train, data["train_limit"], data["split_seed"] + 1)
    val = subset_indices(val, data["val_limit"], data["split_seed"] + 2)
    return IndexedDataset(source, train, config["dataset"], "train"), IndexedDataset(source, val, config["dataset"], "validation")


def test_data(config):
    source = load_source(config, train=False)
    indices = subset_indices(np.arange(len(source)), config["data"]["test_limit"], config["data"]["split_seed"] + 3)
    return IndexedDataset(source, indices, config["dataset"], "test", normalized=False)


def make_loader(dataset, config, shuffle=False, evaluation=False, generator=None):
    section = config["evaluation"] if evaluation else config["training"]
    return DataLoader(dataset, batch_size=section["batch_size"], shuffle=shuffle,
                      num_workers=config["training"]["workers"], worker_init_fn=seed_worker,
                      generator=generator)


def metadata(config, train, val):
    def split_info(dataset):
        labels = np.asarray(dataset.source.targets)[dataset.indices]
        return {"n": len(dataset), "class_distribution": np.bincount(labels, minlength=10).tolist(),
                "identifier_namespace": "official_train", "indices": dataset.indices.tolist()}
    root = Path(config["data"]["root"])
    candidates = list((root / "FashionMNIST" / "raw").glob("*.gz")) if config["dataset"] == "fashion_mnist" else list((root / "cifar-10-batches-py").glob("data_batch*"))
    return {"name": config["dataset"], "source": "torchvision official dataset loader (Fashion-MNIST official GitHub mirror)",
            "official_train_n": len(train.source), "official_test_n": 10000,
            "classes": list(train.source.classes), "split_seed": config["data"]["split_seed"],
            "split_strategy": "stratified train/validation; optional seeded uniform subsets; official test retained separately",
            "train": split_info(train), "validation": split_info(val),
            "normalization": STATS[config["dataset"]], "transform_order": "uint8 to float [0,1], optional test corruption, normalize",
            "source_sha256": {p.name: sha256(p) for p in candidates}}
