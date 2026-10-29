"""Offline integration test with synthetic images, NOT a scientific experiment.

All outputs live in pytest temporary directories. Genuine benchmark results
must come from separately executed commands using the official dataset loader.
"""
from copy import deepcopy

import pandas as pd
import pytest
import torch
from torch.utils.data import Dataset

from neuralshift.config import DEFAULT
from neuralshift.datasets import loaders
from neuralshift.experiments import runner
from neuralshift.training import trainer
from neuralshift.utils.artifacts import atomic_json, read_json, sha256
from neuralshift.utils.validation import validate_run


class SyntheticPipelineFixture(Dataset):
    classes = [f"software_fixture_{i}" for i in range(10)]

    def __init__(self, n, seed):
        self.targets = torch.arange(n) % 10
        self.images = torch.rand(n, 1, 28, 28, generator=torch.Generator().manual_seed(seed))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.images[index], int(self.targets[index])


def test_offline_pipeline_resume_validation_plot_and_tamper_detection(tmp_path, monkeypatch):
    config = deepcopy(DEFAULT)
    config.update(profile="smoke", output=str(tmp_path / "software-test-results"))
    config["data"].update(root=str(tmp_path / "fixture-data"), download=False,
                          validation_fraction=0.2, train_limit=40, val_limit=20, test_limit=20)
    config["training"].update(epochs=1, batch_size=10, threads=1, workers=0, patience=2)
    config["evaluation"].update(batch_size=10, mc_passes=2, bins=5,
                                corruptions=["gaussian_noise"], severities=[1])
    official_train_fixture = SyntheticPipelineFixture(100, seed=101)
    official_test_fixture = SyntheticPipelineFixture(30, seed=202)
    partitions_loaded = []

    def fixture_source(config, train=True):
        assert not config["data"]["download"]
        partitions_loaded.append("train" if train else "test")
        return official_train_fixture if train else official_test_fixture

    monkeypatch.setattr(loaders, "load_source", fixture_source)
    run = trainer.train(config)
    assert partitions_loaded == ["train"]  # No test access during model or temperature selection.
    assert (run / "best.pt").is_file()
    assert read_json(run / "temperature.json")["split"] == "validation"
    before_checkpoint = sha256(run / "best.pt")
    genuine_mc_function = runner.mc_predict

    def interrupted_mc(*args, **kwargs):
        raise RuntimeError("intentional software-test evaluation interruption")

    monkeypatch.setattr(runner, "mc_predict", interrupted_mc)
    with pytest.raises(RuntimeError, match="intentional"):
        runner.evaluate(config, run / "best.pt")
    assert read_json(run / "manifest.json")["status"] == "evaluation_failed"

    def forbid_retraining(*args, **kwargs):
        raise AssertionError("An evaluation failure must reuse training and calibration artifacts")

    monkeypatch.setattr(trainer, "training_data", forbid_retraining)
    assert trainer.train(config, resume=run) == run
    assert sha256(run / "best.pt") == before_checkpoint
    monkeypatch.setattr(runner, "mc_predict", genuine_mc_function)
    assert runner.evaluate(config, run / "best.pt") == run
    result = validate_run(run)
    assert result["valid"] and result["conditions"] == 2
    metrics = pd.read_csv(run / "metrics.csv")
    assert len(metrics) == 14  # Seven scores/calibration combinations, two conditions.
    assert set(metrics.corruption) == {"clean", "gaussian_noise"}
    assert read_json(run / "manifest.json")["test"]["identifier_namespace"] == "official_test"
    # A genuine plot of executed software-test training history, not research evidence.
    from neuralshift.visualization.report_figures import _training_figure
    plot_dir = tmp_path / "software-test-figures"
    plot_dir.mkdir()
    files = _training_figure(run, plot_dir, "Synthetic software test — no research conclusions")
    assert files and any(path.suffix == ".png" for path in files)
    assert all(path.is_file() and path.stat().st_size > 100 for path in files)
    # Both content hashes and prediction-based metric recomputation defend integrity.
    metrics_path = run / "metrics.csv"
    original_bytes = metrics_path.read_bytes()
    metrics.loc[0, "accuracy"] = 0.123456789
    metrics.to_csv(metrics_path, index=False)
    with pytest.raises(ValueError, match="altered artifact"):
        validate_run(run)
    manifest = read_json(run / "manifest.json")
    manifest["artifacts"]["metrics.csv"] = sha256(metrics_path)
    atomic_json(run / "manifest.json", manifest)
    with pytest.raises(ValueError, match="do not match cached predictions"):
        validate_run(run)
    metrics_path.write_bytes(original_bytes)
    manifest["artifacts"]["metrics.csv"] = sha256(metrics_path)
    atomic_json(run / "manifest.json", manifest)
    assert validate_run(run)["valid"]

    from neuralshift.experiments.ablations import mc_sample_ablation
    ablation = mc_sample_ablation(run, [2])
    rows = pd.read_csv(ablation / "mc_samples.csv")
    assert len(rows) == 6 and set(rows.mc_passes) == {2}
    with pytest.raises(ValueError, match="cached pass count"):
        mc_sample_ablation(run, [3])
