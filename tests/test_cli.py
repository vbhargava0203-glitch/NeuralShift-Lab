"""Public CLI syntax and offline error handling."""
from copy import deepcopy
import json

import pytest
import yaml

from neuralshift.cli import main, parser
from neuralshift.config import DEFAULT, load_config


@pytest.mark.parametrize("command", [None, "download-data", "train", "evaluate", "experiment",
                                      "estimate", "figures", "report", "validate-results", "ablate"])
def test_all_commands_offer_help(command, capsys):
    args = ["--help"] if command is None else [command, "--help"]
    with pytest.raises(SystemExit) as exit_info:
        main(args)
    assert exit_info.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_actual_evaluate_and_resume_syntax():
    evaluate = parser().parse_args(["evaluate", "--config", "protocol.yaml", "--checkpoint", "run/best.pt"])
    assert evaluate.checkpoint == "run/best.pt"
    train = parser().parse_args(["train", "--config", "protocol.yaml", "--resume", "run"])
    assert train.resume == "run"


def test_estimate_is_offline_and_computes_matrix(tmp_path, capsys):
    config = deepcopy(DEFAULT)
    config["training"]["epochs"] = 1
    config["data"].update(train_limit=40, val_limit=20, test_limit=20)
    config["evaluation"].update(mc_passes=2, corruptions=["gaussian_noise"], severities=[1])
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert main(["estimate", "--config", str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["training_runs"] == 1
    assert result["training_example_epochs"] == 40
    assert result["evaluation_forward_examples"] == 120


def test_missing_config_is_actionable(tmp_path, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["train", "--config", str(tmp_path / "missing.yaml")])
    assert exit_info.value.code == 2
    assert "missing.yaml" in capsys.readouterr().err


def test_config_inheritance_and_rejected_bad_protocol(tmp_path):
    (tmp_path / "base.yaml").write_text("dataset: cifar10\nseed: 7\n", encoding="utf-8")
    (tmp_path / "child.yaml").write_text("extends: base.yaml\ntraining:\n  epochs: 2\n", encoding="utf-8")
    config = load_config(tmp_path / "child.yaml")
    assert config["dataset"] == "cifar10" and config["seed"] == 7
    assert config["training"]["epochs"] == 2
    assert config["training"]["batch_size"] == DEFAULT["training"]["batch_size"]
    (tmp_path / "bad.yaml").write_text("evaluation:\n  mc_passes: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="at least two"):
        load_config(tmp_path / "bad.yaml")
