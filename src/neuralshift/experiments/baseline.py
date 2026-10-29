"""Train then evaluate one configuration using a single selected checkpoint."""

from neuralshift.training.trainer import train


def run_baseline(config):
    from neuralshift.experiments.runner import evaluate
    run = train(config)
    evaluate(config, run / "best.pt")
    return run
