# Two-minute technical demonstration

1. Open README and execution ledger. State that this is a reduced five-seed empirical pilot.
2. Show a run manifest, resolved configuration, validation-only calibration record, and recorded checkpoint hash.
3. Run `python -m neuralshift.cli validate-results --results results/pilot`.
4. Show accuracy versus severity, reliability, ROC, and risk-coverage figures. Explain their distinct meanings.
5. Open paired_comparisons.csv: the unit is training seed, not dropout pass or corruption image.
6. Show `python -m neuralshift.cli estimate --config configs/experiments.yaml` and identify remaining work.

Do not describe the full research matrix, CUDA, or CIFAR-10 training as completed.
