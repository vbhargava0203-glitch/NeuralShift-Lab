# Scientific limitations

The completed five-seed Fashion-MNIST CNN pilot uses reduced subsets and only three training epochs and five stochastic passes. It cannot establish generality across architectures or datasets. The full 20-run study, full-data Fashion-MNIST, CIFAR-10, trained ResNet-18 comparisons, and dropout-probability ablations remain unexecuted.

Four custom synthetic corruptions are a narrow covariate-shift model, not official CIFAR-10-C and not semantic OOD. Severe transformations may change visual label validity. ECE is bin- and sample-dependent. MC dropout is an approximate uncertainty proxy with architecture-dependent placement. Validation is reused for checkpoint selection and temperature fitting; a separate calibration split would reduce reuse effects.

All comparisons are exploratory. Five seeds are still a small basis for inference. Conditions are averaged within seed and are not independent replicates. Bootstrap intervals quantify training-seed variability conditional on the fixed test sample, not uncertainty from drawing new test examples. Missing/degenerate detection scores stay missing. Negative findings are retained.

No novelty, state-of-the-art result, peer review, real-world reliability guarantee, or confirmatory significance claim is made. CPU Windows execution is tested; CUDA and other operating systems are not locally verified. GitHub Actions are supplied but have not been run on GitHub.
