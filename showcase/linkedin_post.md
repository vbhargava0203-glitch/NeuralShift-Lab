# Suggested technical post

I built NeuralShift Lab to investigate a practical ML question: when an image classifier is wrong under distribution shift, can its uncertainty scores identify the error?

The Python/PyTorch platform includes controlled corruptions, maximum-softmax and entropy baselines, MC dropout, validation-fitted temperature scaling, and selective prediction. Results carry configuration, seed, checkpoint, and prediction-cache provenance.

I completed a reduced Fashion-MNIST CNN pilot across five training seeds and 21 conditions per seed. One useful lesson: a more complex uncertainty score need not be a better error detector. In this particular pilot, MC predictive entropy did not outperform maximum-softmax uncertainty on mean AUROC.

This is an empirical undergraduate research project, not a novel algorithm or a completed full-benchmark study. CIFAR-10, trained ResNet comparisons, and full-data experiments are the next milestones. The code, actual metrics, negative findings, and reproducibility notes are included in the project package.

#MachineLearning #PyTorch #UncertaintyQuantification #ReproducibleResearch

Add the public repository link only after publication is authorized and completed.
