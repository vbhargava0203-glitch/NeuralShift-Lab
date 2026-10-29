# Mathematical and implementation definitions

## Transformations

Pixels are floating point BCHW tensors in [0,1]. Corruptions occur before normalization and preserve labels and identifiers. Validation comes only from the official training partition.

| Family | Severity 1 through 5 | Definition |
| --- | --- | --- |
| Gaussian noise | sigma 0.05, 0.10, 0.15, 0.20, 0.30 | clip(x + sigma * epsilon, 0, 1), independent standard-normal pixel noise |
| Blur | sigma 0.5, 0.8, 1.1, 1.5, 2.0 pixels | Spatial Gaussian filter; implementation fixes kernel and padding |
| Brightness | offset 0.05, 0.10, 0.20, 0.30, 0.40 | clip(x + offset, 0, 1); brightening only |
| Contrast | factor 0.8, 0.6, 0.4, 0.2, 0.1 | Shrink each image/channel deviation from its spatial mean |

Clean uses severity 0. Noise uses a local CPU PyTorch generator and sample-specific seeds. These custom recipes are not CIFAR-10-C. Severe corruption can affect human recognizability.

## Uncertainty

MSP uncertainty is 1-max(p). Entropy is -sum(p log p), with 0 log 0 = 0; normalized entropy divides by log K. Reports use nats unless stated otherwise. For S dropout passes, predictive entropy is H(mean(p_s)), expected entropy is mean(H(p_s)), and their difference is a mutual-information proxy. Variance is population variance over passes averaged over classes. Batch normalization remains frozen. Passes are not independent training replicates.

Temperature minimizes validation NLL using softmax(z/T), T > 0. It preserves argmax. Binary confidence ordering is preserved; multiclass across-example ordering can change. Improved validation NLL does not guarantee improved shifted ECE or error ranking.

## Metrics

NLL is mean -log(p_true). Multiclass Brier score sums squared probability error across classes then averages examples; it is not divided by K. Macro F1 uses the fixed class vocabulary. Undefined classwise divisions use zero.

ECE uses 15 equal-width bins, includes confidence 1 in the final bin, and weights absolute confidence-accuracy gaps by bin sample fraction. Empty bins have zero weight. ECE is sensitive to binning, finite samples, and hidden class-conditional behavior.

Error positives are incorrect deterministic predictions. Higher scores mean greater uncertainty. AUROC and AUPRC are missing when the task has only one outcome class. AUPRC denotes average precision. FPR at target TPR is based on an empirical ROC threshold reaching at least the target; this is an evaluation summary, not a test-tuned deployment rule.

Select predictions by ascending uncertainty with stable ties. At accepted count k, coverage=k/n and risk=cumulative errors/k. AURC is the mean of n prefix risks, a right-step integral without an invented zero-coverage value. The deterministic correctness target is shared by all scores.

Error detection, clean-versus-corrupted detection, and semantic OOD detection remain distinct tasks. The latter is not established by this study.

The pilot uses 3,000 training examples, 600 validation examples, 1,000 held-out official test examples, three epochs and five MC passes. Seeds are 11, 22, 33, 44, 55. This is a reduced empirical study, not the full benchmark.
