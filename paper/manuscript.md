# Evaluating Predictive Reliability Under Controlled Distribution Shifts in Deep Neural Networks

Empirical undergraduate research study. Automatically generated from validated execution artifacts.
Generated: 2026-10-28T16:26:18.272224+00:00. Full five-seed research matrix: **pending** unless independently verified in the execution ledger.

## 1. Abstract
NeuralShift Lab evaluates the ability of uncertainty scores to identify errors made by an image classifier under controlled image corruptions. This report includes 5 validated run directories and 105 dataset/model/seed/condition combinations. These counts describe stored evidence, not completion of the research program. Small or single-seed runs establish pipeline execution and preliminary observations; they do not establish general superiority of any method.

## 2. Introduction
Accuracy, probability calibration, error detection, shift detection, and selective prediction measure different properties. A confident model may be wrong, and a calibrated model need not rank errors well. We examine these properties separately using reproducible image classification experiments.

## 3. Related Work
The maximum-softmax baseline follows Hendrycks and Gimpel (2017). Temperature scaling follows Guo et al. (2017). MC dropout is motivated by Gal and Ghahramani (2016); this implementation is an approximation and does not claim a calibrated Bayesian posterior. Corruption analysis is inspired by Hendrycks and Dietterich (2019), but the custom transformations here are not the official CIFAR-10-C benchmark and scores must not be directly compared with that benchmark.

## 4. Research Questions and Hypotheses
Primary hypothesis: some uncertainty-aware scores may identify deterministic-classifier errors more effectively than 1 minus maximum softmax probability under selected corruptions. RQ1 asks how accuracy changes with severity; RQ2 whether confidence follows accuracy; RQ3 how MSP, entropy, and MC dropout compare; RQ4 whether validation-fitted temperature scaling transfers to shifted data; RQ5 whether findings persist across datasets, architectures, and seeds; RQ6 how abstention changes risk; RQ7 whether paired differences are meaningful. No direction of improvement is assumed.

## 5. Mathematical Background
For logits z, p_k = exp(z_k) / sum_j exp(z_j). MSP uncertainty is 1 - max_k p_k. Entropy H(p) = -sum_k p_k log(p_k), with 0 log 0 = 0. With S dropout passes, p_bar = mean_s p_s. Predictive entropy is H(p_bar), expected entropy is mean_s H(p_s), and mutual information is their difference. The latter is an epistemic-uncertainty proxy. Temperature scaling uses softmax(z/T), T > 0, fitted by minimizing validation NLL. Positive scalar scaling preserves the predicted class. It need not preserve across-example MSP rankings in multiclass classification; in binary classification the confidence ranking is preserved.

## 6. Datasets and Experimental Setup
Supported sources are Fashion-MNIST (official 60,000 training / 10,000 test; 28 by 28 grayscale) and CIFAR-10 (official 50,000 training / 10,000 test; 32 by 32 RGB). Training and validation come only from the official training partition. Reduced profiles use explicitly recorded subsets; their scores are not full-benchmark scores. See each manifest for actual subset sizes, split seed, source checksums, normalization, hardware, library versions, and resolved configuration.

- `5d25a4af27b5`: [manifest](../results/pilot/runs/5d25a4af27b5/manifest.json), [metrics](../results/pilot/runs/5d25a4af27b5/metrics.csv)
- `9411ebb3ea90`: [manifest](../results/pilot/runs/9411ebb3ea90/manifest.json), [metrics](../results/pilot/runs/9411ebb3ea90/metrics.csv)
- `a3dc5434183b`: [manifest](../results/pilot/runs/a3dc5434183b/manifest.json), [metrics](../results/pilot/runs/a3dc5434183b/metrics.csv)
- `e337964e428e`: [manifest](../results/pilot/runs/e337964e428e/manifest.json), [metrics](../results/pilot/runs/e337964e428e/metrics.csv)
- `ee2ed85c603f`: [manifest](../results/pilot/runs/ee2ed85c603f/manifest.json), [metrics](../results/pilot/runs/ee2ed85c603f/metrics.csv)

## 7. Neural Network Architectures
The lightweight CNN uses convolutional feature blocks, ReLU, pooling, optional batch normalization, dropout, adaptive pooling, and a classification head. ResNet-18 uses a 3 by 3 stride-one input convolution, no initial max pooling, and dropout before the classifier. Both are initialized from scratch. Precise configurations and parameter counts belong to the recorded runs. Architecture comparisons require separately trained, matched-seed models.

## 8. Distribution Shift Methodology
Gaussian noise, Gaussian blur, brightness, and contrast operate on floating point pixels in [0, 1], before normalization. Five predefined severity levels represent a controlled subset of synthetic covariate shifts. Random noise is reproducible from recorded seeds and sample identifiers. Corruption is applied only during held-out evaluation. The exact equations, clipping rules, and mappings are documented in `docs/methodology.md` and implemented in the corruption registry. Image-label semantics may become ambiguous at large severity, a threat to validity.

## 9. Uncertainty Estimation Methods
Raw MSP and entropy, temperature-scaled MSP and entropy, and raw MC predictive entropy, mutual information, and probability variance are evaluated. MC dropout changes only dropout module behavior and keeps batch normalization in evaluation mode. The primary error target is the same deterministic classifier for every score, ensuring a matched comparison. MC mean predictions are not silently substituted for that classifier. Temperature is fitted only to validation logits; it is not optimized against final test labels.

## 10. Evaluation Metrics
Accuracy is the fraction correct. Macro F1 averages the classwise harmonic mean of precision and recall. NLL is mean -log p_y; the multiclass Brier score is mean sum_k (p_k - 1[y=k])^2. ECE is sum_b (n_b/n) abs(acc_b-conf_b) over 15 equal-width confidence bins. ECE depends on binning and finite-sample effects and is not a complete calibration measure. Error-detection positives are incorrect predictions, with larger uncertainty scores predicting errors. AUROC, average precision (reported as AUPRC), and observed-threshold FPR at target TPR are undefined when only one outcome class exists. Selective risk is the error rate among the least-uncertain accepted examples. AURC is the arithmetic mean of prefix risks at coverages 1/n through 1, with stable tie ordering.

## 11. Experimental Results
The table below is calculated from stored clean-condition metrics. The full condition table is [validated_metrics.csv](../results/pilot/reports/20261028T162617Z-86a5cd/validated_metrics.csv). Repeated accuracy values across scoring methods reflect a shared deterministic prediction target, not independent experiments.

| run_id | dataset | model | seed | calibration | n | accuracy | nll | ece | auroc | aurc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5d25a4af27b5 | fashion_mnist | cnn | 11 | raw | 1000 | 0.75100 | 0.70606 | 0.10292 | 0.85030 | 0.07680 |
| 5d25a4af27b5 | fashion_mnist | cnn | 11 | temperature | 1000 | 0.75100 | 0.66046 | 0.03034 | 0.85016 | 0.07709 |
| 9411ebb3ea90 | fashion_mnist | cnn | 44 | raw | 1000 | 0.74300 | 0.70030 | 0.07012 | 0.84855 | 0.07950 |
| 9411ebb3ea90 | fashion_mnist | cnn | 44 | temperature | 1000 | 0.74300 | 0.66619 | 0.02827 | 0.84627 | 0.08073 |
| a3dc5434183b | fashion_mnist | cnn | 55 | raw | 1000 | 0.73900 | 0.71949 | 0.08396 | 0.84646 | 0.08328 |
| a3dc5434183b | fashion_mnist | cnn | 55 | temperature | 1000 | 0.73900 | 0.68454 | 0.02925 | 0.84197 | 0.08512 |
| e337964e428e | fashion_mnist | cnn | 22 | raw | 1000 | 0.74600 | 0.70644 | 0.08449 | 0.84677 | 0.08005 |
| e337964e428e | fashion_mnist | cnn | 22 | temperature | 1000 | 0.74600 | 0.66874 | 0.02972 | 0.84484 | 0.08120 |
| ee2ed85c603f | fashion_mnist | cnn | 33 | raw | 1000 | 0.74000 | 0.71384 | 0.07675 | 0.83223 | 0.08602 |
| ee2ed85c603f | fashion_mnist | cnn | 33 | temperature | 1000 | 0.74000 | 0.68048 | 0.02499 | 0.82992 | 0.08688 |

No result in this report is evidence for the unexecuted full research matrix. Figures below are generated from stored predictions and histories. Missing ROC plots indicate an undefined one-class error-detection task; they are not assigned a fabricated score.

## 12. Statistical Analysis
Independent training seed is the resampling unit for paired method comparisons. Condition-matched seed differences are averaged within seed before a percentile bootstrap over seeds (5,000 draws, fixed bootstrap seed 137). A statistical plot is omitted when fewer than two independent seeds are available or condition matching fails. Two seeds provide very limited uncertainty information; at least five seeds are the planned primary design. Executed exploratory paired t-test p-values and paired effect sizes are stored in paired_comparisons.csv; no confirmatory claim is made. Mean, standard deviation, sample count, and t intervals appear in seed_summary.csv. Single-seed intervals remain missing. All current comparisons are exploratory; a future confirmatory family should prespecify contrasts and use Holm correction. Repeated dropout passes and repeated corruptions are not independent training runs. The intervals describe seed variation conditional on this fixed test sample, not population sampling uncertainty.

## 13. Discussion
In the reduced five-seed pilot, clean accuracy averaged 0.7438 (sample SD 0.0049). MC predictive entropy had mean error-detection AUROC difference -0.0429 relative to MSP, with a paired seed-bootstrap 95% interval [-0.0484, -0.0352], after equally averaging the 21 evaluated conditions within each seed. Thus the proposed MC-entropy advantage was not observed in this pilot. This negative result is conditional on the small CNN, short training, fixed test subset, selected corruptions, and five MC passes; it does not establish a general ranking of uncertainty methods. Comparisons remain exploratory.

The stored results are descriptive observations under the exact recorded protocol. Better AUROC would indicate better error ranking for the shared classifier, whereas lower ECE would indicate a smaller binned confidence-accuracy gap. These improvements are not interchangeable. Negative or inconsistent method differences should be retained. This report does not select a winning method from a limited run or infer deployment safety from confidence scores.

## 14. Limitations
The full dataset-by-model-by-seed matrix and controlled ablations remain pending unless listed in the execution ledger. Reduced subsets, short training, and few stochastic passes limit any preliminary analysis. Only four custom corruption families are included. No semantic OOD benchmark, real-world deployment study, or novel algorithm is claimed. Calibration and early stopping share validation data, which can create validation reuse effects; a separate calibration partition is a worthwhile extension.

## 15. Threats to Validity
Potential threats include class imbalance in small subsets, finite-sample ECE bias, near-degenerate detection tasks, test reuse during exploratory iteration, severe-corruption label ambiguity, and architecture-dependent dropout placement. Repeated observations from the same seed are correlated. Scientific conclusions require prespecified comparisons and independent follow-up rather than test-driven configuration selection.

## 16. Reproducibility Statement
Run manifests preserve resolved configuration, hardware and software details, seeds, statuses, checkpoint locations and hashes. Caches store deterministic logits, labels, identifiers, and stochastic probabilities. Validation is required before these tables or figures are produced. Start from the pinned environment in `requirements.txt`, install the package, then use `python -m neuralshift.cli --help`. Deterministic execution is hardware- and library-dependent; bitwise equivalence across different platforms is not promised.

## 17. Conclusion
This repository supplies an executable empirical protocol and the validated evidence listed above. The primary hypothesis is not accepted or rejected by software tests or a one-seed smoke run. The next scientific milestone is a prespecified five-seed comparison with complete matched conditions, uncertainty intervals, and documented practical effect sizes.

## 18. References
- Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks*. ICML. https://proceedings.mlr.press/v70/guo17a.html
- Gal, Y., and Ghahramani, Z. (2016). *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning*. ICML. https://proceedings.mlr.press/v48/gal16.html
- Hendrycks, D., and Gimpel, K. (2017). *A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks*. ICLR. https://arxiv.org/abs/1610.02136
- Hendrycks, D., and Dietterich, T. (2019). *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations*. ICLR. https://arxiv.org/abs/1903.12261
- Xiao, H., Rasul, K., and Vollgraf, R. (2017). *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms*. https://arxiv.org/abs/1708.07747
- Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images*. Technical report, University of Toronto. https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
- He, K., Zhang, X., Ren, S., and Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR. https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html


## 19. Appendix
### Stored evidence
Each table row is traceable through its run identifier to a manifest, checkpoint, condition cache, and metric computation. The report-generation process does not impute missing runs.

### Figures

![01a_dataset_examples](../results/pilot/runs/5d25a4af27b5/figures/20261028T162341Z-ce0773/01a_dataset_examples.png)

![01b_class_distribution](../results/pilot/runs/5d25a4af27b5/figures/20261028T162341Z-ce0773/01b_class_distribution.png)

![03_corruption_severity](../results/pilot/runs/5d25a4af27b5/figures/20261028T162341Z-ce0773/03_corruption_severity.png)

![02_training_convergence](../results/pilot/runs/5d25a4af27b5/figures/20261028T162341Z-ce0773/02_training_convergence.png)

![04_accuracy_degradation](../results/pilot/runs/5d25a4af27b5/figures/20261028T162341Z-ce0773/04_accuracy_degradation.png)

![05_confidence_reliability](../results/pilot/runs/5d25a4af27b5/figures/20261028T162341Z-ce0773/05_confidence_reliability.png)

![09_method_comparison](../results/pilot/runs/5d25a4af27b5/figures/20261028T162341Z-ce0773/09_method_comparison.png)

![06_calibration_clean_0](../results/pilot/runs/5d25a4af27b5/figures/20261028T162341Z-ce0773/06_calibration_clean_0.png)

![07_error_detection_clean_0](../results/pilot/runs/5d25a4af27b5/figures/20261028T162341Z-ce0773/07_error_detection_clean_0.png)

![08_selective_prediction_clean_0](../results/pilot/runs/5d25a4af27b5/figures/20261028T162341Z-ce0773/08_selective_prediction_clean_0.png)

![06_calibration_blur_5](../results/pilot/runs/5d25a4af27b5/figures/20261028T162341Z-ce0773/06_calibration_blur_5.png)

![07_error_detection_blur_5](../results/pilot/runs/5d25a4af27b5/figures/20261028T162341Z-ce0773/07_error_detection_blur_5.png)

![08_selective_prediction_blur_5](../results/pilot/runs/5d25a4af27b5/figures/20261028T162341Z-ce0773/08_selective_prediction_blur_5.png)

![06_calibration_brightness_5](../results/pilot/runs/5d25a4af27b5/figures/20261028T162341Z-ce0773/06_calibration_brightness_5.png)

![07_error_detection_brightness_5](../results/pilot/runs/5d25a4af27b5/figures/20261028T162341Z-ce0773/07_error_detection_brightness_5.png)

![08_selective_prediction_brightness_5](../results/pilot/runs/5d25a4af27b5/figures/20261028T162341Z-ce0773/08_selective_prediction_brightness_5.png)

![06_calibration_contrast_5](../results/pilot/runs/5d25a4af27b5/figures/20261028T162341Z-ce0773/06_calibration_contrast_5.png)

![07_error_detection_contrast_5](../results/pilot/runs/5d25a4af27b5/figures/20261028T162341Z-ce0773/07_error_detection_contrast_5.png)

![08_selective_prediction_contrast_5](../results/pilot/runs/5d25a4af27b5/figures/20261028T162341Z-ce0773/08_selective_prediction_contrast_5.png)

![06_calibration_gaussian_noise_5](../results/pilot/runs/5d25a4af27b5/figures/20261028T162341Z-ce0773/06_calibration_gaussian_noise_5.png)

![07_error_detection_gaussian_noise_5](../results/pilot/runs/5d25a4af27b5/figures/20261028T162341Z-ce0773/07_error_detection_gaussian_noise_5.png)

![08_selective_prediction_gaussian_noise_5](../results/pilot/runs/5d25a4af27b5/figures/20261028T162341Z-ce0773/08_selective_prediction_gaussian_noise_5.png)

![01a_dataset_examples](../results/pilot/runs/9411ebb3ea90/figures/20261028T162411Z-e01802/01a_dataset_examples.png)

![01b_class_distribution](../results/pilot/runs/9411ebb3ea90/figures/20261028T162411Z-e01802/01b_class_distribution.png)

![03_corruption_severity](../results/pilot/runs/9411ebb3ea90/figures/20261028T162411Z-e01802/03_corruption_severity.png)

![02_training_convergence](../results/pilot/runs/9411ebb3ea90/figures/20261028T162411Z-e01802/02_training_convergence.png)

![04_accuracy_degradation](../results/pilot/runs/9411ebb3ea90/figures/20261028T162411Z-e01802/04_accuracy_degradation.png)

![05_confidence_reliability](../results/pilot/runs/9411ebb3ea90/figures/20261028T162411Z-e01802/05_confidence_reliability.png)

![09_method_comparison](../results/pilot/runs/9411ebb3ea90/figures/20261028T162411Z-e01802/09_method_comparison.png)

![06_calibration_clean_0](../results/pilot/runs/9411ebb3ea90/figures/20261028T162411Z-e01802/06_calibration_clean_0.png)

![07_error_detection_clean_0](../results/pilot/runs/9411ebb3ea90/figures/20261028T162411Z-e01802/07_error_detection_clean_0.png)

![08_selective_prediction_clean_0](../results/pilot/runs/9411ebb3ea90/figures/20261028T162411Z-e01802/08_selective_prediction_clean_0.png)

![06_calibration_blur_5](../results/pilot/runs/9411ebb3ea90/figures/20261028T162411Z-e01802/06_calibration_blur_5.png)

![07_error_detection_blur_5](../results/pilot/runs/9411ebb3ea90/figures/20261028T162411Z-e01802/07_error_detection_blur_5.png)

![08_selective_prediction_blur_5](../results/pilot/runs/9411ebb3ea90/figures/20261028T162411Z-e01802/08_selective_prediction_blur_5.png)

![06_calibration_brightness_5](../results/pilot/runs/9411ebb3ea90/figures/20261028T162411Z-e01802/06_calibration_brightness_5.png)

![07_error_detection_brightness_5](../results/pilot/runs/9411ebb3ea90/figures/20261028T162411Z-e01802/07_error_detection_brightness_5.png)

![08_selective_prediction_brightness_5](../results/pilot/runs/9411ebb3ea90/figures/20261028T162411Z-e01802/08_selective_prediction_brightness_5.png)

![06_calibration_contrast_5](../results/pilot/runs/9411ebb3ea90/figures/20261028T162411Z-e01802/06_calibration_contrast_5.png)

![07_error_detection_contrast_5](../results/pilot/runs/9411ebb3ea90/figures/20261028T162411Z-e01802/07_error_detection_contrast_5.png)

![08_selective_prediction_contrast_5](../results/pilot/runs/9411ebb3ea90/figures/20261028T162411Z-e01802/08_selective_prediction_contrast_5.png)

![06_calibration_gaussian_noise_5](../results/pilot/runs/9411ebb3ea90/figures/20261028T162411Z-e01802/06_calibration_gaussian_noise_5.png)

![07_error_detection_gaussian_noise_5](../results/pilot/runs/9411ebb3ea90/figures/20261028T162411Z-e01802/07_error_detection_gaussian_noise_5.png)

![08_selective_prediction_gaussian_noise_5](../results/pilot/runs/9411ebb3ea90/figures/20261028T162411Z-e01802/08_selective_prediction_gaussian_noise_5.png)

![01a_dataset_examples](../results/pilot/runs/a3dc5434183b/figures/20261028T162446Z-c932e4/01a_dataset_examples.png)

![01b_class_distribution](../results/pilot/runs/a3dc5434183b/figures/20261028T162446Z-c932e4/01b_class_distribution.png)

![03_corruption_severity](../results/pilot/runs/a3dc5434183b/figures/20261028T162446Z-c932e4/03_corruption_severity.png)

![02_training_convergence](../results/pilot/runs/a3dc5434183b/figures/20261028T162446Z-c932e4/02_training_convergence.png)

![04_accuracy_degradation](../results/pilot/runs/a3dc5434183b/figures/20261028T162446Z-c932e4/04_accuracy_degradation.png)

![05_confidence_reliability](../results/pilot/runs/a3dc5434183b/figures/20261028T162446Z-c932e4/05_confidence_reliability.png)

![09_method_comparison](../results/pilot/runs/a3dc5434183b/figures/20261028T162446Z-c932e4/09_method_comparison.png)

![06_calibration_clean_0](../results/pilot/runs/a3dc5434183b/figures/20261028T162446Z-c932e4/06_calibration_clean_0.png)

![07_error_detection_clean_0](../results/pilot/runs/a3dc5434183b/figures/20261028T162446Z-c932e4/07_error_detection_clean_0.png)

![08_selective_prediction_clean_0](../results/pilot/runs/a3dc5434183b/figures/20261028T162446Z-c932e4/08_selective_prediction_clean_0.png)

![06_calibration_blur_5](../results/pilot/runs/a3dc5434183b/figures/20261028T162446Z-c932e4/06_calibration_blur_5.png)

![07_error_detection_blur_5](../results/pilot/runs/a3dc5434183b/figures/20261028T162446Z-c932e4/07_error_detection_blur_5.png)

![08_selective_prediction_blur_5](../results/pilot/runs/a3dc5434183b/figures/20261028T162446Z-c932e4/08_selective_prediction_blur_5.png)

![06_calibration_brightness_5](../results/pilot/runs/a3dc5434183b/figures/20261028T162446Z-c932e4/06_calibration_brightness_5.png)

![07_error_detection_brightness_5](../results/pilot/runs/a3dc5434183b/figures/20261028T162446Z-c932e4/07_error_detection_brightness_5.png)

![08_selective_prediction_brightness_5](../results/pilot/runs/a3dc5434183b/figures/20261028T162446Z-c932e4/08_selective_prediction_brightness_5.png)

![06_calibration_contrast_5](../results/pilot/runs/a3dc5434183b/figures/20261028T162446Z-c932e4/06_calibration_contrast_5.png)

![07_error_detection_contrast_5](../results/pilot/runs/a3dc5434183b/figures/20261028T162446Z-c932e4/07_error_detection_contrast_5.png)

![08_selective_prediction_contrast_5](../results/pilot/runs/a3dc5434183b/figures/20261028T162446Z-c932e4/08_selective_prediction_contrast_5.png)

![06_calibration_gaussian_noise_5](../results/pilot/runs/a3dc5434183b/figures/20261028T162446Z-c932e4/06_calibration_gaussian_noise_5.png)

![07_error_detection_gaussian_noise_5](../results/pilot/runs/a3dc5434183b/figures/20261028T162446Z-c932e4/07_error_detection_gaussian_noise_5.png)

![08_selective_prediction_gaussian_noise_5](../results/pilot/runs/a3dc5434183b/figures/20261028T162446Z-c932e4/08_selective_prediction_gaussian_noise_5.png)

![01a_dataset_examples](../results/pilot/runs/e337964e428e/figures/20261028T162520Z-e3f12f/01a_dataset_examples.png)

![01b_class_distribution](../results/pilot/runs/e337964e428e/figures/20261028T162520Z-e3f12f/01b_class_distribution.png)

![03_corruption_severity](../results/pilot/runs/e337964e428e/figures/20261028T162520Z-e3f12f/03_corruption_severity.png)

![02_training_convergence](../results/pilot/runs/e337964e428e/figures/20261028T162520Z-e3f12f/02_training_convergence.png)

![04_accuracy_degradation](../results/pilot/runs/e337964e428e/figures/20261028T162520Z-e3f12f/04_accuracy_degradation.png)

![05_confidence_reliability](../results/pilot/runs/e337964e428e/figures/20261028T162520Z-e3f12f/05_confidence_reliability.png)

![09_method_comparison](../results/pilot/runs/e337964e428e/figures/20261028T162520Z-e3f12f/09_method_comparison.png)

![06_calibration_clean_0](../results/pilot/runs/e337964e428e/figures/20261028T162520Z-e3f12f/06_calibration_clean_0.png)

![07_error_detection_clean_0](../results/pilot/runs/e337964e428e/figures/20261028T162520Z-e3f12f/07_error_detection_clean_0.png)

![08_selective_prediction_clean_0](../results/pilot/runs/e337964e428e/figures/20261028T162520Z-e3f12f/08_selective_prediction_clean_0.png)

![06_calibration_blur_5](../results/pilot/runs/e337964e428e/figures/20261028T162520Z-e3f12f/06_calibration_blur_5.png)

![07_error_detection_blur_5](../results/pilot/runs/e337964e428e/figures/20261028T162520Z-e3f12f/07_error_detection_blur_5.png)

![08_selective_prediction_blur_5](../results/pilot/runs/e337964e428e/figures/20261028T162520Z-e3f12f/08_selective_prediction_blur_5.png)

![06_calibration_brightness_5](../results/pilot/runs/e337964e428e/figures/20261028T162520Z-e3f12f/06_calibration_brightness_5.png)

![07_error_detection_brightness_5](../results/pilot/runs/e337964e428e/figures/20261028T162520Z-e3f12f/07_error_detection_brightness_5.png)

![08_selective_prediction_brightness_5](../results/pilot/runs/e337964e428e/figures/20261028T162520Z-e3f12f/08_selective_prediction_brightness_5.png)

![06_calibration_contrast_5](../results/pilot/runs/e337964e428e/figures/20261028T162520Z-e3f12f/06_calibration_contrast_5.png)

![07_error_detection_contrast_5](../results/pilot/runs/e337964e428e/figures/20261028T162520Z-e3f12f/07_error_detection_contrast_5.png)

![08_selective_prediction_contrast_5](../results/pilot/runs/e337964e428e/figures/20261028T162520Z-e3f12f/08_selective_prediction_contrast_5.png)

![06_calibration_gaussian_noise_5](../results/pilot/runs/e337964e428e/figures/20261028T162520Z-e3f12f/06_calibration_gaussian_noise_5.png)

![07_error_detection_gaussian_noise_5](../results/pilot/runs/e337964e428e/figures/20261028T162520Z-e3f12f/07_error_detection_gaussian_noise_5.png)

![08_selective_prediction_gaussian_noise_5](../results/pilot/runs/e337964e428e/figures/20261028T162520Z-e3f12f/08_selective_prediction_gaussian_noise_5.png)

![01a_dataset_examples](../results/pilot/runs/ee2ed85c603f/figures/20261028T162551Z-877182/01a_dataset_examples.png)

![01b_class_distribution](../results/pilot/runs/ee2ed85c603f/figures/20261028T162551Z-877182/01b_class_distribution.png)

![03_corruption_severity](../results/pilot/runs/ee2ed85c603f/figures/20261028T162551Z-877182/03_corruption_severity.png)

![02_training_convergence](../results/pilot/runs/ee2ed85c603f/figures/20261028T162551Z-877182/02_training_convergence.png)

![04_accuracy_degradation](../results/pilot/runs/ee2ed85c603f/figures/20261028T162551Z-877182/04_accuracy_degradation.png)

![05_confidence_reliability](../results/pilot/runs/ee2ed85c603f/figures/20261028T162551Z-877182/05_confidence_reliability.png)

![09_method_comparison](../results/pilot/runs/ee2ed85c603f/figures/20261028T162551Z-877182/09_method_comparison.png)

![06_calibration_clean_0](../results/pilot/runs/ee2ed85c603f/figures/20261028T162551Z-877182/06_calibration_clean_0.png)

![07_error_detection_clean_0](../results/pilot/runs/ee2ed85c603f/figures/20261028T162551Z-877182/07_error_detection_clean_0.png)

![08_selective_prediction_clean_0](../results/pilot/runs/ee2ed85c603f/figures/20261028T162551Z-877182/08_selective_prediction_clean_0.png)

![06_calibration_blur_5](../results/pilot/runs/ee2ed85c603f/figures/20261028T162551Z-877182/06_calibration_blur_5.png)

![07_error_detection_blur_5](../results/pilot/runs/ee2ed85c603f/figures/20261028T162551Z-877182/07_error_detection_blur_5.png)

![08_selective_prediction_blur_5](../results/pilot/runs/ee2ed85c603f/figures/20261028T162551Z-877182/08_selective_prediction_blur_5.png)

![06_calibration_brightness_5](../results/pilot/runs/ee2ed85c603f/figures/20261028T162551Z-877182/06_calibration_brightness_5.png)

![07_error_detection_brightness_5](../results/pilot/runs/ee2ed85c603f/figures/20261028T162551Z-877182/07_error_detection_brightness_5.png)

![08_selective_prediction_brightness_5](../results/pilot/runs/ee2ed85c603f/figures/20261028T162551Z-877182/08_selective_prediction_brightness_5.png)

![06_calibration_contrast_5](../results/pilot/runs/ee2ed85c603f/figures/20261028T162551Z-877182/06_calibration_contrast_5.png)

![07_error_detection_contrast_5](../results/pilot/runs/ee2ed85c603f/figures/20261028T162551Z-877182/07_error_detection_contrast_5.png)

![08_selective_prediction_contrast_5](../results/pilot/runs/ee2ed85c603f/figures/20261028T162551Z-877182/08_selective_prediction_contrast_5.png)

![06_calibration_gaussian_noise_5](../results/pilot/runs/ee2ed85c603f/figures/20261028T162551Z-877182/06_calibration_gaussian_noise_5.png)

![07_error_detection_gaussian_noise_5](../results/pilot/runs/ee2ed85c603f/figures/20261028T162551Z-877182/07_error_detection_gaussian_noise_5.png)

![08_selective_prediction_gaussian_noise_5](../results/pilot/runs/ee2ed85c603f/figures/20261028T162551Z-877182/08_selective_prediction_gaussian_noise_5.png)

![10_paired_differences_fashion_mnist_cnn](../results/pilot/figures/statistics-20261028T162617Z-697318/10_paired_differences_fashion_mnist_cnn.png)

### Run configuration appendix

#### 5d25a4af27b5
```json
{
  "profile": "development",
  "dataset": "fashion_mnist",
  "model": "cnn",
  "seed": 11,
  "data": {
    "root": "data",
    "download": false,
    "validation_fraction": 0.1,
    "split_seed": 2026,
    "train_limit": 3000,
    "val_limit": 600,
    "test_limit": 1000
  },
  "model_options": {
    "dropout": 0.25,
    "batch_norm": true
  },
  "training": {
    "epochs": 3,
    "batch_size": 128,
    "lr": 0.001,
    "weight_decay": 0.0001,
    "patience": 5,
    "device": "cpu",
    "workers": 0,
    "threads": 4,
    "deterministic": true
  },
  "evaluation": {
    "batch_size": 128,
    "mc_passes": 5,
    "bins": 15,
    "corruptions": [
      "gaussian_noise",
      "blur",
      "brightness",
      "contrast"
    ],
    "severities": [
      1,
      2,
      3,
      4,
      5
    ],
    "corruption_seed": 1729
  },
  "output": "results/pilot"
}
```

#### 9411ebb3ea90
```json
{
  "profile": "development",
  "dataset": "fashion_mnist",
  "model": "cnn",
  "seed": 44,
  "data": {
    "root": "data",
    "download": false,
    "validation_fraction": 0.1,
    "split_seed": 2026,
    "train_limit": 3000,
    "val_limit": 600,
    "test_limit": 1000
  },
  "model_options": {
    "dropout": 0.25,
    "batch_norm": true
  },
  "training": {
    "epochs": 3,
    "batch_size": 128,
    "lr": 0.001,
    "weight_decay": 0.0001,
    "patience": 5,
    "device": "cpu",
    "workers": 0,
    "threads": 4,
    "deterministic": true
  },
  "evaluation": {
    "batch_size": 128,
    "mc_passes": 5,
    "bins": 15,
    "corruptions": [
      "gaussian_noise",
      "blur",
      "brightness",
      "contrast"
    ],
    "severities": [
      1,
      2,
      3,
      4,
      5
    ],
    "corruption_seed": 1729
  },
  "output": "results/pilot"
}
```

#### a3dc5434183b
```json
{
  "profile": "development",
  "dataset": "fashion_mnist",
  "model": "cnn",
  "seed": 55,
  "data": {
    "root": "data",
    "download": false,
    "validation_fraction": 0.1,
    "split_seed": 2026,
    "train_limit": 3000,
    "val_limit": 600,
    "test_limit": 1000
  },
  "model_options": {
    "dropout": 0.25,
    "batch_norm": true
  },
  "training": {
    "epochs": 3,
    "batch_size": 128,
    "lr": 0.001,
    "weight_decay": 0.0001,
    "patience": 5,
    "device": "cpu",
    "workers": 0,
    "threads": 4,
    "deterministic": true
  },
  "evaluation": {
    "batch_size": 128,
    "mc_passes": 5,
    "bins": 15,
    "corruptions": [
      "gaussian_noise",
      "blur",
      "brightness",
      "contrast"
    ],
    "severities": [
      1,
      2,
      3,
      4,
      5
    ],
    "corruption_seed": 1729
  },
  "output": "results/pilot"
}
```

#### e337964e428e
```json
{
  "profile": "development",
  "dataset": "fashion_mnist",
  "model": "cnn",
  "seed": 22,
  "data": {
    "root": "data",
    "download": false,
    "validation_fraction": 0.1,
    "split_seed": 2026,
    "train_limit": 3000,
    "val_limit": 600,
    "test_limit": 1000
  },
  "model_options": {
    "dropout": 0.25,
    "batch_norm": true
  },
  "training": {
    "epochs": 3,
    "batch_size": 128,
    "lr": 0.001,
    "weight_decay": 0.0001,
    "patience": 5,
    "device": "cpu",
    "workers": 0,
    "threads": 4,
    "deterministic": true
  },
  "evaluation": {
    "batch_size": 128,
    "mc_passes": 5,
    "bins": 15,
    "corruptions": [
      "gaussian_noise",
      "blur",
      "brightness",
      "contrast"
    ],
    "severities": [
      1,
      2,
      3,
      4,
      5
    ],
    "corruption_seed": 1729
  },
  "output": "results/pilot"
}
```

#### ee2ed85c603f
```json
{
  "profile": "development",
  "dataset": "fashion_mnist",
  "model": "cnn",
  "seed": 33,
  "data": {
    "root": "data",
    "download": false,
    "validation_fraction": 0.1,
    "split_seed": 2026,
    "train_limit": 3000,
    "val_limit": 600,
    "test_limit": 1000
  },
  "model_options": {
    "dropout": 0.25,
    "batch_norm": true
  },
  "training": {
    "epochs": 3,
    "batch_size": 128,
    "lr": 0.001,
    "weight_decay": 0.0001,
    "patience": 5,
    "device": "cpu",
    "workers": 0,
    "threads": 4,
    "deterministic": true
  },
  "evaluation": {
    "batch_size": 128,
    "mc_passes": 5,
    "bins": 15,
    "corruptions": [
      "gaussian_noise",
      "blur",
      "brightness",
      "contrast"
    ],
    "severities": [
      1,
      2,
      3,
      4,
      5
    ],
    "corruption_seed": 1729
  },
  "output": "results/pilot"
}
```

Aggregate tables: [seed summaries](../results/pilot/reports/20261028T162617Z-86a5cd/seed_summary.csv) and [paired comparisons](../results/pilot/reports/20261028T162617Z-86a5cd/paired_comparisons.csv).
