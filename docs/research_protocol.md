# Prespecified research protocol

Primary hypothesis: some uncertainty-aware scores may rank errors more reliably than 1-MSP under selected corruptions. Improvement is not assumed.

Train each dataset/model/seed combination once. Split only the official training data. Select checkpoints and fit temperatures using validation data. Test labels must not guide training, checkpoint selection, hyperparameters, temperature, or deployed thresholds. Freeze the protocol before evaluation.

The planned matrix uses two datasets, two architectures, at least five independent seeds where feasible, and clean plus twenty corrupted conditions. Every uncertainty score uses the same deterministic prediction errors. MC mean classification is not substituted into the primary comparison.

Report condition-level metrics before aggregation. Compare AUROC to raw MSP on matched training seeds and conditions; also report AUPRC, ECE, NLL, Brier, and AURC. Seed-level intervals condition on the fixed evaluation sample. Repeated stochastic passes and corruptions are not independent training runs.

Current broad comparisons are exploratory. A confirmatory follow-up must prespecify its family of contrasts and a multiplicity policy such as Holm correction. Statistical and practical importance differ. Keep failed runs in the ledger; do not delete unfavorable results.

Controlled ablations vary MC pass count using common prefix draws, dropout probability with retraining, architecture, severity, calibration, or training seed one factor at a time. Architecture differences may be confounded by optimization suitability and dropout placement. Never tune on the final test set and claim the selected score is unbiased.

Smoke verifies software execution. Development provides limited observations. Only executed, validated multi-seed research runs support broader conclusions.
