# Architecture

The CLI resolves configuration and seeds execution, then constructs data and model modules. Training sees only the official training partition, split into training and validation. Validation loss selects the checkpoint. Evaluation fits a temperature on validation logits before held-out test evaluation.

Corruptions operate on BCHW floating-point pixels in [0,1] before normalization. Sample-specific noise seeds preserve reproducibility across batching. The same checkpoint and examples serve every score. MC dropout enables dropout while keeping batch-normalization statistics frozen.

Metrics, prediction caches, and provenance are persisted together. Validation gates figures and reports. Visualizations never create substitute scientific data. Versioned exports preserve prior derived artifacts. Notebooks consume results; the CLI remains the reproducible execution path.
