# Results schema

Each run lives under results/smoke/runs/RUN_ID.

| Artifact | Meaning |
| --- | --- |
| manifest.json | ID, status, resolved configuration, environment, data metadata, timestamps, artifact checksums |
| history.csv | Epoch losses, accuracies, learning rate, duration, gradient norm |
| checkpoint | Model and training/resume state selected only by validation |
| temperature.json | Positive scalar and validation fitting provenance |
| metrics.csv | Dataset, model, seed, corruption, severity, method, calibration, sample count, computed metrics |
| predictions/NAME_SEVERITY.npz | logits (N,K), labels (N), ids (N), mc_probabilities (S,N,K) |
| samples.npz | Real images (N,C,H,W) in [0,1] and labels |
| figures/EXPORT_ID | PNG, vector PDF, source/status manifests |
| reports/EXPORT_ID | Markdown manuscript, combined metrics, source manifest |

Scalar metrics include accuracy, macro_f1, nll, brier, ece, mean_confidence, auroc, auprc, fpr_at_tpr, aurc. Undefined detection values are missing, not imputed. Structured classwise arrays are stored separately where available.

Logical row key: dataset/model/seed/corruption/severity/method/calibration within one protocol. A seed alone does not identify a protocol. Every score shares deterministic errors. Raw and temperature-scaled deterministic scores have separate rows; MC scores are raw.

Validate before publication. Inconsistent shapes, identifiers, checksums, or missing evidence must be rejected. A missing plot is not a completed negative result; read its status metadata.
