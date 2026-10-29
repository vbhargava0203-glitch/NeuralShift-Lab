# Verified research highlights

- Five independent training seeds: 11, 22, 33, 44, 55.
- 3,000 training, 600 validation, and 1,000 held-out official test images per pilot run.
- Three training epochs and five MC passes; 21 conditions per seed; 735 metric rows.
- Validation-only checkpoint selection and temperature fitting.
- Each saved scalar metric recomputed from hashed prediction caches.
- Raw/temp deterministic scores and raw MC scores share one error target.

In the reduced five-seed pilot, clean accuracy averaged 0.7438 (sample SD 0.0049). MC predictive entropy had mean error-detection AUROC difference -0.0429 relative to MSP, with a paired seed-bootstrap 95% interval [-0.0484, -0.0352], after equally averaging the 21 evaluated conditions within each seed. Thus the proposed MC-entropy advantage was not observed in this pilot. This negative result is conditional on the small CNN, short training, fixed test subset, selected corruptions, and five MC passes; it does not establish a general ranking of uncertainty methods. Comparisons remain exploratory.

Evidence: ../results/pilot/reports/20261028T162617Z-86a5cd/validated_metrics.csv and paired_comparisons.csv. Source manifests are under ../results/pilot/runs/. Full benchmark and architecture/dataset comparisons remain pending.
