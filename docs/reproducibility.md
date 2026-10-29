# Reproducibility

Create a clean environment from requirements-lock.txt, install the package, and retain pip freeze. The lock is the actually exercised Windows CPU environment, not proof of every OS/CUDA combination.

Retain resolved configuration, split metadata, source hashes, history, checkpoint, calibration provenance, prediction caches, identifiers, and metrics with the manifest. Git revision is recorded when available. Missing provenance is not invented.

Python, NumPy, PyTorch, loader generators, and workers are seeded. Strict deterministic mode can reject unsupported operations. Hardware and library versions can still change numerical behavior. Cross-platform bitwise identity is not guaranteed.

Use train --config CONFIG --resume RUN_DIR for interrupted training. Do not mix checkpoints and metrics from different runs. Validate results before generating figures or reports. Derived exports use new directories. Missing or insufficient statistical evidence is recorded in status files.

Keep datasets, environments, and large checkpoints out of source Git. Review manifests for machine-specific paths before public release, preserve licenses, and attach a compact evidence bundle through appropriate artifact hosting.
