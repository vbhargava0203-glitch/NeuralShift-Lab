# NeuralShift Lab - Technical Handoff

Delivered 29 October 2026. **Working research software and a completed reduced five-seed pilot. The full research program is not complete.**

## A. Implementation status

Implemented: official Fashion-MNIST/CIFAR-10 loading, seeded splits, CNN and small-image ResNet-18, AdamW training, scheduler, early stopping, checkpoint/resume, four five-level corruptions, MSP/entropy/MC dropout, validation-only temperature scaling, classification/calibration/error/shift/selective metrics, paired seed statistics, matrix execution, cached MC-pass ablation, automated figures/reports, CLI, tests, notebooks, CI, and publication materials.

Remaining execution: full-data study, CIFAR-10 training, trained ResNet comparisons, dropout-probability and architecture ablations, CUDA testing, and GitHub-hosted CI. The training-factor ablation runner exists but those ablations have not run. The four notebooks contain runnable analysis code but were syntax-checked rather than executed as Jupyter notebooks.

## B. Test evidence

- `python -m pytest -q`: **74 passed**, zero failures, in 32.02 seconds. Local run used an explicit workspace `--basetemp` because the sandbox's default temporary directory was not writable.
- `python -m ruff check src tests scripts`: passed.
- `python -m mypy src/neuralshift/config.py src/neuralshift/models`: passed, five source files. This is scoped type checking, not whole-repository type coverage.
- `python -m pip check`: no broken requirements.
- `python -m pip install -e . --no-deps --no-build-isolation`: succeeded in the isolated CPython 3.12.14 environment.
- End-to-end real smoke and all five pilot run validations passed, including scalar metric recomputation from cached predictions.
- Tests cover recovery from evaluation interruption, frozen batch normalization during MC dropout, no test-data access during training, and deliberate artifact tampering.

The local setup initially encountered a temporary-directory permission issue and duplicate NumPy/Pillow metadata from concurrent installation. Both were fixed. Runtime and metadata versions now agree; exact audit corrections are recorded in the manifests. A new clean installation from the final lock on another machine has not been executed.

See [evidence logs](evidence/) and [environment audit](evidence/environment_audit.json).

## C. Experiment evidence

Hardware: Windows CPU execution, PyTorch 2.14.1+cpu and torchvision 0.29.1+cpu. CUDA availability was false for this installed build; no claim is made about physical GPU presence.

| Profile | Dataset / model | Seeds | Training / validation / test | Epochs | MC passes | Conditions |
| --- | --- | --- | --- | --- | --- | --- |
| Smoke | Fashion-MNIST / CNN | 42 | 512 / 128 / 128 | 1 | 3 | 9 |
| Reduced pilot | Fashion-MNIST / CNN | 11,22,33,44,55 | 3000 / 600 / 1000 | 3 | 5 | 21 per seed |

Six validated runs exist. Pilot training epoch time totals 49.3 seconds; recorded pilot evaluation time totals 257.8 seconds. These timings exclude setup, data download, and report generation and are not a full-study runtime forecast.

Run IDs and paths: [execution ledger](experiments/manifests/execution_ledger.json). Prediction caches, selected checkpoints, validation logits, sample IDs, metric CSV/JSON, and hashes are preserved under `results/pilot/runs/` and `results/smoke/runs/`.

## D. Research outputs

- **735 pilot metric rows**: five seeds x 21 conditions x seven score/calibration combinations.
- 111 generated pilot PNG figures, accompanied by vector PDFs; all ten requested figure families are represented, including paired statistical comparisons.
- [Printable manuscript](paper/manuscript.pdf), [full Markdown manuscript](paper/manuscript.md), [seed summaries](results/pilot/reports/20261028T162617Z-86a5cd/seed_summary.csv), and [paired comparisons](results/pilot/reports/20261028T162617Z-86a5cd/paired_comparisons.csv).
- Nested MC sample-count ablation at 2, 3, and 5 passes on the seed-11 checkpoint; no retraining. See `results/pilot/runs/5d25a4af27b5/ablations/`.
- Portfolio summary, verified highlights, selected figures, demonstration script, suggested LinkedIn post, and release checklist in `showcase/`.

Observed clean accuracy averaged **74.38%**, sample SD **0.49 percentage points**, over five pilot seeds. MC predictive entropy minus MSP mean error-AUROC was **-0.0429**, with exploratory paired-seed bootstrap 95% interval **[-0.0484, -0.0352]**, averaging all 21 conditions within each seed. This pilot did not support an MC-entropy advantage. No universal superiority or confirmatory significance is claimed.

## E. Reproducibility

From the extracted `NeuralShift-Lab` directory, using CPython 3.12:

```bash
python -m venv .venv
# PowerShell activation:
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python -m pytest -q
python -m neuralshift.cli download-data --config configs/cpu_smoke.yaml
python -m neuralshift.cli experiment --config configs/cpu_smoke.yaml
python -m neuralshift.cli validate-results --results results/smoke
python -m neuralshift.cli experiment --config configs/pilot_multiseed.yaml
python -m neuralshift.cli validate-results --results results/pilot
python -m neuralshift.cli report --results results/pilot
python scripts/export_pdf.py --results results/pilot --output paper/manuscript-rerun.pdf
```

Individual training/evaluation:

```bash
python -m neuralshift.cli train --config configs/fashion_mnist.yaml
python -m neuralshift.cli evaluate --config configs/fashion_mnist.yaml --checkpoint PATH_TO_PRINTED_RUN/best.pt
python -m neuralshift.cli train --config configs/fashion_mnist.yaml --resume PATH_TO_PRINTED_RUN
```

The experiment command reuses an identical recorded run. To create an independent run, prespecify a different seed or output directory. Raw data is intentionally absent from the delivery ZIP; download it to retrain. Saved predictions support offline metric validation without downloading data. Complete-run `last.pt` optimizer snapshots are omitted from the ZIP; selected `best.pt` checkpoints are included.

## F. GitHub readiness

The package includes MIT licensing, provisional collective citation metadata, contribution guidelines, code of conduct, Git ignore rules, and two GitHub Actions workflows. No repository was published, no external service deployed, and no GitHub checks are claimed as executed. Confirm authorship and the public repository URL before release. Raw datasets and dependency environments are excluded from the ZIP. Store prediction caches/checkpoints in a release artifact or explicit Git LFS workflow; they are ignored in source Git history.

## G. Scientific limitations

Only the reduced Fashion-MNIST CNN protocol has empirical evidence. Five seeds, 1,000 fixed test examples, short training, and five MC passes limit conclusions. The four custom corruptions do not constitute semantic OOD or a comprehensive real-world shift benchmark. Intervals reflect training-seed variability, not resampling uncertainty for the test population. Validation is shared by model selection and temperature fitting. All current hypothesis comparisons are exploratory and uncorrected; multiple-testing and practical-effect plans must precede confirmatory work. More detail: [limitations](paper/limitations.md).

## H. Next research milestone

Run the full-data five-seed Fashion-MNIST CNN protocol:

```bash
python -m neuralshift.cli estimate --config configs/research_fashion_cnn.yaml
python -m neuralshift.cli experiment --config configs/research_fashion_cnn.yaml
python -m neuralshift.cli validate-results --results results/research-fashion-cnn
python -m neuralshift.cli report --results results/research-fashion-cnn
```

Acceptance: five independent validated runs, complete 21-condition matched caches, full official held-out test evaluation, 30 MC passes, paired seed-level comparisons, and a report that preserves negative or inconclusive findings. This milestone is configured but **not executed** in this delivery.
