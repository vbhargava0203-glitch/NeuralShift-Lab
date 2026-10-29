# NeuralShift Lab

**Understanding machine learning reliability under distribution shift.**

A reproducible Python research platform for comparing classification accuracy, confidence calibration, error detection, and selective prediction under controlled image corruptions. This is an empirical undergraduate research study using established methods; it does not claim a novel algorithm, peer review, or deployment readiness.

**Research question:** How reliably can confidence and uncertainty scores identify incorrect image predictions when the input distribution changes?

## Evidence status

Implementation and executed experiments are separate deliverables. The release execution ledger records completed checks and actual runs. The full two-dataset, two-model, five-seed research matrix is **pending** until its artifacts are validated. A smoke run does not establish the research hypothesis.

The delivered evidence is described in [HANDOFF.md](HANDOFF.md) and [the manuscript](paper/manuscript.md). Regenerate actual results from a completed run:

```bash
python -m neuralshift.cli validate-results --results results/smoke/runs/RUN_ID
python -m neuralshift.cli report --results results/smoke/runs/RUN_ID
```

Reports include computed tables, figures, and manifest links. Missing evidence raises an error instead of producing invented plots.

## Install

Use CPython 3.12, the locally exercised version, and run from this directory:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux instead: source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python -m neuralshift.cli --help
pytest
ruff check .
```

The lock records the tested Windows CPU environment. CI describes additional checks; it does not prove they ran. For CUDA use a fresh environment and an official matching PyTorch/torchvision build, then install the scientific dependencies, freeze, and retest. Do not apply the CPU lock over a CUDA environment.

## Quick start with real data

```bash
python -m neuralshift.cli download-data --config configs/cpu_smoke.yaml
python -m neuralshift.cli estimate --config configs/cpu_smoke.yaml
python -m neuralshift.cli experiment --config configs/cpu_smoke.yaml
```

Replace `RUN_ID` below with the printed run identifier:

```bash
python -m neuralshift.cli validate-results --results results/smoke/runs/RUN_ID
python -m neuralshift.cli figures --results results/smoke/runs/RUN_ID
python -m neuralshift.cli report --results results/smoke/runs/RUN_ID
```

The smoke profile uses real Fashion-MNIST and explicitly recorded subsets. Synthetic fixtures are restricted to software tests. The development profile increases the workload:

```bash
python -m neuralshift.cli experiment --config configs/fashion_mnist.yaml
```

Individual training and evaluation are available:

```bash
python -m neuralshift.cli train --config configs/fashion_mnist.yaml
python -m neuralshift.cli evaluate --config configs/fashion_mnist.yaml --checkpoint PATH_TO_CHECKPOINT
python -m neuralshift.cli train --config configs/fashion_mnist.yaml --resume PATH_TO_RUN_DIRECTORY
```

Resume applies to interrupted training. Completed results are not silently replaced. Every subcommand supports `--help`.

## Full experiments

```bash
python -m neuralshift.cli download-data --config configs/fashion_mnist.yaml
python -m neuralshift.cli download-data --config configs/cifar10.yaml
python -m neuralshift.cli estimate --config configs/experiments.yaml
python -m neuralshift.cli experiment --config configs/experiments.yaml
```

The planned matrix contains Fashion-MNIST and CIFAR-10, CNN and adapted ResNet-18, independent seeds, one clean condition, four corruption families at five severities, raw/temperature-scaled deterministic scores, and MC dropout. Each dataset/model/seed combination trains once and reuses its validation-selected checkpoint. Review the estimate before full execution. No full-matrix runtime is claimed without measurement.

## Methods and metrics

| Component | Implementation |
| --- | --- |
| Data | Official torchvision datasets, seeded train/validation splits, isolated official test partition |
| CNN | 32/64-channel 3x3 blocks, optional BN, ReLU, pooling, dropout, adaptive 4x4 pooling, 128-unit head |
| ResNet-18 | From scratch, 3x3 stride-one stem, no initial max pool, dropout before classifier |
| Training | Cross entropy, AdamW, scheduler, early stopping, validation checkpoint selection, gradient monitoring |
| Shift | Gaussian noise, blur, additive brightening, contrast reduction before normalization |
| Scores | 1-MSP, entropy, MC predictive entropy, mutual-information proxy, probability variance |
| Calibration | Positive scalar temperature fitted using validation logits and labels |
| Classification | Accuracy, macro F1, precision/recall, confusion matrix, NLL, multiclass Brier score |
| Calibration metrics | 15-bin ECE, NLL, Brier, reliability diagrams |
| Error detection | Incorrect is positive; AUROC, average precision/AUPRC, FPR at target TPR |
| Selective prediction | Coverage, accepted-example error risk, AURC with stable ties |

The primary comparison fixes deterministic predictions for every uncertainty score. MC scoring does not silently change the error target. Calibration, error ranking, and semantic OOD detection are distinct. See [methodology](docs/methodology.md).

## Architecture

```text
configs/                     execution profiles and matrix
src/neuralshift/
  datasets/ models/          sources and architectures
  training/                  optimizer, checkpointing, stopping
  corruptions/ uncertainty/  image transforms and scores
  evaluation/                metrics and paired statistics
  experiments/               reusable orchestration
  visualization/             validated figures and reports
  utils/                     atomic artifacts and validation
tests/                       software fixtures and integration checks
results/smoke/runs/RUN_ID/          manifest, checkpoint, caches, metrics
paper/ docs/ notebooks/       methodology, protocol, analysis
showcase/                    technical portfolio materials
```

Each run records resolved configuration, seeds, dataset metadata, hardware/software, execution status, and checkpoint checksum. Prediction caches retain sample identities. Figure and report exports are versioned. Raw datasets and large artifacts are excluded from source Git history.

## Research questions

1. How do accuracy and confidence change with corruption severity?
2. Which scores best rank errors on the same checkpoint and examples?
3. Does clean-validation temperature scaling transfer to corruption?
4. Do effects persist across datasets, architectures, and seeds?
5. How does abstention change selective risk?
6. Are paired differences practically meaningful and stable?

Use paired seed-level comparisons. Distinguish training variation from test-sample variation; dropout passes are not independent runs. Retain negative findings. Exploratory comparisons do not establish statistical significance.

## Hardware guidance

CPU execution is supported. Reduced profiles limit examples, epochs, and stochastic passes. MC evaluation scales approximately with the number of passes. Stochastic caches require about `4 * passes * examples * classes` bytes per condition before compression. Add logits and checkpoints. Use `estimate` and measured smoke time before launching larger work. CUDA execution and cross-platform bitwise determinism are not assumed.

## Limitations, license, citation

Four custom corruptions cover a narrow subset of covariate shifts. Reduced-data results do not imply full-benchmark performance. ECE depends on bins and sample count; MC dropout is approximate. Full-scale experiments and semantic OOD evaluation remain separate milestones. See [limitations](paper/limitations.md).

Code uses the [MIT license](LICENSE). Dependencies and datasets retain separate terms; the code license does not grant image redistribution rights. Raw data and pretrained weights are not bundled by default. Use [CITATION.cff](CITATION.cff) for software citation and cite the [original methods](paper/references.bib). No DOI, institution, or publication venue is claimed.

Read [experiment guide](docs/experiment_guide.md), [reproducibility](docs/reproducibility.md), [schema](docs/results_schema.md), [protocol](docs/research_protocol.md), and [contributing](CONTRIBUTING.md).

## Executed pilot

In the reduced five-seed pilot, clean accuracy averaged 0.7438 (sample SD 0.0049). MC predictive entropy had mean error-detection AUROC difference -0.0429 relative to MSP, with a paired seed-bootstrap 95% interval [-0.0484, -0.0352], after equally averaging the 21 evaluated conditions within each seed. Thus the proposed MC-entropy advantage was not observed in this pilot. This negative result is conditional on the small CNN, short training, fixed test subset, selected corruptions, and five MC passes; it does not establish a general ranking of uncertainty methods. Comparisons remain exploratory.

![Observed paired comparisons](showcase/figures/10_paired_differences_fashion_mnist_cnn.png)

These text values were generated from the validated pilot CSV files. Reproduce the five-seed pilot with:

```bash
python -m neuralshift.cli experiment --config configs/pilot_multiseed.yaml
python -m neuralshift.cli report --results results/pilot
python -m neuralshift.cli ablate --results results/pilot/runs/5d25a4af27b5 --mc-passes 2 3 5
```

For a fresh run, use its printed run ID for ablation. Training-factor ablations are defined in configs/ablations.yaml and run using `python scripts/run_ablations.py --config configs/ablations.yaml`; use `--estimate-only` first. See [PDF export](docs/pdf_export.md).
