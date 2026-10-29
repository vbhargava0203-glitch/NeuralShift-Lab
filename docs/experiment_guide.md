# Experiment guide

From the installed repository root:

```bash
python -m neuralshift.cli download-data --config configs/cpu_smoke.yaml
python -m neuralshift.cli estimate --config configs/cpu_smoke.yaml
python -m neuralshift.cli experiment --config configs/cpu_smoke.yaml
python -m neuralshift.cli validate-results --results results/smoke/runs/RUN_ID
python -m neuralshift.cli report --results results/smoke/runs/RUN_ID
```

Replace RUN_ID with the actual printed identifier. Use fashion_mnist.yaml for development, cifar10.yaml for the second dataset, and experiments.yaml for the planned matrix. Read the configuration and estimate before launching full research.

Separate execution uses train --config CONFIG, then evaluate --config CONFIG --checkpoint CHECKPOINT. Resume interrupted training by adding --resume RUN_DIR to train. Completed runs are preserved. Every command supports --help.

Figures alone: figures --results RUN_DIR. A root containing matched completed runs can be used for report aggregation. Duplicate seed-condition rows are not independent replicates; statistical plots omit such groups.

Optional PDF manuscript workflow: install Pandoc and XeLaTeX separately, then run from the generated report directory:

```bash
pandoc manuscript.md --from markdown --standalone --pdf-engine=xelatex -V geometry:margin=22mm -o manuscript.pdf
```

Pandoc/XeLaTeX availability and final PDF rendering are separate checks, not implied by Python tests. Inspect exported pages before publication. The Python pipeline generates Markdown and PNG/vector-PDF figures directly.

Missing datasets must be downloaded from real documented sources; synthetic substitutes belong only in tests. Choose CPU explicitly if CUDA is unavailable and preserve the changed profile.

The delivered package also includes a tested ReportLab exporter: `python scripts/export_pdf.py --results results/pilot --output paper/manuscript-new.pdf`. See pdf_export.md. Use `make report RESULTS=results/smoke` after the smoke profile, or the default results/pilot after the multi-seed pilot.
