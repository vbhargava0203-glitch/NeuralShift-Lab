# Reproducible manuscript export

Generate the report from verified runs, then render a compact printable edition with all narrative sections, clean-condition seed summaries, and ten selected figure panels:

```bash
python -m neuralshift.cli report --results results/pilot
python scripts/export_pdf.py --results results/pilot --output paper/manuscript.pdf
```

ReportLab is pinned in requirements-lock.txt. The exporter refuses to overwrite an existing PDF; use a new filename for another revision. The full Markdown report contains every per-run figure and configuration. The PDF selects one clearly identified run for descriptive per-run panels and uses all matched seeds for the statistical panel. Exported plots also include vector PDF versions.
