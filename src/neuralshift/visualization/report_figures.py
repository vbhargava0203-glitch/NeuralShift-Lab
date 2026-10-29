"""Rebuild figures and a research manuscript from validated run directories.

Derived outputs receive unique directories. Reports never infer that a configured
experiment ran: their evidence is restricted to validated manifests and caches.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import softmax, xlogy
from sklearn.metrics import roc_curve

from .calibration_plots import reliability_diagram
from .plots import COLORS, configure_style, save_figure, table_markdown
from .robustness_plots import robustness_figures

TITLE = "Evaluating Predictive Reliability Under Controlled Distribution Shifts in Deep Neural Networks"
REFERENCES = """- Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks*. ICML. https://proceedings.mlr.press/v70/guo17a.html
- Gal, Y., and Ghahramani, Z. (2016). *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning*. ICML. https://proceedings.mlr.press/v48/gal16.html
- Hendrycks, D., and Gimpel, K. (2017). *A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks*. ICLR. https://arxiv.org/abs/1610.02136
- Hendrycks, D., and Dietterich, T. (2019). *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations*. ICLR. https://arxiv.org/abs/1903.12261
- Xiao, H., Rasul, K., and Vollgraf, R. (2017). *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms*. https://arxiv.org/abs/1708.07747
- Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images*. Technical report, University of Toronto. https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
- He, K., Zhang, X., Ren, S., and Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR. https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html
"""


def _stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:6]


def discover_runs(results_path):
    """Require complete, valid runs; incomplete sibling runs are reported as errors."""
    root = Path(results_path).resolve()
    if (root / "manifest.json").exists():
        runs = [root]
    else:
        runs = sorted({p.parent for p in root.rglob("manifest.json")})
    if not runs:
        raise ValueError(f"No completed experiment with manifest.json and metrics.csv under {root}")
    from neuralshift.utils.validation import validate_run
    for run in runs:
        validate_run(run)
    return runs


def _scores(cache):
    p = softmax(cache["logits"], axis=1)
    scores = {"msp": 1 - p.max(1), "entropy": -xlogy(p, p).sum(1)}
    if "mc_probabilities" in cache:
        samples = cache["mc_probabilities"]
        mean = samples.mean(0)
        predictive = -xlogy(mean, mean).sum(1)
        expected = -xlogy(samples, samples).sum(2).mean(0)
        scores.update(mc_entropy=predictive, mc_mi=np.maximum(predictive - expected, 0),
                      mc_variance=samples.var(0).mean(1))
    return p, scores


def _dataset_figures(run, output, title):
    path = run / "samples.npz"
    if not path.exists():
        return [], ["Dataset overview and corruption examples unavailable: samples.npz absent."]
    files = []
    with np.load(path, allow_pickle=False) as samples:
        images, labels = samples["images"], samples["labels"]
    n = min(12, len(images))
    fig, axes = plt.subplots(2, 6, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < n:
            image = images[i].transpose(1, 2, 0)
            ax.imshow(image[..., 0] if image.shape[-1] == 1 else image,
                      cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"Class {int(labels[i])}")
    fig.suptitle(title + " | stored sample images")
    fig.tight_layout()
    files += save_figure(fig, output, "01a_dataset_examples")
    clean_cache = run / "predictions" / "clean_0.npz"
    if clean_cache.exists():
        with np.load(clean_cache, allow_pickle=False) as cache:
            all_labels = cache["labels"]
        fig, ax = plt.subplots(figsize=(7, 3.4))
        ax.bar(np.arange(10), np.bincount(all_labels, minlength=10), color="#0072B2")
        ax.set(xlabel="Class index", ylabel="Evaluated examples", xticks=np.arange(10),
               title=title + " | evaluated test-subset class distribution")
        fig.tight_layout()
        files += save_figure(fig, output, "01b_class_distribution")
    from neuralshift.corruptions.registry import apply_corruption
    import torch
    names = ["gaussian_noise", "blur", "brightness", "contrast"]
    fig, axes = plt.subplots(4, 6, figsize=(10, 7))
    for row, name in enumerate(names):
        for severity in range(6):
            image = torch.as_tensor(images[0:1]).float()
            if severity:
                image = apply_corruption(image, name, severity, seed=0)
            array = image[0].numpy().transpose(1, 2, 0)
            axes[row, severity].imshow(array[..., 0] if array.shape[-1] == 1 else array,
                                       cmap="gray", vmin=0, vmax=1)
            axes[row, severity].set(xticks=[], yticks=[])
            if row == 0:
                axes[row, severity].set_title("Clean" if severity == 0 else f"Severity {severity}")
            if severity == 0:
                axes[row, severity].set_ylabel(name)
    fig.suptitle(title + " | one stored sample; illustrative corruption seed 0")
    fig.tight_layout()
    files += save_figure(fig, output, "03_corruption_severity")
    return files, []


def _training_figure(run, output, title):
    path = run / "history.csv"
    if not path.exists():
        return []
    history = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for label, color in [("train", "#0072B2"), ("val", "#D55E00")]:
        axes[0].plot(history.epoch, history[f"{label}_loss"], "o-", label=label, color=color)
        axes[1].plot(history.epoch, history[f"{label}_accuracy"], "o-", label=label, color=color)
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss", ylim=(0, None))
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    axes[0].legend(frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    return save_figure(fig, output, "02_training_convergence")


def _detection_figures(cache, output, title, suffix):
    probabilities, scores = _scores(cache)
    errors = (probabilities.argmax(1) != cache["labels"]).astype(int)
    files = []
    if len(np.unique(errors)) == 2:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        for method, score in scores.items():
            fpr, tpr, _ = roc_curve(errors, score)
            ax.plot(fpr, tpr, label=method, color=COLORS[method])
        ax.plot([0, 1], [0, 1], "--", color="0.5")
        ax.set(xlabel="False-positive rate (correct examples flagged)",
               ylabel="True-positive rate (errors flagged)", xlim=(0, 1), ylim=(0, 1), title=title)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        files += save_figure(fig, output, "07_error_detection_" + suffix)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for method, score in scores.items():
        order = np.argsort(score, kind="stable")
        coverage = np.arange(1, len(errors) + 1) / len(errors)
        risk = np.cumsum(errors[order]) / np.arange(1, len(errors) + 1)
        ax.plot(coverage, risk, label=method, color=COLORS[method])
    ax.set(xlabel="Coverage (accepted fraction)", ylabel="Selective risk (error rate)",
           xlim=(0, 1), ylim=(0, 1), title=title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    files += save_figure(fig, output, "08_selective_prediction_" + suffix)
    return files


def _method_figure(frame, output, title):
    raw = frame[frame.calibration == "raw"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, metric in zip(axes.flat, ["auroc", "auprc", "ece", "aurc"], strict=True):
        table = raw.pivot(index=["corruption", "severity"], columns="method", values=metric)
        for method in table.columns:
            ax.plot(np.arange(len(table)), table[method], "o-", markersize=3,
                    label=method, color=COLORS.get(method))
        ax.set(xticks=np.arange(len(table)), xticklabels=[f"{c}:{s}" for c, s in table.index],
               ylabel=metric.upper(), ylim=(0, 1))
        ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(title + " | fixed deterministic classifier for all scoring methods")
    fig.tight_layout()
    return save_figure(fig, output, "09_method_comparison")


def _statistical_figure(runs, output):
    frames = []
    for run in runs:
        frame = pd.read_csv(run / "metrics.csv")
        frames.append(frame[frame.calibration == "raw"].assign(run_id=run.name))
    merged = pd.concat(frames, ignore_index=True)
    outputs = []
    notes = []
    for (dataset, model), frame in merged.groupby(["dataset", "model"]):
        keys = ["seed", "corruption", "severity", "method"]
        if frame.duplicated(keys).any():
            notes.append(f"{dataset}/{model}: duplicate seed-condition-method records; pooled statistics omitted.")
            continue
        if frame.seed.nunique() < 2:
            notes.append(f"{dataset}/{model}: fewer than two independent training seeds; statistical intervals pending.")
            continue
        wide = frame.pivot(index=["seed", "corruption", "severity"], columns="method", values="auroc")
        rows = []
        for method in [c for c in wide.columns if c != "msp"]:
            pair = wide[["msp", method]].dropna()
            # Every seed must cover exactly the same condition set for an aggregate.
            sets = [set(zip(g.index.get_level_values("corruption"),
                            g.index.get_level_values("severity"))) for _, g in pair.groupby(level="seed")]
            if not sets or any(s != sets[0] for s in sets):
                notes.append(f"{dataset}/{model}/{method}: unmatched condition sets; aggregate omitted.")
                continue
            delta = (pair[method] - pair.msp).groupby(level="seed").mean().to_numpy()
            if len(delta) < 2:
                continue
            rng = np.random.default_rng(137)
            bootstrap = rng.choice(delta, size=(5000, len(delta)), replace=True).mean(1)
            lo, hi = np.quantile(bootstrap, [.025, .975])
            rows.append({"method": method, "difference": delta.mean(), "ci_lower": lo, "ci_upper": hi,
                         "n_seeds": len(delta), "sd": delta.std(ddof=1)})
        if not rows:
            continue
        comparisons = pd.DataFrame(rows)
        comparisons.to_csv(output / f"paired_{dataset}_{model}.csv", index=False)
        fig, ax = plt.subplots(figsize=(7, 4))
        for i, row in comparisons.iterrows():
            ax.plot([row.ci_lower, row.ci_upper], [i, i], color=COLORS[row.method], linewidth=2)
            ax.scatter(row.difference, i, color=COLORS[row.method])
        ax.axvline(0, color="0.4", linestyle="--")
        ax.set(yticks=np.arange(len(rows)), yticklabels=comparisons.method,
               xlabel="Mean AUROC difference from MSP (positive favors method)",
               title=f"{dataset} / {model} | paired training-seed bootstrap\n95% percentile intervals; exploratory")
        fig.tight_layout()
        outputs += save_figure(fig, output, f"10_paired_differences_{dataset}_{model}")
    return outputs, notes


def generate_figures(results_path) -> list[Path]:
    runs = discover_runs(results_path)
    configure_style()
    files, notes = [], []
    for run in runs:
        frame = pd.read_csv(run / "metrics.csv")
        if frame.empty:
            raise ValueError(f"Empty metrics: {run}")
        first = frame.iloc[0]
        title = f"{first.dataset} / {first.model} / training seed {first.seed}"
        output = run / "figures" / _stamp()
        output.mkdir(parents=True)
        extra, messages = _dataset_figures(run, output, title)
        files += extra
        notes += messages
        files += _training_figure(run, output, title)
        files += robustness_figures(frame, output, title)
        files += _method_figure(frame, output, title)
        temperature = json.loads((run / "temperature.json").read_text())["temperature"]
        caches = sorted((run / "predictions").glob("*.npz"))
        # Predeclared clean and greatest evaluated severity per family, not a best-result selection.
        selected = [p for p in caches if p.stem == "clean_0"]
        for family in sorted(frame.loc[frame.corruption != "clean", "corruption"].unique()):
            severity = int(frame.loc[frame.corruption == family, "severity"].max())
            path = run / "predictions" / f"{family}_{severity}.npz"
            if path.exists():
                selected.append(path)
        for cache_path in selected:
            with np.load(cache_path, allow_pickle=False) as cache:
                label = title + " | " + cache_path.stem
                files += reliability_diagram(cache, temperature, output, label, "06_calibration_" + cache_path.stem)
                files += _detection_figures(cache, output, label, cache_path.stem)
        (output / "figure_manifest.json").write_text(json.dumps({
            "source_run": str(run), "files": [p.name for p in files if p.parent == output],
            "notes": messages, "source": "validated stored predictions and metrics",
        }, indent=2), encoding="utf-8")
    root = Path(results_path).resolve()
    aggregate = root / "figures" / ("statistics-" + _stamp())
    aggregate.mkdir(parents=True)
    extra, messages = _statistical_figure(runs, aggregate)
    files += extra
    notes += messages
    (aggregate / "status.json").write_text(json.dumps({"notes": notes}, indent=2), encoding="utf-8")
    return files


def generate_report(results_path) -> Path:
    runs = discover_runs(results_path)
    protocol_groups = {}
    for run in runs:
        cfg = json.loads((run / 'config.json').read_text(encoding='utf-8'))
        key = (cfg['dataset'], cfg['model'])
        cfg.pop('seed', None)
        cfg.pop('output', None)
        signature = json.dumps(cfg, sort_keys=True)
        protocol_groups.setdefault(key, set()).add(signature)
    if any(len(signatures) > 1 for signatures in protocol_groups.values()):
        raise ValueError('Mixed experiment protocols: generate a separate report per profile')
    figures = generate_figures(results_path)
    frames = [pd.read_csv(r / "metrics.csv").assign(run_id=r.name) for r in runs]
    frame = pd.concat(frames, ignore_index=True)
    root = Path(results_path).resolve()
    report_dir = root / "reports" / _stamp()
    report_dir.mkdir(parents=True)
    table_path = report_dir / "validated_metrics.csv"
    frame.to_csv(table_path, index=False)
    from neuralshift.evaluation.statistics import paired_comparison, summarize_values
    summary_rows, paired_rows = [], []
    group_keys = ['dataset', 'model', 'corruption', 'severity', 'method', 'calibration']
    for key, group in frame.groupby(group_keys):
        if group.seed.duplicated().any():
            raise ValueError('Duplicate seed for a condition; report separate experiment profiles')
        for metric in ['accuracy', 'mean_confidence', 'nll', 'brier', 'ece', 'auroc', 'auprc', 'aurc']:
            summary_rows.append(dict(zip(group_keys, key)) | {'metric': metric} | summarize_values(group[metric]))
    for (dataset, model), group in frame[frame.calibration == 'raw'].groupby(['dataset', 'model']):
        for method in ['entropy', 'mc_entropy', 'mc_mi', 'mc_variance']:
            for metric in ['auroc', 'auprc', 'aurc']:
                result = paired_comparison(group, metric, method, 'msp', seed=137, bootstrap_samples=5000)
                paired_rows.append({'dataset': dataset, 'model': model, **result})
    pd.DataFrame(summary_rows).to_csv(report_dir / 'seed_summary.csv', index=False)
    pd.DataFrame(paired_rows).to_csv(report_dir / 'paired_comparisons.csv', index=False)
    clean = frame[(frame.corruption == "clean") & (frame.method == "msp")]
    summary_cols = ["run_id", "dataset", "model", "seed", "calibration", "n", "accuracy", "nll", "ece", "auroc", "aurc"]
    conditions = frame[["dataset", "model", "seed", "corruption", "severity"]].drop_duplicates()
    manifests = [json.loads((r / "manifest.json").read_text(encoding="utf-8")) for r in runs]
    evidence = "\n".join(f"- `{r.name}`: [manifest]({(r / 'manifest.json').as_posix()}), "
                          f"[metrics]({(r / 'metrics.csv').as_posix()})" for r in runs)
    observations = " ".join(
        f"{r['dataset']}/{r['model']}: {r['method_a']} minus MSP mean AUROC difference {r['mean_difference']:.4f} across {r['n_seeds']} training seeds."
        for r in paired_rows if r['metric'] == 'auroc' and r['mean_difference'] is not None
    )
    observations += " Negative differences favor MSP for this protocol; these are exploratory observations, not universal method rankings."
    text = f"""# {TITLE}

Empirical undergraduate research study. Automatically generated from validated execution artifacts.
Generated: {datetime.now(timezone.utc).isoformat()}. Full five-seed research matrix: **pending** unless independently verified in the execution ledger.

## 1. Abstract
NeuralShift Lab evaluates the ability of uncertainty scores to identify errors made by an image classifier under controlled image corruptions. This report includes {len(runs)} validated run directories and {len(conditions)} dataset/model/seed/condition combinations. These counts describe stored evidence, not completion of the research program. Small or single-seed runs establish pipeline execution and preliminary observations; they do not establish general superiority of any method.

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

{evidence}

## 7. Neural Network Architectures
The lightweight CNN uses convolutional feature blocks, ReLU, pooling, optional batch normalization, dropout, adaptive pooling, and a classification head. ResNet-18 uses a 3 by 3 stride-one input convolution, no initial max pooling, and dropout before the classifier. Both are initialized from scratch. Precise configurations and parameter counts belong to the recorded runs. Architecture comparisons require separately trained, matched-seed models.

## 8. Distribution Shift Methodology
Gaussian noise, Gaussian blur, brightness, and contrast operate on floating point pixels in [0, 1], before normalization. Five predefined severity levels represent a controlled subset of synthetic covariate shifts. Random noise is reproducible from recorded seeds and sample identifiers. Corruption is applied only during held-out evaluation. The exact equations, clipping rules, and mappings are documented in `docs/methodology.md` and implemented in the corruption registry. Image-label semantics may become ambiguous at large severity, a threat to validity.

## 9. Uncertainty Estimation Methods
Raw MSP and entropy, temperature-scaled MSP and entropy, and raw MC predictive entropy, mutual information, and probability variance are evaluated. MC dropout changes only dropout module behavior and keeps batch normalization in evaluation mode. The primary error target is the same deterministic classifier for every score, ensuring a matched comparison. MC mean predictions are not silently substituted for that classifier. Temperature is fitted only to validation logits; it is not optimized against final test labels.

## 10. Evaluation Metrics
Accuracy is the fraction correct. Macro F1 averages the classwise harmonic mean of precision and recall. NLL is mean -log p_y; the multiclass Brier score is mean sum_k (p_k - 1[y=k])^2. ECE is sum_b (n_b/n) abs(acc_b-conf_b) over 15 equal-width confidence bins. ECE depends on binning and finite-sample effects and is not a complete calibration measure. Error-detection positives are incorrect predictions, with larger uncertainty scores predicting errors. AUROC, average precision (reported as AUPRC), and observed-threshold FPR at target TPR are undefined when only one outcome class exists. Selective risk is the error rate among the least-uncertain accepted examples. AURC is the arithmetic mean of prefix risks at coverages 1/n through 1, with stable tie ordering.

## 11. Experimental Results
The table below is calculated from stored clean-condition metrics. The full condition table is [validated_metrics.csv](validated_metrics.csv). Repeated accuracy values across scoring methods reflect a shared deterministic prediction target, not independent experiments.

{table_markdown(clean[summary_cols])}

No result in this report is evidence for the unexecuted full research matrix. Figures below are generated from stored predictions and histories. Missing ROC plots indicate an undefined one-class error-detection task; they are not assigned a fabricated score.

## 12. Statistical Analysis
Independent training seed is the resampling unit for paired method comparisons. Condition-matched seed differences are averaged within seed before a percentile bootstrap over seeds (5,000 draws, fixed bootstrap seed 137). A statistical plot is omitted when fewer than two independent seeds are available or condition matching fails. Two seeds provide very limited uncertainty information; at least five seeds are the planned primary design. Executed exploratory paired t-test p-values and paired effect sizes are stored in paired_comparisons.csv; no confirmatory claim is made. Mean, standard deviation, sample count, and t intervals appear in seed_summary.csv. Single-seed intervals remain missing. All current comparisons are exploratory; a future confirmatory family should prespecify contrasts and use Holm correction. Repeated dropout passes and repeated corruptions are not independent training runs. The intervals describe seed variation conditional on this fixed test sample, not population sampling uncertainty.

## 13. Discussion
{observations}

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
{REFERENCES}

## 19. Appendix
### Stored evidence
Each table row is traceable through its run identifier to a manifest, checkpoint, condition cache, and metric computation. The report-generation process does not impute missing runs.

### Figures
"""
    for path in figures:
        if path.suffix == ".png":
            relative = Path(os.path.relpath(path, report_dir)).as_posix()
            text += f"\n![{path.stem}]({relative})\n"
    text += "\n### Run configuration appendix\n"
    for run, manifest in zip(runs, manifests, strict=True):
        text += f"\n#### {run.name}\n```json\n{json.dumps(manifest.get('config', {}), indent=2)}\n```\n"
    text += '\nAggregate tables: [seed summaries](seed_summary.csv) and [paired comparisons](paired_comparisons.csv).\n'
    path = report_dir / "manuscript.md"
    path.write_text(text, encoding="utf-8")
    (report_dir / "report_manifest.json").write_text(json.dumps({
        "runs": [str(r) for r in runs], "metrics": str(table_path),
        "figures": [str(p) for p in figures], "research_matrix_complete": False,
    }, indent=2), encoding="utf-8")
    return path
