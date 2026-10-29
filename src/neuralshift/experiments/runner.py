"""Checkpoint reuse, deterministic corruption caches, and traceable evaluation."""
import itertools
import logging
import time
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from neuralshift.config import validate_config
from neuralshift.datasets.loaders import make_loader, test_data
from neuralshift.datasets.transforms import normalize
from neuralshift.evaluation.calibration import calibration_metrics
from neuralshift.evaluation.classification import classification_metrics
from neuralshift.evaluation.detection import detection_metrics
from neuralshift.evaluation.selective_prediction import risk_coverage
from neuralshift.experiments.robustness import conditions, corrupt_batch
from neuralshift.experiments.uncertainty import scored_predictions
from neuralshift.reproducibility import seed_everything, select_device, timestamp
from neuralshift.training.checkpoints import load_checkpoint
from neuralshift.training.trainer import make_model, train
from neuralshift.uncertainty.mc_dropout import mc_predict
from neuralshift.utils.artifacts import atomic_csv, atomic_json, atomic_npz, read_json, sha256


def condition_metrics(config, name, severity, cache, temperature):
    rows, details = [], []
    labels = cache['labels']
    errors = (cache['logits'].argmax(1) != labels).astype(int)
    for method, calibration, probability, score in scored_predictions(cache['logits'], cache['mc_probabilities'], temperature):
        cls = classification_metrics(probability, labels)
        cal = calibration_metrics(probability, labels, bins=config['evaluation']['bins'])
        detection = detection_metrics(errors, score)
        selective = risk_coverage(errors, score)
        row = {'dataset': config['dataset'], 'model': config['model'], 'seed': config['seed'],
               'corruption': name, 'severity': severity, 'method': method, 'calibration': calibration,
               'n': len(labels), **{k: cls[k] for k in ('accuracy', 'macro_f1', 'nll', 'brier', 'mean_confidence')},
               'ece': cal['ece'], **{k: detection[k] for k in ('auroc', 'auprc', 'fpr_at_tpr')},
               'aurc': selective['aurc']}
        rows.append(row)
        details.append({**row, 'classification': cls, 'calibration_details': cal,
                        'detection': detection, 'selective': selective})
    return rows, details


def evaluate(config, checkpoint):
    checkpoint = Path(checkpoint).resolve()
    run = checkpoint.parent
    manifest = read_json(run / 'manifest.json')
    if manifest['config'] != config:
        raise ValueError('Evaluation config must match the finalized run configuration')
    if manifest['status'] in {'evaluation_complete', 'validated'}:
        from neuralshift.utils.validation import validate_run
        validate_run(run)
        return run
    if manifest['status'] not in {'training_complete', 'evaluating', 'evaluation_failed'}:
        raise ValueError('Finish training and validation calibration before test evaluation')
    if checkpoint.name != manifest['checkpoint']['path'] or sha256(checkpoint) != manifest['checkpoint']['sha256']:
        raise ValueError('Only the recorded validation-selected best checkpoint may be evaluated')
    seed_everything(config['seed'], config['training']['deterministic'], config['training']['threads'])
    device = select_device(config['training']['device'])
    model = make_model(config).to(device)
    model.load_state_dict(load_checkpoint(checkpoint, device)['model'])
    model.eval()
    dataset = test_data(config)
    loader = make_loader(dataset, config, evaluation=True)
    temperature = read_json(run / 'temperature.json')['temperature']
    manifest.update(status='evaluating', evaluation_started_at=timestamp())
    manifest.setdefault('cache_checksums', {})
    manifest['test'] = {'n': len(dataset), 'identifier_namespace': 'official_test', 'indices': dataset.indices.tolist()}
    atomic_json(run / 'manifest.json', manifest)
    images, labels, _ = next(iter(loader))
    atomic_npz(run / 'samples.npz', images=images[:12].numpy(), labels=labels[:12].numpy())
    rows, details = [], []
    started = time.perf_counter()
    try:
        for number, (name, severity) in enumerate(conditions(config)):
            relative = f'predictions/{name}_{severity}.npz'
            cache_path = run / relative
            if cache_path.exists():
                expected = manifest['cache_checksums'].get(relative)
                if expected is None or sha256(cache_path) != expected:
                    raise ValueError(f'Unverified or altered prediction cache: {relative}')
            else:
                torch.manual_seed(config['seed'] + 100000 + number)
                all_logits, all_mc, all_y, all_ids = [], [], [], []
                for images, labels, ids in loader:
                    corrupted = corrupt_batch(images, ids, name, severity, config['evaluation']['corruption_seed'])
                    x = normalize(corrupted, config['dataset']).to(device)
                    with torch.inference_mode():
                        all_logits.append(model(x).cpu().numpy())
                    all_mc.append(mc_predict(model, x, config['evaluation']['mc_passes'])['probabilities'].cpu().numpy())
                    all_y.append(labels.numpy())
                    all_ids.append(ids.numpy())
                atomic_npz(cache_path, logits=np.concatenate(all_logits), mc_probabilities=np.concatenate(all_mc, axis=1),
                           labels=np.concatenate(all_y), ids=np.concatenate(all_ids))
                manifest['cache_checksums'][relative] = sha256(cache_path)
                atomic_json(run / 'manifest.json', manifest)
            with np.load(cache_path, allow_pickle=False) as cache:
                condition_rows, condition_details = condition_metrics(config, name, severity, cache, temperature)
            rows.extend(condition_rows)
            details.extend(condition_details)
            logging.info('%s evaluated %s severity %d', run.name, name, severity)
        atomic_csv(run / 'metrics.csv', pd.DataFrame(rows))
        atomic_json(run / 'metrics_detailed.json', details)
        shift_rows = []
        with np.load(run / 'predictions/clean_0.npz') as clean:
            clean_scores = {(m, c): s for m, c, _, s in scored_predictions(clean['logits'], clean['mc_probabilities'], temperature)}
        for name, severity in conditions(config)[1:]:
            with np.load(run / f'predictions/{name}_{severity}.npz') as cache:
                for method, calibration, _, score in scored_predictions(cache['logits'], cache['mc_probabilities'], temperature):
                    baseline = clean_scores[(method, calibration)]
                    metric = detection_metrics(np.r_[np.zeros(len(baseline)), np.ones(len(score))], np.r_[baseline, score])
                    shift_rows.append({'task': 'clean_vs_corrupted', 'positive': 'corrupted', 'corruption': name,
                                       'severity': severity, 'method': method, 'calibration': calibration, **metric})
        atomic_json(run / 'shift_detection.json', shift_rows)
        core = ['best.pt', 'history.csv', 'config.json', 'dataset.json', 'validation_logits.npz', 'temperature.json',
                'samples.npz', 'metrics.csv', 'metrics_detailed.json', 'shift_detection.json']
        manifest['artifacts'] = {p: sha256(run / p) for p in core} | manifest['cache_checksums']
        manifest.update(status='evaluation_complete', evaluation_finished_at=timestamp(), evaluation_seconds=time.perf_counter() - started)
        atomic_json(run / 'manifest.json', manifest)
        from neuralshift.utils.validation import validate_run
        validate_run(run)
        return run
    except Exception as exc:
        manifest.update(status='evaluation_failed', error=str(exc))
        atomic_json(run / 'manifest.json', manifest)
        raise


def expand_matrix(config):
    matrix = config.get('matrix')
    if matrix is None:
        return [config]
    result = []
    for dataset, model, seed in itertools.product(matrix['datasets'], matrix['models'], matrix['seeds']):
        c = deepcopy(config)
        c.pop('matrix')
        c.update(dataset=dataset, model=model, seed=seed)
        validate_config(c)
        result.append(c)
    return result


def experiment(config):
    runs = []
    for item in expand_matrix(config):
        candidates = []
        for path in sorted((Path(item['output']) / 'runs').glob('*/manifest.json')):
            if read_json(path).get('config') == item:
                candidates.append(path.parent)
        if len(candidates) > 1:
            raise ValueError('Multiple identical runs found; select one using train --resume and evaluate explicitly')
        run = train(item, resume=candidates[0] if candidates else None)
        runs.append(evaluate(item, run / 'best.pt'))
    atomic_json(Path(config['output']) / 'experiment_manifest.json', {'runs': [str(r) for r in runs],
                'completed_at': timestamp(), 'configuration': config})
    return runs


def estimate(config):
    items = expand_matrix(config)
    train_examples = sum((c['data']['train_limit'] or (54000 if c['dataset'] == 'fashion_mnist' else 45000)) * c['training']['epochs'] for c in items)
    forward_examples = sum((c['data']['test_limit'] or 10000) * len(conditions(c)) * (1 + c['evaluation']['mc_passes']) for c in items)
    storage = sum((c['data']['test_limit'] or 10000) * len(conditions(c)) * 10 * (1 + c['evaluation']['mc_passes']) * 4 for c in items)
    return {'training_runs': len(items), 'training_example_epochs': train_examples,
            'evaluation_forward_examples': forward_examples, 'uncompressed_prediction_gib': storage / 1024**3,
            'wall_time': 'Hardware-dependent; time a smoke run first. Checkpoints, optimizer states and raw data require extra storage.'}
