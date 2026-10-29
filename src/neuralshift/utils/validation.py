"""Fail-closed artifact validation including metric recomputation from predictions."""
from pathlib import Path
import numpy as np
import pandas as pd
from neuralshift.experiments.robustness import conditions
from neuralshift.utils.artifacts import atomic_json, read_json, sha256


def validate_run(run_dir):
    run = Path(run_dir)
    manifest = read_json(run / 'manifest.json')
    if manifest['status'] not in {'evaluation_complete', 'validated'}:
        raise ValueError(f'Incomplete run: {run}')
    for relative, digest in manifest['artifacts'].items():
        path = run / relative
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f'Missing or altered artifact: {path}')
    config = read_json(run / 'config.json')
    if config != manifest['config']:
        raise ValueError('Manifest and resolved config differ')
    dataset = read_json(run / 'dataset.json')
    train_ids, val_ids = dataset['train']['indices'], dataset['validation']['indices']
    if set(train_ids) & set(val_ids):
        raise ValueError('Training/validation leakage detected')
    calibration = read_json(run / 'temperature.json')
    if calibration['split'] != 'validation' or not 0 < calibration['temperature'] < float('inf'):
        raise ValueError('Invalid calibration provenance')
    if calibration['checkpoint_sha256'] != sha256(run / 'best.pt') or calibration['validation_logits_sha256'] != sha256(run / 'validation_logits.npz'):
        raise ValueError('Calibration source changed')
    with np.load(run / 'validation_logits.npz', allow_pickle=False) as v:
        if set(v['ids'].tolist()) != set(val_ids) or len(v['labels']) != calibration['n']:
            raise ValueError('Calibration did not use the recorded validation subset')
    from neuralshift.experiments.runner import condition_metrics
    expected_rows = []
    expected_ids, expected_labels = None, None
    for name, severity in conditions(config):
        with np.load(run / f'predictions/{name}_{severity}.npz', allow_pickle=False) as cache:
            logits, mc, labels, ids = cache['logits'], cache['mc_probabilities'], cache['labels'], cache['ids']
            if logits.shape != (len(labels), 10) or mc.shape != (config['evaluation']['mc_passes'], len(labels), 10):
                raise ValueError('Invalid prediction shape')
            if not np.isfinite(logits).all() or not np.isfinite(mc).all() or (mc < 0).any() or not np.allclose(mc.sum(-1), 1, atol=1e-5):
                raise ValueError('Non-finite logits or invalid stochastic probabilities')
            if not np.isin(labels, np.arange(10)).all() or len(set(ids.tolist())) != len(ids):
                raise ValueError('Invalid labels or duplicate sample IDs')
            if expected_ids is None:
                expected_ids, expected_labels = ids.copy(), labels.copy()
                if ids.tolist() != manifest['test']['indices']:
                    raise ValueError('Test IDs differ from recorded subset')
            elif not np.array_equal(expected_ids, ids) or not np.array_equal(expected_labels, labels):
                raise ValueError('Conditions must use identical examples with unchanged labels')
            rows, _ = condition_metrics(config, name, severity, cache, calibration['temperature'])
            expected_rows.extend(rows)
    actual, expected = pd.read_csv(run / 'metrics.csv'), pd.DataFrame(expected_rows)
    try:
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=False, rtol=1e-7, atol=1e-9)
    except AssertionError as exc:
        raise ValueError('Stored metrics do not match cached predictions') from exc
    result = {'valid': True, 'run_id': manifest['run_id'], 'conditions': len(conditions(config)),
              'metric_rows': len(actual), 'checks': ['sha256', 'split_disjointness', 'validation_calibration_provenance',
              'identical_test_ids_and_labels', 'finite_probabilities', 'metric_recomputation']}
    atomic_json(run / 'validation.json', result)
    manifest['status'] = 'validated'
    atomic_json(run / 'manifest.json', manifest)
    return result


def validate_results(path):
    root = Path(path)
    runs = [root] if (root / 'manifest.json').exists() else sorted(p.parent for p in root.rglob('manifest.json'))
    if not runs:
        raise ValueError(f'No run manifests found in {root}')
    return [validate_run(run) for run in runs]
