"""One-factor plans and nested MC-pass ablations from existing cached samples."""
from copy import deepcopy
from pathlib import Path
from uuid import uuid4
import numpy as np
import pandas as pd
from neuralshift.config import validate_config
from neuralshift.utils.artifacts import atomic_csv, atomic_json, read_json, sha256


def ablation_configs(base, factor, values):
    """Training changes require new runs; severity/calibration use cached metrics."""
    allowed = {'dropout': ('model_options', 'dropout'), 'model': ('model',), 'seed': ('seed',)}
    if factor not in allowed:
        raise ValueError('Training factors: dropout, model, seed. Use cached results for severity/calibration/MC-pass ablations.')
    for value in values:
        config = deepcopy(base)
        config.pop('matrix', None)
        path = allowed[factor]
        if len(path) == 1:
            config[path[0]] = value
        else:
            config[path[0]][path[1]] = value
        config['ablation'] = {'factor': factor, 'value': value, 'control': 'All other resolved settings held fixed'}
        validate_config(config)
        yield config


def mc_sample_ablation(run_dir, passes):
    """Use nested prefixes of the same stochastic samples; no model retraining.

    Corruption severity and calibration are already paired within metrics.csv.
    Results are descriptive; repeated pass counts are not independent seeds.
    """
    from neuralshift.utils.validation import validate_run
    from neuralshift.experiments.robustness import conditions
    from neuralshift.experiments.runner import condition_metrics
    run = Path(run_dir)
    validate_run(run)
    config = read_json(run / 'config.json')
    temperature = read_json(run / 'temperature.json')['temperature']
    if not passes or any(not isinstance(n, int) or n < 2 or n > config['evaluation']['mc_passes'] for n in passes):
        raise ValueError('MC sample counts must be integers from 2 through the cached pass count')
    rows = []
    for count in sorted(set(passes)):
        for name, severity in conditions(config):
            with np.load(run / f'predictions/{name}_{severity}.npz') as data:
                cache = {key: data[key] for key in data.files}
            cache['mc_probabilities'] = cache['mc_probabilities'][:count]
            metrics, _ = condition_metrics(config, name, severity, cache, temperature)
            rows.extend({**row, 'mc_passes': count} for row in metrics if row['method'].startswith('mc_'))
    output = run / 'ablations' / uuid4().hex[:12]
    output.mkdir(parents=True)
    atomic_csv(output / 'mc_samples.csv', pd.DataFrame(rows))
    atomic_json(output / 'ablation_manifest.json', {'source_run': run.name, 'source_manifest_sha256': sha256(run/'manifest.json'),
                'factor': 'mc_passes', 'values': sorted(set(passes)), 'design': 'nested prefix samples; fixed checkpoint and examples',
                'independent_training_runs': 1, 'confirmatory_inference': False})
    return output
