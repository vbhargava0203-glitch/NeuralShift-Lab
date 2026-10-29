"""Execute prespecified training-factor ablations; post-hoc MC uses the ablate CLI."""
import argparse
from neuralshift.config import load_config
from neuralshift.experiments.ablations import ablation_configs
from neuralshift.experiments.runner import experiment, estimate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--estimate-only', action='store_true')
    args = parser.parse_args()
    base = load_config(args.config)
    plan = base.pop('ablation_plan')
    for config in ablation_configs(base, plan['factor'], plan['values']):
        print(estimate(config) if args.estimate_only else experiment(config))
