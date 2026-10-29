"""Research command-line interface."""
import argparse
import json
from neuralshift.config import load_config
from neuralshift.utils.logging import configure_logging


def parser():
    root = argparse.ArgumentParser(prog='neuralshift', description='Reproducible reliability experiments under controlled distribution shifts')
    commands = root.add_subparsers(dest='command', required=True)
    for name in ('download-data', 'train', 'evaluate', 'experiment', 'estimate'):
        sub = commands.add_parser(name, help=f'{name} using a resolved YAML protocol')
        sub.add_argument('--config', required=True)
        if name == 'train':
            sub.add_argument('--resume', help='Resume a run directory with identical configuration')
        if name == 'evaluate':
            sub.add_argument('--checkpoint', required=True, help='Recorded validation-selected best.pt')
    for name in ('figures', 'report', 'validate-results'):
        sub = commands.add_parser(name, help=f'{name} from stored run artifacts')
        sub.add_argument('--results', required=True, help='One run directory or a results root')
    sub = commands.add_parser('ablate', help='Nested MC pass-count analysis using verified caches')
    sub.add_argument('--results', required=True, help='One completed run directory')
    sub.add_argument('--mc-passes', nargs='+', type=int, required=True)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    configure_logging()
    try:
        config = load_config(args.config) if hasattr(args, 'config') else None
        if args.command == 'download-data':
            from neuralshift.datasets.loaders import load_source
            config['data']['download'] = True
            result = {'train_n': len(load_source(config, True)), 'test_n': len(load_source(config, False)), 'dataset': config['dataset']}
        elif args.command == 'train':
            from neuralshift.training.trainer import train
            result = train(config, args.resume)
        elif args.command in {'evaluate', 'experiment', 'estimate'}:
            from neuralshift.experiments.runner import estimate, evaluate, experiment
            result = evaluate(config, args.checkpoint) if args.command == 'evaluate' else experiment(config) if args.command == 'experiment' else estimate(config)
        elif args.command == 'ablate':
            from neuralshift.experiments.ablations import mc_sample_ablation
            result = mc_sample_ablation(args.results, args.mc_passes)
        elif args.command == 'validate-results':
            from neuralshift.utils.validation import validate_results
            result = validate_results(args.results)
        else:
            from neuralshift.visualization.report_figures import generate_figures, generate_report
            result = generate_figures(args.results) if args.command == 'figures' else generate_report(args.results)
        print(json.dumps(result, indent=2, default=str))
        return 0
    except (ValueError, FileNotFoundError, RuntimeError, KeyError) as exc:
        parser().exit(2, f'neuralshift: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
