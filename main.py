import argparse
import sys

from experiments import EXPERIMENTS


def main(args):
    experiment()
    print('Saving outputs...')
    experiment.save_outputs()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', help='Experiments')
    for experiment_key in EXPERIMENTS.keys():
        EXPERIMENTS[experiment_key].add_args(subparsers)
    args = parser.parse_args()
    experiment = EXPERIMENTS[args.command](args)
    sys.exit(main(args) or 0)
