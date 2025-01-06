import argparse
import sys
import os

def main(args):
    experiment()
    experiment.save_outputs()


if __name__ == "__main__":
    current = os.path.dirname(os.path.realpath(__file__))
    experiments = os.path.dirname(current)

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', help='Experiments')
    ltd_general.LtdGeneral.add_args(subparsers)
    args = parser.parse_args()
    experiment = ltd_general.LtdGeneral
    sys.exit(main(args) or 0)
