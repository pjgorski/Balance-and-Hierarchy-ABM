import argparse


class Experiment:
    def __init__(self, args):
        self.args = args

    def __call__(self):
        raise NameError("Callable not implemented")

    @classmethod
    def name(cls) -> str:
        return cls.__name__

    def add_args(self, parser):
        pass

    def save_outputs(self):
        print("Ooops, looks like there is nothing to be saved")
        
    def clear(self):
        pass
