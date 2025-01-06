"""Registering experiments"""

EXPERIMENTS = {}


def register_experiment(cls):
    if cls.name in EXPERIMENTS:
        raise ValueError("Registering already existing experiment")
    EXPERIMENTS[cls.name()] = cls
    return cls
