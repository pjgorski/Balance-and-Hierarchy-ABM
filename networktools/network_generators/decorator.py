"""Registering network generators"""

generator = {}


def register_generator(cls):
    if cls.name in generator:
        raise ValueError("Registering already existing network generator")
    generator[cls.name()] = cls
    return cls
