import pkgutil
import importlib

from .decorator import generator

for loader, name, is_pkg in pkgutil.walk_packages(__path__, __name__+'.'):
    importlib.import_module(name)
