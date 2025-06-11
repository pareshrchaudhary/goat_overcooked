import importlib

def load(name):
    mod = importlib.import_module("mapbt.envs.mpe.scenarios." + name)
    return mod
