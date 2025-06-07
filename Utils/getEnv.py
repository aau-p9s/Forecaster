import os

help_dict = {}

def getEnv(key:str, default:str) -> str:
    help_dict[key] = default
    v = os.environ.get(key)
    if v is not None:
        return v
    else:
        return default
