import os

def getEnv(key:str, default:str) -> str:
    v = os.environ.get(key)
    if v is not None:
        return v
    else:
        return default
