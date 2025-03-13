#!/usr/bin/env python3

from Api.api import start_api
from Database.dbhandler import DbConnection
import os

def insert_model():
    # model_path = "Assets\\autotheta_model.pth"
    # res = model_repository.insert_model("AutoThetaTest2", model_path)
    # print(f"Inserted: {res}")
    pass

def getEnv(key:str, default:str):
    return os.environ.get(key) if key in os.environ else default

if __name__ == '__main__':
    database = getEnv("FORECASTER__PGSQL__DATABASE", "autoscaler")
    user = getEnv("FORECASTER__PGSQL__USER", "root")
    password = getEnv("FORECASTER__PGSQL__PASSWORD", "password")
    addr = getEnv("FORECASTER__PGSQL__ADDR", "0.0.0.0")
    port = getEnv("FORECASTER__PGSQL__PORT", "5432")
    api_addr = getEnv("FORECASTER__ADDR", "0.0.0.0")
    api_port = getEnv("FORECASTER__PORT", "8081")
    db = DbConnection(database, user, password, addr, port)
    start_api(db, api_addr, int(api_port))
