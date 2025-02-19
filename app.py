#!/usr/bin/env python3

from REST.api import start_api
from Database.dbhandler import DbConnection
from threading import Thread

def insert_model():
    # model_path = "Assets\\autotheta_model.pth"
    # res = model_repository.insert_model("AutoThetaTest2", model_path)
    # print(f"Inserted: {res}")
    pass

if __name__ == '__main__':
    db = DbConnection("p10s", "postgres", "password", "localhost", 5432)
    start_api()