#!/usr/bin/env python3

from REST.api import start_api
from Database.dbhandler import DbConnection, ModelRepository, ForecastRepository
from threading import Thread

def insert_model():
    model_path = "Assets\\autotheta_model.pth"
    res = model_repository.insert_model("AutoThetaTest2", model_path)
    print(f"Inserted: {res}")

if __name__ == '__main__':
    db = DbConnection("p10s", "postgres", "password", "localhost", 5432)
    model_repository = ModelRepository(db)
    forecast_repository = ForecastRepository(db)
    #start_api()
    insert_model()

