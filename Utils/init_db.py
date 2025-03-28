#!/usr/bin/env python3

from darts.models import RandomForest
from darts.timeseries import TimeSeries
import sys

from Database.Models.Model import Model
from Database.ServiceRepository import ServiceRepository
from Database.dbhandler import DbConnection
from Database.ModelRepository import ModelRepository


db = DbConnection("autoscaler", "root", "password", sys.argv[-2], "5432")
repo = ModelRepository(db)
service_repo = ServiceRepository(db)


models = repo.get_all_models()

for model in models:
    repo.delete_model(model)

for service in service_repo.get_all_services():
    if not service.id == sys.argv[-1]:
        service_repo.delete_service(service)

model = RandomForest(output_chunk_length=20, lags=[-5], lags_past_covariates=None)

ts = TimeSeries.from_csv("./Assets/test_data.csv")

model.fit(ts)
repo.insert_model(Model("", model, sys.argv[-1]))

