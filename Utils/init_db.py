#!/usr/bin/env python3

from darts.models import RandomForest
from darts.timeseries import TimeSeries
import sys
import csv
import pandas
import pickle
import tempfile

from Database.Models.Model import Model
from Database.dbhandler import DbConnection
from Database.ModelRepository import ModelRepository


db = DbConnection("autoscaler", "root", "password", sys.argv[-2], "5432")
repo = ModelRepository(db)

model = RandomForest(output_chunk_length=20, lags=[-5], lags_past_covariates=None)

ts = TimeSeries.from_csv("./test_data.csv")

model.fit(ts)

temporary_dir = tempfile.mkdtemp()

model.save(f"{temporary_dir}/model.pth")

with open(f"{temporary_dir}/model.pth", "rb") as file:
    obj = file.read()

repo.insert_model(Model(None, "RandomForest", obj, sys.argv[-1]))
