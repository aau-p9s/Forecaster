#!/usr/bin/env python
# coding: utf-8

import torch
import darts.models as models
from darts.models.forecasting.forecasting_model import ForecastingModel
from hyperparameters import HyperParameterConfig  # Import config from the external file
from darts.utils.likelihood_models import GaussianLikelihood
from darts.utils.missing_values import fill_missing_values
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from darts import TimeSeries
import optuna
import json
import numpy as np
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.metrics import rmse
import matplotlib.pyplot as plt
import logging
import sys
import inspect
from darts.datasets import AirPassengersDataset

# Load all the model classes
models = [
    cls for name, cls in vars(models).items()
    if inspect.isclass(cls) and issubclass(cls, ForecastingModel)
]

DEVICE_NAME = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    num_workers = 8
else:
    num_workers = 0
torch.set_float32_matmul_precision("medium")

series = AirPassengersDataset().load()
train_series, val_series = series.split_after(0.75)
val_display_series = series[-2880*2:] # Plot four days (weekend + start week)
val_series = val_display_series[-1440*2:]

# Optuna vars
storage_name = "sqlite:///{}.db".format("model-tuning")
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

def tune_model(model, trials):
    rmse_value = 0
    global best_score
    best_score = float('inf')

    model_name = (
        model.model.split('(', 1)[0] if model.model is not None
        else f"{model._module_name[1:]}_{model.rnn_type_or_module}" if model.rnn_type_or_module is not None
        else model._module_name[1:]
    )
    
    def objective(trial):
        config = HyperParameterConfig(trial)
        global rmse_value
        global best_score
        
        # Get the model parameters
        model_class = model()
        model_params = model_class.model_params
    
        # Match the parameters with suggestions from HyperParameterConfig
        params = {}
        for param_name, param_default in model_params.items():
            if hasattr(config, param_name):
                params[param_name] = getattr(config, param_name)
            else:
                params[param_name] = param_default  # Use default if not in HyperParameterConfig
    
        # Instantiate the model
        model = model(**params)
        print(f"MODEL PARAMS: {model.model_params}")

        # Fit the model
        print("TRAINING")
        model.fit(train_series)
        print("PREDICTING")
        forecast = model.predict(len(val_display_series)) #Forecast 4 days
        forecast_display_series = forecast[-1440*2:] # Use the last two days of forecast to calculate rmse
        
        rmse_value = rmse(val_series, forecast_display_series)

        if rmse_value < best_score:
            best_score = rmse_value
        return rmse_value
    
    # Create Optuna study and optimize
    try:
        study = optuna.create_study(direction="minimize", study_name=model_name + "_study", storage=storage_name, load_if_exists=True)
        study.optimize(objective, n_trials=trials)
    except Exception as err:
        print(f"\nSTUDY FAILED: {err=}, {type(err)=}\n")
        

if __name__ == '__main__':
    
    for model in models:
        print(f"\Tuning {model}\n")
        try:
            tune_model(model(), 75)
            print(f"\nDone with: {model}\n")
        except Exception as err:
            print(f"\nError: {err=}, {type(err)=}\n")

