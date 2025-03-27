import torch
import darts.models as models
from darts.models.forecasting.forecasting_model import ForecastingModel
from .hyperparameters import HyperParameterConfig
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import optuna
from darts.metrics import rmse
import logging
import os
import sys
import inspect
from darts.datasets import AirPassengersDataset
import ML.Darts.Utils.preprocessing as preprocessing
from .handle_covariates import *
from darts import TimeSeries
from ..Utils.preprocessing import *

class Tuner:
    
    def __init__(self, serviceId, data: TimeSeries, forecast_period, train_val_split = 0.75, gpu = 0, trials = 75):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_workers = 8 if torch.cuda.is_available() else 0
        torch.set_float32_matmul_precision("medium")
        self.gpu = gpu
        self.trials = trials
        self.serviceId = serviceId
        
        # Optuna vars
        #self.db_url = os.getenv("OPTUNA_STORAGE_URL", "postgresql://optuna:password@optuna-db:5431/optuna")
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        
        #Load dataset
        #self.series = load_data(data, "min")
        self.series = data
        self.forecast_period = forecast_period
        self.train_series, self.val_series = self.series.split_after(train_val_split)
        self.models = [
            cls for name, cls in vars(models).items()
            if inspect.isclass(cls) and issubclass(cls, ForecastingModel)
        ]
        self.past_covariates = None
        self.future_covariates = None

    def __tune_model(self, modelName):
        model = next((m for m in self.models if str.lower(modelName) in str.lower(m.__name__)), None)
        model_name = model.__name__
        
        def objective(trial, model=model):
            
            # Get the model parameters
            model_class = model
            model_params = model._get_default_model_params()
            config = HyperParameterConfig(trial, model, series=self.train_series)
            uses_covariates = False
        
            # Match the parameters with suggestions from HyperParameterConfig
            params = {}
            for param_name, param_default in model_params.items():
                try:
                    # Retrieve the parameter value from the config
                    if hasattr(config, param_name): 
                        params[param_name] = getattr(config, param_name)
                    
                    if param_name in ["lags_future_covariates", "lags_past_covariates"]: #kwargs and covariates should be handled elsewhere
                        # TODO: Implement handling of kwargs

                        if self.past_covariates is None and self.future_covariates is None:
                            self.past_covariates = generate_past_covariates(self.series)
                            self.future_covariates = generate_future_covariates(self.series)
                    
                    # If not present in config, use the default value
                    if param_name == "kwargs":
                        params.pop["kwargs"]

                except AttributeError as ae:
                    # Attribute does not exist in config
                    print(ae)
                    params[param_name] = param_default
                except Exception as e:
                    # Other exceptions, set the parameter to default
                    print(f"Error occurred while setting '{param_name}': {str(e)}, using default value.")
                    #params[param_name] = param_default
                
            if all(x is not None for x in (params["lags_future_covariates"], params["lags_past_covariates"], self.past_covariates, self.future_covariates)):
                uses_covariates = True
            else:
                params["lags_future_covariates"] = None
                params["lags_past_covariates"] = None


            # Instantiate the model
            if model is not None:
                model = model_class(**params)
            else:
                raise Exception("Model not found")
            print(f"MODEL PARAMS: {model.model_params}")

            # Fit the model
            print("TRAINING")
            if (uses_covariates):
                model.fit(self.train_series, past_covariates=self.past_covariates, future_covariates=self.future_covariates)
            else:
                model.fit(self.train_series)
            print("PREDICTING")
            forecast = model.predict(self.forecast_period)
            
            rmse_value = rmse(self.val_series, forecast)
            return rmse_value
        
        # Create Optuna study and optimize
        try:
            #study = optuna.create_study(direction="minimize", study_name=model_name + "_study", storage=self.db_url, load_if_exists=True, pruner=optuna.pruners.PatientPruner(wrapped_pruner=None, min_delta=0.05, patience=1))
            study = optuna.create_study(direction="minimize", storage="sqlite:///model-tuning", study_name=f"{self.serviceId}_{model_name}", load_if_exists=True, pruner=optuna.pruners.PatientPruner(wrapped_pruner=None, min_delta=0.05, patience=1))
            study.optimize(objective, n_trials=self.trials, catch=(Exception, ))
            return study
        except Exception as err:
            print(f"\nSTUDY FAILED: {err=}, {type(err)=}\n")

    def tune_all_models(self):
        for model in self.models:
            print(f"\Tuning {model} for service {self.serviceId}\n")
            try:
                study = self.__tune_model(model())
                print(f"\nDone with {model} for service {self.serviceId}\n")
                return study
            except Exception as err:
                print(f"\nError: {err=}, {type(err)=}\n")
    def tune_model_x(self, modelName):
        try:
            print(f"\Tuning {modelName} for service {self.serviceId}\n")
            study = self.__tune_model(modelName)
            print(f"\nDone with: {modelName} for service {self.serviceId}\n")
            return study
        except Exception as err:
            print(f"\nError: {err=}, {type(err)=}\n")
