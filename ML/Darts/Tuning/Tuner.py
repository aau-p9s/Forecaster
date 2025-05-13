from joblib import logger
import torch
import darts.models as models
from darts.models.forecasting.forecasting_model import ForecastingModel

from Database.ModelRepository import ModelRepository
from Database.Utils import gen_uuid
from .hyperparameters import HyperParameterConfig, encode_time, ENCODERS
from .hyperparameters import ENCODERS, encode_time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import optuna
from darts.metrics import rmse, mae, smape
import logging
import os
import sys
import inspect
from darts.datasets import AirPassengersDataset
import ML.Darts.Utils.preprocessing as preprocessing
from .handle_covariates import *
from darts import TimeSeries
from ..Utils.preprocessing import *
from Database.Models.Model import Model
import pdb
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel

EARLY_STOP_THRESHOLD = 10


class EarlyStoppingExceeded(optuna.exceptions.OptunaError):
    early_stop = EARLY_STOP_THRESHOLD
    early_stop_count = 0
    best_score = None


def early_stopping_opt(study, trial):
    if EarlyStoppingExceeded.best_score == None:
        EarlyStoppingExceeded.best_score = study.best_value

    if study.best_value < EarlyStoppingExceeded.best_score:
        EarlyStoppingExceeded.best_score = study.best_value
        EarlyStoppingExceeded.early_stop_count = 0
    else:
        if EarlyStoppingExceeded.early_stop_count > EarlyStoppingExceeded.early_stop:
            EarlyStoppingExceeded.early_stop_count = 0
            EarlyStoppingExceeded.best_score = None
            raise EarlyStoppingExceeded()
        else:
            EarlyStoppingExceeded.early_stop_count = (
                EarlyStoppingExceeded.early_stop_count + 1
            )
    return


class Tuner:
    def __init__(
        self,
        serviceId,
        model_repository:ModelRepository,
        data: TimeSeries,
        forecast_period,
        output = "output",
        train_val_split=0.75,
        gpu=0,
        trials=75,
        exclude_models=[],
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        torch.set_float32_matmul_precision("medium")
        self.gpu = gpu
        self.output = output if output is not None else "output"
        os.makedirs(output, exist_ok=True)
        self.trials = trials
        self.serviceId = serviceId
        print(f"Using {self.device} and {self.gpu}\n")
        self.exclude_models = exclude_models
        self.model_repository = model_repository
        # Optuna vars
        self.logger = logging.getLogger()
        os.makedirs("logs", exist_ok=True)
        log_filename = os.path.join("logs", f"tuner_{str(serviceId)}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log")
        optuna.logging.get_logger("optuna").addHandler(
            logging.StreamHandler(sys.stdout)
        )
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(
            logging.FileHandler(
                log_filename, mode="w"
            )
        )
        optuna.logging.enable_propagation()

        # Load dataset
        self.series = data
        self.forecast_period = forecast_period
        self.train_series, self.val_series = self.series.split_after(train_val_split)
        # self.models = [
        #     cls
        #     for name, cls in vars(models).items()
        #     if inspect.isclass(cls) and issubclass(cls, ForecastingModel)
        # ]
        self.models = model_repository.get_all_models_by_service(serviceId)
        self.past_covariates = None
        self.future_covariates = None
        self.best_score = float("inf")
        self.best_model: ForecastingModel = None

    def __tune_model(self, model: ForecastingModel | Model | str):
        if isinstance(model, str):  # Used for testing
            print("Is string")
            self.model = next(
                (m for m in self.models if str.lower(model) in str.lower(m.__name__)),
                None,
            )
            self.model_name = self.model.__class__.__name__
        elif isinstance(model, Model):
            print("Is Model")
            self.model = model.model
            self.model_name = model.model.__class__.__name__

        else:
            print("Is ForecastingModel")
            self.model = model
            self.model_name = model.__name__

        def objective(trial, model=self.model):
            try:
                # Get the model parameters
                model_class = model.model.__class__
                model_params = model.model._get_default_model_params()
                # model._model_params
                config = HyperParameterConfig(trial, model, series=self.train_series)
                uses_covariates = False

                # Match the parameters with suggestions from HyperParameterConfig
                params = {}
                for param_name, param_default in model_params.items():
                    try:
                        # Retrieve the parameter value from the config
                        if hasattr(config, param_name):
                            params[param_name] = getattr(config, param_name)
                        else:
                            params[param_name] = param_default
                        if param_name in [
                            "lags_future_covariates",
                            "lags_past_covariates",
                        ]:
                            # TODO: Implement handling of kwargs
                            if (
                                self.past_covariates is None
                                and self.future_covariates is None
                            ):
                                self.past_covariates = generate_past_covariates(
                                    self.series
                                )
                                self.future_covariates = generate_future_covariates(
                                    self.series
                                )
                        if param_name in ["add_encoders"]:
                            encoder_str = trial.suggest_categorical(
                                "add_encoders", list(ENCODERS.keys())
                            )
                            encoders = ENCODERS[encoder_str]
                            if encoder_str == "month":
                                if encoders is not None:
                                    encoders["custom"] = {"past": [encode_time]}
                            params["add_encoders"] = encoders
                        # If not present in config, use the default value
                    except AttributeError as ae:
                        # Attribute does not exist in config
                        print(ae)
                        params[param_name] = param_default
                    except Exception as e:
                        # Other exceptions, set the parameter to default
                        print(
                            f"Error occurred while setting '{param_name}': {str(e)}, using default value."
                        )
                        # params[param_name] = param_default

                if params.get("kwargs") is not None:
                    del params["kwargs"]
                if params.get("autoces_args") is not None:
                    if hasattr(config, "season_length"):
                        params["season_length"] = getattr(config, "season_length")
                    del params["autoces_args"]
                if params.get("autotheta_args") is not None:
                    if hasattr(config, "season_length"):
                        params["season_length"] = getattr(config, "season_length")
                    del params["autotheta_args"]
                if params.get("autoets_args") is not None:
                    if hasattr(config, "season_length"):
                        params["season_length"] = getattr(config, "season_length")
                    del params["autoets_args"]
                if params.get("autotheta_kwargs") is not None:
                    del params["autotheta_kwargs"]
                if params.get("prophet_kwargs") is not None:
                    del params["prophet_kwargs"]
                if params.get("autoces_kwargs") is not None:
                    del params["autoces_kwargs"]
                if params.get("autoTBATS_args") is not None:
                    del params["autoTBATS_args"]
                if params.get("autoTBATS_kwargs") is not None:
                    del params["autoTBATS_kwargs"]
                if params.get("autoarima_args") is not None:
                    del params["autoarima_args"]
                if params.get("autoets_kwargs") is not None:
                    del params["autoets_kwargs"]
                if params.get("autoarima_kwargs") is not None:
                    del params["autoarima_kwargs"]
                if params.get("fit_kwargs") is not None:
                    del params["fit_kwargs"]
                if all(
                    x is not None
                    for x in (
                        params.get("lags_future_covariates"),
                        params.get("lags_past_covariates"),
                        self.past_covariates,
                        self.future_covariates,
                    )
                ):
                    uses_covariates = True
                elif params.get("add_relative_index"):
                    uses_covariates = True
                else:
                    for param in [
                        "lags_future_covariates",
                        "lags_past_covariates",
                        "past_covariates",
                        "future_covariates",
                        "lags",
                    ]:
                        if param not in model_params and param in params:
                            # If the parameter does not exist in model_params, remove it from params
                            del params[param]
                        elif param in model_params and param in params:
                            if param == "lags_future_covariates":
                                params["lags_future_covariates"] = None
                            elif param == "lags_past_covariates":
                                params["lags_past_covariates"] = None
                            elif param == "past_covariates":
                                self.past_covariates = None
                            elif param == "future_covariates":
                                self.future_covariates = None

                if "pl_trainer_kwargs" in model_params.items() or isinstance(
                    self.model, TorchForecastingModel
                ):
                    params["pl_trainer_kwargs"] = {
                        "accelerator": "gpu",
                        "devices": self.gpu,  # change to self.gpu when implementing
                        "strategy": "auto",
                    }

                # Instantiate the model
                if model is not None:
                    model = model_class(**params)
                else:
                    raise Exception("Model not found")
                print(f"MODEL PARAMS: \n{model.model_params}\n")

                # Fit the model
                print("TRAINING")
                if uses_covariates:
                    model.fit(
                        self.train_series,
                        past_covariates=self.past_covariates,
                        future_covariates=self.future_covariates,
                    )
                else:
                    model.fit(self.train_series)
                print("PREDICTING")

                forecast = model.predict(self.forecast_period)
                val_target = self.val_series[
                    : len(forecast)
                ]  # This is used to ensure that the forecast is validated against the ground truth for the same timeframe
                rmse_value = rmse(val_target, forecast)
                mae_value = mae(val_target, forecast)
                smape_value = smape(val_target, forecast)

                trial.set_user_attr("MAE", str(mae_value))
                trial.set_user_attr("SMAPE", str(smape_value))

                if rmse_value < self.best_score:
                    self.best_score = rmse_value
                    self.best_model = model
                    plt.figure()
                    val_target.plot(label="Actual")
                    forecast.plot(label="Forecast")
                    plt.text(
                        0.1,
                        0.97,
                        f"MAE: {mae_value}\nSMAPE: {smape_value}",
                        transform=plt.gca().transAxes,
                        fontsize=10,
                        verticalalignment="top",
                        bbox=dict(facecolor="white", alpha=0.6),
                    )
                    plt.title(f"Best forecast for {self.model_name} RMSE: {rmse_value}")
                    plt.legend()
                    plt.grid(True)
                    os.makedirs(f"{self.output}/figures", exist_ok=True)
                    plt.savefig(
                        f"{self.output}/figures/{self.model_name}_{trial.number}"
                    )
                return rmse_value
            except EarlyStoppingExceeded:
                raise
            except Exception as e:
                self.logger.error(f"Trial failed: {e}")
                trial.set_user_attr("Error", str(e))
                print(f"Error: {e}")
                return float("inf")

        # Create Optuna study and optimize
        # study = optuna.create_study(direction="minimize", study_name=model_name + "_study", storage=self.db_url, load_if_exists=True, pruner=optuna.pruners.PatientPruner(wrapped_pruner=None, min_delta=0.05, patience=1))
        study = optuna.create_study(
            direction="minimize",
            storage="sqlite:///model-tuning.db",
            study_name=f"{self.serviceId}_{self.model_name}",
            load_if_exists=True,
        )
        try:
            study.optimize(
                objective, n_trials=self.trials, callbacks=[early_stopping_opt]
            )
            print(f"Study done: {study.best_trial}")
            if model is None:
                self.logger.warning("No successful model was trained.")
            return (study, model)
        except EarlyStoppingExceeded as e:
            self.logger.info("Early stopped")
            print(
                f"Study stopped due to rmse not improving for {EARLY_STOP_THRESHOLD} trials.\n"
            )
            study.set_user_attr("Early stopped", str(e))
            return (study, model)
        except Exception as e:
            self.logger.error(e)
            study.set_user_attr("Exception", str(e))
            return (study, model)

    def tune_all_models(self) -> list[tuple[optuna.Study, ForecastingModel]]:
        studies_and_models = []
        trained_model: ForecastingModel = None
        for model in self.models:
            self.best_model = None
            self.best_score = float("inf")
            print(
                f"Best model and score reset to {self.best_model} and {self.best_score}"
            )
            print(f"\nTuning {model} for service {self.serviceId}\n")
            try:
                if model.model.__class__.__name__ is not None and model.model.__class__.__name__ in self.exclude_models:
                    self.logger.info(f"EXCLUDING MODEL {model.__name__}")
                    print(f"EXCLUDING MODEL {model.__name__}")
                    continue
                else:
                    study, trained_model = self.__tune_model(model)
                print(f"\nDone with {model} for service {self.serviceId}\n")
                output = self.output
                if not os.path.exists(output):
                    os.makedirs(output)
                if self.model_name is not None:
                    model_folder = os.path.join(output, self.model_name)
                else:
                    model_folder = os.path.join(output, self.__class__.__name__)
                if not os.path.exists(model_folder):
                    os.makedirs(model_folder)
                print(f"Saving model {self.best_model} to {model_folder}")
                if self.best_model is not None:
                    self.best_model.save(
                        f"{model_folder}/{self.best_model.__class__.__name__}.pth"
                    )
                else:
                    self.logger.error("No model to save")
                failed_trials = [
                    t for t in study.trials if t.state == optuna.trial.TrialState.FAIL
                ]
                self.logger.info("Saving model params")
                best_trial_data = {
                    "number": study.best_trial.number,
                    "value": study.best_trial.value,
                    "params": study.best_trial.params,
                    "trials": len(study.trials),
                    "failed_trials": len(failed_trials),
                    "user_attrs": study.best_trial.user_attrs,
                }
                with open(f"{model_folder}/best_trial.json", "w") as f:
                    json.dump(best_trial_data, f, indent=4)
                print(f"Trial json saved to {model_folder} in best_trial.json")
                print(f"Model and best trial data saved to {model_folder}")
                studies_and_models.append((best_trial_data, trained_model))
                model.trainedTime = datetime.date.today()
                model.model = trained_model
                self.model_repository.insert_model(model)
            except Exception as err:
                print(f"\nError: {err=}, {type(err)=}\n")
        return studies_and_models

    def tune_model_x(self, model: Model):
        trained_model: ForecastingModel = None
        try:
            self.best_model = None
            self.best_score = float("inf")
            print(
                f"Best model and score reset to {self.best_model} and {self.best_score}"
            )
            print(f"\nTuning {model} for service {self.serviceId}\n")
            
            study, trained_model = self.__tune_model(model)
            output = self.output
            if not os.path.exists(output):
                os.makedirs(output)
            if self.model_name is not None:
                model_folder = os.path.join(output, self.model_name)
            else:
                model_folder = os.path.join(output, self.__class__.__name__)
            if not os.path.exists(model_folder):
                os.makedirs(model_folder)
            print(f"Saving model {self.best_model} to {model_folder}")
            if self.best_model is not None:
                self.best_model.save(
                    f"{model_folder}/{self.best_model.__class__.__name__}.pth"
                )
            else:
                self.logger.error("No model to save")
            failed_trials = [
                t for t in study.trials if t.state == optuna.trial.TrialState.FAIL
            ]

            best_trial_data = {
                "number": study.best_trial.number,
                "value": study.best_trial.value,
                "params": study.best_trial.params,
                "trials": len(study.trials),
                "failed_trials": len(failed_trials),
            }
            with open(f"{model_folder}/best_trial.json", "w") as f:
                json.dump(best_trial_data, f, indent=4)
            print(f"Trial json saved to {model_folder} in best_trial.json")
            if isinstance(model, Model):
                print(f"\nDone with: {model.name} for service {self.serviceId}\n")
            else:
                print(f"\nDone with {model} for service {self.serviceId}\n")
            model.model = trained_model
            return (best_trial_data, model)
        except Exception as err:
            print(f"\nError: {err}, {type(err)}\n")
