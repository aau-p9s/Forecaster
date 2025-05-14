
from Database.ModelRepository import ModelRepository
from Database.ForecastRepository import Forecast, ForecastRepository
from Database.Utils import gen_uuid
from .Darts.Training.ensemble_training import EnsembleTrainer
from Database.Models.Model import Model
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
from darts.models.forecasting.forecasting_model import ForecastingModel
import torch
import os
from ML.Darts.Utils.preprocessing import run_transformer_pipeline, load_data, load_json_data
from darts.metrics import rmse, mae, smape
import datetime
import uuid

class Trainer:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.set_float32_matmul_precision("medium")
    trained_models:list[Model] = []

    def __init__(self, models:list[Model], serviceId, data:str, forecast_period, repository:ModelRepository, forecast_repository:ForecastRepository, split_train_val=0.75):
        self.models = models
        self.serviceId = serviceId
        self.data = data
        self.forecast_period = forecast_period
        self.split_train_val = split_train_val
        self.repository = repository
        self.forecast_repository = forecast_repository
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_model(self):
        for model in self.models:
            try:
                # 1. Preprocess
                self.series = load_json_data(self.data)
                preprocessed_series, missing_value_ratio, scaler = run_transformer_pipeline(self.series)
                self.train_series, self.val_series = preprocessed_series.split_after(self.split_train_val)
                model.scaler = scaler

                # 1. Train model using Darts
                print(f"Training {model.modelId} for {self.serviceId}")
                try:
                    model.model = model.model.fit(self.train_series)
                except Exception as e:
                    print(f"Error fitting {model.modelId}: {str(e)}")
                    #raise e
                    continue
                print(f"{model.modelId} fitted for {self.serviceId}")
                model.trainedTime = datetime.date.today()
                print(f"Predicting {model.modelId} for {self.serviceId}")
                forecast = model.model.predict(self.forecast_period)
                val_target = self.val_series[
                    : len(forecast)
                ]
                print(f"{model.name} predicted for {self.serviceId}")
                rmse_value = rmse(val_target, forecast)
                print(f"RMSE for {model.name}: {rmse_value}")

                forecast = Forecast(model.modelId, forecast, rmse_value)
                #self.forecast_repository.insert_forecast(forecast, self.serviceId) #This is handled by the forecaster when requesting predict endpoint
                #print("Forecast inserted in db")

                # 2. Insert trained model into db
                self.repository.insert_model(model)
                print(f"{model.__class__.__name__} inserted in db")

            except Exception as e:
                print(f"Error training {model.modelId}: {str(e)}")
                #raise e

    def train_ensemble(self, ensemble_candidates:list[ForecastingModel]):
        self.series = load_json_data(self.data)
        preprocessed_series, missing_values_ratio, scaler = run_transformer_pipeline(self.series)
        self.train_series, self.val_series = preprocessed_series[0].split_after(self.split_train_val)
        
        trainer = EnsembleTrainer(ensemble_candidates, self.train_series, self.val_series, self.forecast_period, split_train_val=self.split_train_val)
        print("Training learned ensemble model")
        learned = trainer.create_learned_ensemble_model()
        
        candidate_names = "_".join(type(model).__name__ for model in ensemble_candidates)
        ensemble_name = f"Learned_Ensemble_{candidate_names}"

        model = Model(gen_uuid(), ensemble_name, learned[2], self.serviceId, scaler)
        self.repository.insert_model(model)
        forecast = Forecast(model.modelId, learned[1], learned[0])
        #self.forecast_repository.insert_forecast(forecast, self.serviceId) #This is handled by the forecaster when requesting predict endpoint

        print("Training naive ensemble model")
        naive = trainer.create_naive_ensemble_model()
        ensemble_name = f"Naive_Ensemble_{candidate_names}"
        model = Model(gen_uuid(), ensemble_name, naive[2], self.serviceId, scaler)
        self.repository.insert_model(model)
        forecast = Forecast(model.modelId, naive[1], naive[0])
        self.forecast_repository.insert_forecast(forecast, self.serviceId)
        return f"Ensemble models trained and inserted in db"
