
from Database.ModelRepository import ModelRepository
from Database.ForecastRepository import Forecast, ForecastRepository
from .Darts.Training.ensemble_training import EnsembleTrainer
from Database.Models.Model import Model
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
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

    def __init__(self, models:list[Model], serviceId, data, forecast_period, split_train_val, repository:ModelRepository, forecast_repository:ForecastRepository):
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
                preprocessed_series = run_transformer_pipeline(self.series, model.scaler)
                self.train_series, self.val_series = preprocessed_series[0].split_after(self.split_train_val)

                # 1. Train model using Darts
                print(f"Training {model.__class__.__name__} for {self.serviceId}")
                model.model = model.model.fit(self.series)
                print(f"{model.__class__.__name__} fitted for {self.serviceId}")
                model.trainedTime = datetime.date.today()
                print(f"Predicting {model.__class__.__name__} for {self.serviceId}")
                forecast = model.predict(self.forecast_period)
                val_target = self.val_series[
                    : len(forecast)
                ]
                model.model.predict(self.forecast_period)
                print(f"{model.__class__.__name__} predicted for {self.serviceId}")
                rmse_value = rmse(val_target, forecast)
                print(f"RMSE for {model.__class__.__name__}: {rmse_value}")

                forecast = Forecast(model.modelId, forecast, rmse_value)
                self.forecast_repository.insert_forecast(forecast, self.serviceId)
                print("Forecast inserted in db")

                # 2. Insert trained model into db
                self.repository.insert_model(model)
                print(f"{model.__class__.__name__} inserted in db")

            except Exception as e:
                print(f"Error training {model.__class__.__name__}: {str(e)}")

    def train_ensemble(self, ensemble_candidates):
        self.series = load_json_data(self.data)
        preprocessed_series = run_transformer_pipeline(self.series)
        self.train_series, self.val_series = preprocessed_series[0].split_after(self.split_train_val)
        
        trainer = EnsembleTrainer(ensemble_candidates, self.train_series, self.val_series, self.forecast_period, split_train_val=self.split_train_val)
        print("Training learned ensemble model")
        learned = trainer.create_learned_ensemble_model()
        learned_id = uuid.uuid4()
        model = Model(learned_id, learned[2], self.serviceId, self.models[0].scaler)
        self.repository.insert_model(model)
        forecast = Forecast(learned_id, learned[1], learned[0])
        self.forecast_repository.insert_forecast(forecast, self.serviceId)
        print("Training naive ensemble model")
        naive = trainer.create_naive_ensemble_model()
        naive_id = uuid.uuid4()
        model = Model(naive_id, naive[2], self.serviceId, self.models[0].scaler)
        self.repository.insert_model(model)
        forecast = Forecast(naive_id, naive[1], naive[0])
        self.forecast_repository.insert_forecast(forecast, self.serviceId)
        return f"Ensemble models trained and inserted in db"