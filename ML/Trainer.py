
from Database.ModelRepository import ModelRepository
from .Darts.Training.ensemble_training import EnsembleTrainer
from Database.Models.Model import Model


class Trainer:
    trained_models = []
    def __init__(self, models : list[Model], serviceId, series, forecast_period, split_train_val, repository:ModelRepository):
        self.models = models
        self.serviceId = serviceId
        self.series = series
        self.forecast_period = forecast_period
        self.split_train_val = split_train_val
        self.repository = repository

    def train_model(self):
        for model in self.models:
            try:
                # 1. Train model using Darts
                model.forecastingModel = model.forecastingModel.fit(self.series)
                print(f"{model.__class__.__name__} fitted")
                
                # 2. Insert trained model into db
                self.repository.insert_model(model)
                print(f"{model.__class__.__name__} inserted in db")

            except Exception as e:
                print(f"Error training {model.__class__.__name__}: {str(e)}")

    def train_ensemble(self, ensemble_candidates):
        trainer = EnsembleTrainer(ensemble_candidates, self.series, self.forecast_period, split_train_val=self.split_train_val)
        learned = trainer.create_learned_ensemble_model()
        naive = trainer.create_naive_ensemble_model()
        return (learned, naive)
