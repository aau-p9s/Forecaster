
from Database.ModelRepository import ModelRepository
from .Darts.Training.ensemble_training import EnsembleTrainer
from Database.Models.Model import Model

class Trainer:
    trained_models:list[Model] = []
    def __init__(self, models, serviceId, series, forecast_period, split_train_val, repository:ModelRepository):
        self.models = models
        self.serviceId = serviceId
        self.series = series
        self.forecast_period = forecast_period
        self.split_train_val = split_train_val
        self.repository = repository

    def train_model(self):
        for model in self.models:
            # 1. Train model using Darts
            # 2. return trained model
            self.trained_models.append(Model("", model, self.serviceId))

        for model in self.trained_models:
            self.repository.insert_model(model)

    def train_ensemble(self, ensemble_candidates):
        trainer = EnsembleTrainer(ensemble_candidates, self.series, self.forecast_period, split_train_val=self.split_train_val)
        learned = trainer.create_learned_ensemble_model()
        naive = trainer.create_naive_ensemble_model()
        return (learned, naive)
