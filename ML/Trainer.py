from ML.model import Model
from .Darts.Training.ensemble_training import EnsembleTrainer
class Trainer:
    trained_models = {}
    def __init__(self, models, serviceId, series, forecast_period, split_train_val):
        self.models = models
        self.serviceId = serviceId
        self.series = series
        self.forecast_period = forecast_period
        self.split_train_val = split_train_val
        pass

    def train_model(self):
        for model in self.models:
            # 1. Train model using Darts
            # 2. return trained model
            self.trained_models.add(Model(model, None, self.serviceId))
        return self.trained_models
    
    def train_ensemble(self, ensemble_candidates):
        trainer = EnsembleTrainer(ensemble_candidates, self.series, self.forecast_period, split_train_val=self.split_train_val)
        learned = trainer.create_learned_ensemble_model()
        naive = trainer.create_naive_ensemble_model()
        return (learned, naive)