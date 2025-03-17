from ML.model import Model
class Trainer:
    trained_models = {}
    def __init__(self, models, serviceId):
        self.models = models
        self.serviceId = serviceId
        pass

    def train_model(self):
        for model in self.models:
            # 1. Train model using Darts
            # 2. return trained model
            self.trained_models.add(Model(model, None, self.serviceId))
        return self.trained_models
    
    def train_ensemble(self, ensemble_candidates):
        # 1. Train ensemble model using Darts EnsembleTraining module (in other repository)
        # 2. return trained model
        pass