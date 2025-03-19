
from Database.ModelRepository import ModelRepository
from ML.Model import Model



class Trainer:
    trained_models:list[Model] = []
    def __init__(self, models, serviceId, repository:ModelRepository):
        self.models = models
        self.serviceId = serviceId
        self.repository = repository

    def train_model(self):
        for model in self.models:
            # 1. Train model using Darts
            # 2. return trained model
            self.trained_models.append(Model(model, None, self.serviceId))

        for model in self.trained_models:
            self.repository.insert_model(model.name, model.binary, model.trainedTime, self.serviceId)
        
