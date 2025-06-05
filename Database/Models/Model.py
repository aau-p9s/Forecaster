import datetime
import io
import tempfile
from uuid import UUID
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import MinMaxScaler

from darts.models.forecasting.forecasting_model import ForecastingModel

class Model:
    def __init__(self, modelId:UUID, modelName, model:ForecastingModel, serviceId:UUID, scaler=Scaler(MinMaxScaler(feature_range=(0, 1)))):
        self.modelId = modelId
        self.model:ForecastingModel = model
        self.name = modelName if modelName is not None else model.__class__.__name__
        self.trainedTime = datetime.date.today()
        self.serviceId = serviceId
        self.scaler = scaler

    def get_binary(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            self.model.save(f"{directory}/model.pth")
            with open(f"{directory}/model.pth", "rb") as file:
                return file.read()
        
