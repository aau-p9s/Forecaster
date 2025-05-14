import datetime
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
        temporary_dir = tempfile.mkdtemp()
        
        self.model.save(f"{temporary_dir}/model.pth")
        
        with open(f"{temporary_dir}/model.pth", "rb") as file:
            return file.read()
