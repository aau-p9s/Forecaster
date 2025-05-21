import datetime
import tempfile
from uuid import UUID
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import MinMaxScaler
import cloudpickle as pickle
import os

from darts.models.forecasting.forecasting_model import ForecastingModel

class Model:
    def __init__(self, modelId:UUID, modelName, model:ForecastingModel, serviceId:UUID, trainedTime:str, scaler=Scaler(MinMaxScaler(feature_range=(0, 1)))):
        self.modelId = modelId
        self.model:ForecastingModel = model
        self.name = modelName if modelName is not None else model.__class__.__name__
        self.trainedTime = trainedTime
        self.serviceId = serviceId
        self.scaler = scaler

    def get_binary(self):
        temporary_dir = tempfile.mkdtemp()
        path = os.path.join(temporary_dir, "model.pkl")

        with open(path, "wb") as f:
            pickle.dump(self, f)

        with open(path, "rb") as f:
            return f.read()
