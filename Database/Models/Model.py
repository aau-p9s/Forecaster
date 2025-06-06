import io
import tempfile
from uuid import UUID
from darts.dataprocessing.transformers import Scaler
from pandas import Timedelta
from datetime import timedelta, date
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from darts.models.forecasting.forecasting_model import ForecastingModel

class Model:
    def __init__(self, modelId:UUID, modelName, model:ForecastingModel, serviceId:UUID, scaler=Scaler(MinMaxScaler(feature_range=(0, 1)))):
        self.modelId = modelId
        self.model:ForecastingModel = model
        self.name = modelName if modelName is not None else model.__class__.__name__
        self.trainedTime = date.today()
        self.serviceId = serviceId
        self.scaler = scaler

    def get_binary(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            self.model.save(f"{directory}/model.pth")
            with open(f"{directory}/model.pth", "rb") as file:
                return file.read()
    
    def get_trained_frequency(self, default: Timedelta) -> Timedelta:
        if self.model.training_series is not None:
            frequency = self.model.training_series.freq
            if isinstance(frequency, int):
                raise ValueError("fuck pandas")
            fixed_frequency = f"{frequency.n}{frequency.name}"
            return pd.to_timedelta(fixed_frequency)
        else:
            return default
