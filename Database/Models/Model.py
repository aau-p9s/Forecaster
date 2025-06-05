import datetime
import io
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
        file = io.BytesIO()
        writer = io.BufferedWriter(file)
        self.model.save(writer)
        writer.flush()
        file.seek(0)
        reader = io.BufferedReader(file)
        return reader.read()
        
