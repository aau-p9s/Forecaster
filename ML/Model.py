import datetime
import darts.models as models
import inspect
import torch
from darts.models.forecasting.forecasting_model import ForecastingModel
import io

class Model:
    def __init__(self, modelId, name, binary, serviceId):
        self.modelId = modelId
        self.name = name
        self.binary = self.load_model_from_blob(binary)
        self.trainedTime = datetime.date.today()
        self.serviceId = serviceId

    def load_model_from_blob(self, blob):
        """Converts a BLOB from the DB into a Darts model."""
        with io.BytesIO(blob) as buffer:
            buffered_reader = io.BufferedReader(buffer)
            # Load the Darts model using the Darts method
            return ForecastingModel.load(buffered_reader)