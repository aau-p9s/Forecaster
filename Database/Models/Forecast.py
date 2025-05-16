from typing import Any
from uuid import UUID
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler

class Forecast:
    def __init__(self, modelId:UUID, forecast: TimeSeries, error:Any=float('inf')):
        self.modelId = modelId
        self.forecast = forecast
        self.error = error

    def serialize(self) -> str:
        return self.forecast.to_json()
    
    def inverse_scale(self, scaler : Scaler):
        if scaler is not None:
            self.forecast = scaler.inverse_transform(self.forecast)
        else:
            raise ValueError("Scaler is None, cannot inverse scale the forecast.")
