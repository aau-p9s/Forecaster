from typing import Any
from darts import TimeSeries


class Forecast:
    def __init__(self, modelId, forecast: TimeSeries, error:Any=float('inf')):
        self.modelId = modelId
        self.forecast = forecast
        self.error = error

    def serialize(self) -> str:
        return self.forecast.to_json()
