from json import dumps, loads
from typing import Any
from darts import TimeSeries


class Forecast:
    def __init__(self, modelId, forecast: TimeSeries, error:Any=float('inf')):
        self.modelId = modelId
        self.forecast = forecast
        self.error = error

    def serialize(self) -> str:
        data = loads(self.forecast.to_json())["data"]
        return dumps([{"timestamp":ts, "cpu_percentage":ps} for ts,ps in data])
