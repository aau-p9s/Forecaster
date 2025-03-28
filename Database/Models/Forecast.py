from json import dumps, loads
from typing import Any
import datetime
from darts import TimeSeries


class Forecast:
    def __init__(self, modelId, forecast: TimeSeries, error:Any=float('inf')):
        self.modelId = modelId
        self.forecast = forecast
        self.error = error

    def serialize(self) -> str:
        data = loads(self.forecast.to_json())["data"]
        
        return dumps({
            "columns":["cpu"],
            "timestamp":[datetime.datetime.fromtimestamp(d[0]).strftime("%y-%m-%dT%H:%M:%S") for d in data],
            "value":[[d[1]] for d in data]
        })
