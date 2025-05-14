from typing import Any
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
import json

class Forecast:
    def __init__(self, modelId, forecast: TimeSeries, error:Any=float('inf')):
        self.modelId = modelId
        self.forecast = forecast
        self.error = error

    def serialize(self) -> str:
        df = self.forecast.pd_dataframe()

        return json.dumps({
            "columns": df.columns.tolist(),
            "timestamp": [ts.isoformat(timespec='milliseconds') for ts in df.index.to_pydatetime()],
            "value": df.values.tolist()
        })
    
    def inverse_scale(self, scaler : Scaler):
        if scaler is not None:
            self.forecast = scaler.inverse_transform(self.forecast)
        else:
            raise ValueError("Scaler is None, cannot inverse scale the forecast.")