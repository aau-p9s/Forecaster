from datetime import datetime
from json import dumps
from typing import Any
from uuid import UUID
from darts import TimeSeries

from Database.Entities.Entity import Entity

class Forecast(Entity[str, str, datetime, str, bool]):
    def __init__(self, service_id: UUID, created_at: datetime, model_id: UUID, forecast: TimeSeries, has_manual_change: bool, error:float=float('inf')):
        self.forecast = forecast
        self.error = error
        self.model_id = model_id
        self.service_id = service_id
        self.created_at = created_at
        self.has_manual_change = has_manual_change
        super().__init__()

    def serialize(self) -> str:
        return self.forecast.to_json()
    
    @staticmethod
    def from_row(id, service_id, created_at, model_id, forecast, has_manual_change):
        return Forecast(UUID(service_id), created_at, UUID(model_id), TimeSeries.from_json(dumps(forecast)), has_manual_change).with_id(UUID(id))

    def to_row(self):
        return str(self.id), str(self.model_id), str(self.service_id), self.created_at, self.serialize(), self.has_manual_change
