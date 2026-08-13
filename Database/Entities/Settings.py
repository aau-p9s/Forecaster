from uuid import UUID

import pandas as pd
from pandas import Timedelta

from Database.Entities.Entity import Entity



class Settings(Entity[str, int, int, int, int, int, int]):
    def __init__(self, service_id:UUID, scale_up:int, scale_down:int, min_replicas:int, max_replicas:int, scale_period:Timedelta, train_interval:Timedelta) -> None:
        self.service_id = service_id
        self.scale_up = scale_up
        self.scale_down = scale_down
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.scale_period = scale_period
        self.train_interval = train_interval
        super().__init__()

    @staticmethod
    def from_row(id, service_id, scale_up, scale_down, min_replicas, max_replicas, scale_period, train_interval):
        return Settings(UUID(service_id), scale_up, scale_down, min_replicas, max_replicas, pd.to_timedelta(f"{scale_period}ms"), pd.to_timedelta(f"{train_interval}ms")).with_id(UUID(id))

    def to_row(self):
        return str(self.id), str(self.service_id), self.scale_up, self.scale_down, self.min_replicas, self.max_replicas, int(self.scale_period.total_seconds()*1000), int(self.train_interval.total_seconds()*1000)
