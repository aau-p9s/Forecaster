from datetime import datetime
from typing import Any
from uuid import UUID

from Database.Entities.Entity import Entity



class Historical(Entity[str, datetime, dict[str, Any]]):
    def __init__(self, service_id:UUID, timestamp:datetime, data:dict[str, Any]) -> None:
        self.service_id = service_id
        self.timestamp = timestamp
        self.data = data
        super().__init__()
        

    @staticmethod
    def from_row(id, service_id, timestamp, data):
        return Historical(UUID(service_id), timestamp, data).with_id(UUID(id))

    def to_row(self):
        return str(self.id), str(self.service_id), self.timestamp, self.data
