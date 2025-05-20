from typing import Any
from uuid import UUID


class Historical:
    def __init__(self, id:UUID, service_id:UUID, timestamp:float, data:dict[str, Any]) -> None:
        self.id = id
        self.service_id = service_id
        self.timestamp = timestamp
        self.data = data

