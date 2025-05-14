from typing import Any


class Historical:
    def __init__(self, id:str, service_id:str, timestamp:float, data:dict[str, Any]) -> None:
        self.id = id
        self.service_id = service_id
        self.timestamp = timestamp
        self.data = data

