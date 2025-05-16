from uuid import UUID


class Service:
    def __init__(self, id:UUID, name:str, autoscaling_enabled:bool) -> None:
        self.id = id
        self.name = name
        self.autoscaling_enabled = autoscaling_enabled
