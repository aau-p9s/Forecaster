from uuid import UUID

from Database.Entities.Entity import Entity



class Service(Entity[str, bool]):
    def __init__(self, name:str, autoscaling_enabled:bool) -> None:
        self.name = name
        self.autoscaling_enabled = autoscaling_enabled
        super().__init__()

    @staticmethod
    def from_row(id, name, autoscaling_enabled):
        return Service(name, autoscaling_enabled).with_id(UUID(id))

    def to_row(self):
        return str(self.id), self.name, self.autoscaling_enabled
