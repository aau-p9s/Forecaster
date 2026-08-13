from Database.Entities.Service import Service
from Database.Repository import Repository
from Database.dbhandler import DbConnection

class ServiceRepository(Repository[Service]):
    _class = Service
    def __init__(self, db: DbConnection):
        self.db = db

    def get_service_by_name(self, name:str) -> Service:
        rows = self.get_by("name", name)
        if len(rows) == 0:
            raise ValueError("No such service")
        return rows[0]

    def table_name(self) -> str:
        return super().table_name() + "s"
