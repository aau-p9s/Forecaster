from uuid import UUID
from Database.Entities.Historical import Historical
from Database.Repository import Repository
from Database.dbhandler import DbConnection


class HistoricDataRepository(Repository[Historical]):
    _class = Historical
    def __init__(self, db: DbConnection):
        self.db = db

    def get_by_service(self, service_id:UUID) -> list[Historical]:
        rows = self.db.execute_get("SELECT * FROM historicdata WHERE serviceid = %s", [
            str(service_id)
        ])
        return [
            self._class.from_row(*row)
            for row in rows
        ]

    def delete_all(self):
        self.db.execute("DELETE FROM historicdata")

    def table_name(self) -> str:
        return "historicdata"
