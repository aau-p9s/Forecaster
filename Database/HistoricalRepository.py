
from Database.Models.Historical import Historical
from Database.dbhandler import DbConnection


class HistoricalRepository:
    def __init__(self, db: DbConnection):
        self.db = db

    def get_by_service(self, service_id:str) -> list[Historical]:
        rows = self.db.execute_get("SELECT id, serviceid, createdat, historicaldata FROM historicaldata WHERE serviceid = %s", [
            service_id
        ])
        return [Historical(row[0], row[1], row[2], row[3]) for row in rows]
