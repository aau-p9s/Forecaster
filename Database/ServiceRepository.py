from psycopg2 import DatabaseError
from Database.Models.Service import Service
from Database.dbhandler import DbConnection

class ServiceRepository:
    def __init__(self, db: DbConnection):
        self.db = db

    def get_all_services(self) -> list[Service]:
        """Gets all services running in cluster."""
        return [Service(row[0], row[1], row[2]) for row in self.db.execute_query("SELECT * FROM services;")]

    def get_service_by_id(self, id) -> Service:
        """Gets service with specific id."""
        rows = self.db.execute_query('SELECT * FROM services WHERE "id" = %s;', [id])
        if len(rows) > 0:
            row = rows[0]
            return Service(row[0], row[1], row[2])
        raise DatabaseError

    def get_service_by_name(self, name) -> Service:
        """Gets cluster service by name."""
        result = self.db.execute_query('SELECT * FROM services WHERE "name" = %s;', [name])
        if len(result) > 0:
            row = result[0]
            return Service(row[0], row[1], row[2])
        raise DatabaseError
