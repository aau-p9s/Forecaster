from json import loads

from psycopg2 import DatabaseError
from Database.Models.Settings import Setting
from Database.dbhandler import DbConnection

class SettingsRepository:
    def __init__(self, db: DbConnection):
        self.db = db

    def get_settings(self, service_id) -> Setting:
        """Gets the current settings for the service."""
        rows = self.db.execute_get('SELECT * FROM settings WHERE "Id"=%s;', [service_id])
        if len(rows) > 0:
            row = rows[0]
            return Setting(row[0], row[1], int(row[2]), int(row[3]), int(row[4]), int(row[5]), int(row[6]), int(row[7]), loads(row[8]), loads(row[9]))
        raise DatabaseError

