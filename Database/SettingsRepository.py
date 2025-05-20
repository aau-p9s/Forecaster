from json import loads
from uuid import UUID

from psycopg2 import DatabaseError
from Database.Models.Settings import Setting
from Database.dbhandler import DbConnection

class SettingsRepository:
    def __init__(self, db: DbConnection):
        self.db = db

    def get_settings(self, service_id:UUID) -> Setting:
        """Gets the current settings for the service."""
        rows = self.db.execute_get('SELECT id, serviceid, scaleup, scaledown, minreplicas, maxreplicas, scaleperiod, traininterval FROM settings WHERE serviceid = %s;', [str(service_id)])
        print(rows)
        if len(rows) > 0:
            row = rows[0]
            return Setting(UUID(row[0]), UUID(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5]), int(row[6]), int(row[7]))
        raise DatabaseError(f"Error, settings table returned {len(rows)} rows: {rows} with service_id: {service_id}")

