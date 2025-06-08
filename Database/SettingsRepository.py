from uuid import UUID

from Database.Entities.Settings import Settings
from Database.Repository import Repository
from Database.dbhandler import DbConnection

class SettingsRepository(Repository[Settings]):
    _class = Settings
    def __init__(self, db: DbConnection):
        self.db = db

    def get_settings(self, service_id:UUID) -> Settings:
        """Gets the current settings for the service."""
        rows = self.get_by("serviceid", str(service_id))
        if len(rows) == 0:
            raise ValueError("Error, no settings for service")
        return rows[0]

