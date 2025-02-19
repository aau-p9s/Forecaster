from Database.dbhandler import DbConnection

class SettingsRepository:
    def __init__(self, db: DbConnection):
        self.db = db

    def get_settings(self, service_id):
        """Gets the current settings for the service."""
        return self.db.execute_query('SELECT * FROM settings WHERE "Id"=%s;', (service_id,))

    def set_settings(self, scaleup, scaledown, scaleperiod, id):
        """Update settings for service."""
        return self.db.execute_query('UPDATE settings SET "Scaleup"=%s, "Scaledown"=%s, "Scaleperiod"=%s WHERE id=%s;',
        (scaleup, scaledown, scaleperiod, id)
)