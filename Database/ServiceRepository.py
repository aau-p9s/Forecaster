from Database.dbhandler import DbConnection

class ServiceRepository:
    def __init__(self, db: DbConnection):
        self.db = db

    def get_all_services(self):
        """Gets all services running in cluster."""
        return self.db.execute_query("SELECT * FROM services;")

    def get_service_by_id(self, id):
        """Gets service with specific id."""
        return self.db.execute_query('SELECT * FROM services WHERE "id" = %s;', (id,))

    def get_service_by_name(self, name):
        """Gets cluster service by name."""
        return self.db.execute_query('SELECT * FROM services WHERE "name" = %s;', (name,))
