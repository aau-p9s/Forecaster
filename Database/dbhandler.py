import psycopg2
import uuid

class DbConnection:
    def __init__(self, dbname, username, password, hostname, port):
        self.connection = psycopg2.connect(database=dbname, user=username, password=password, host=hostname, port=port)
        self.cursor = self.connection.cursor()
    
    def execute_query(self, query_string, params=None):
        """Executes query given a querystring and optional parameters.
        Args:
            query_string: Sql query statement
            params: Query parameters
        """
        if params == None:
            self.cursor.execute(query_string)
        else:
            self.cursor.execute(query_string, params)
        data = self.cursor.fetchall()
        self.connection.commit()
        return data
    
    def close(self):
        """Closes the database connection."""
        self.connection.close()

class ModelRepository:
    def __init__(self, db: DbConnection):
        self.db = db

    def get_all_models_by_service(self, serviceId):
        return self.db.execute_query('SELECT * from models WHERE "ServiceId" = %s;', (serviceId,))
    
    def get_by_modelname_and_service(self, modelname, serviceId):
        return self.db.execute_query('SELECT * FROM models WHERE "Name" = %s AND "ServiceId" = %s ORDER BY "TrainedTime" ASC LIMIT 1;', (modelname, serviceId))
    
    def insert_model(self, modelname, modelpath, trainedTime, serviceId):
        with open(modelpath, "rb") as file:
            binary_data = file.read()
        query = 'INSERT INTO models ("Id", "Name", "ModelBin", "TrainedTime", "ServiceId") VALUES (%s, %s, %s, %s, %s) RETURNING *; '
        params = (str(uuid.uuid4()), modelname, psycopg2.Binary(binary_data), trainedTime, serviceId)
        return self.db.execute_query(query, params)
    
class ForecastRepository:
    def __init__(self, db: DbConnection):
        self.db = db

    def insert_forecast(self, model_id, forecast, serviceId):
        """Inserts a forecast linked to a model and a service."""
        query = 'INSERT INTO forecasts ("Id", "ModelId", "Forecast", "ServiceId") VALUES (%s, %s, %s, %s) RETURNING *;'
        params = (str(uuid.uuid4()), model_id, forecast, serviceId)
        return self.db.execute_query(query, params, fetch_one=True)

    def get_forecasts_by_model_and_service(self, model_id, serviceId):
        """Gets all forecasts for a given model and service."""
        query = 'SELECT * FROM forecasts WHERE "ModelId" = %s AND "ServiceId" = %s ORDER BY "Timestamp" ASC;'
        return self.db.execute_query(query, (model_id, serviceId))
    
    def get_forecasts_by_service(self, serviceId):
        """Gets all forecasts for a given model and service."""
        query = 'SELECT * FROM forecasts WHERE "ServiceId" = %s ORDER BY "Timestamp" ASC;'
        return self.db.execute_query(query, (serviceId,))

    def get_latest_forecast(self, model_id, serviceId):
        """Gets the latest forecast for a model and service."""
        query = 'SELECT * FROM forecasts WHERE "ModelId" = %s AND "ServiceId" = %s ORDER BY "Timestamp" DESC LIMIT 1;'
        return self.db.execute_query(query, (model_id, serviceId), fetch_one=True)
    
    def get_latest_forecast_by_service(self, serviceId):
        """Gets the latest forecast for a service."""
        query = 'SELECT * FROM forecasts WHERE "ServiceId" = %s ORDER BY "Timestamp" DESC LIMIT 1;'
        return self.db.execute_query(query, (serviceId,), fetch_one=True)
    
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

class ServiceRepository:
    def __init__(self, db: DbConnection):
        self.db = db
    
    def get_all_services(self):
        """Gets all services running in cluster."""
        return self.db.execute_query("SELECT * FROM services;")

    def get_service_by_id(self, id):
        """Gets service with specific id."""
        return self.db.execute_query('SELECT * FROM services WHERE "Id" = %s;', (id,))
    
    def get_service_by_name(self, name):
        """Gets cluster service by name."""
        return self.db.execute_query('SELECT * FROM services WHERE "Name" = %s;', (name,))
    
def gen_uuid():
    return str(uuid.uuid4())