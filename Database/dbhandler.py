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

    def get_all_models(self):
        return self.db.execute_query("SELECT * from models;")
    
    def get_by_model_name(self, modelname):
        return self.db.execute_query('SELECT * FROM models WHERE "Name" = %s ORDER BY "TrainedTime" ASC LIMIT 1;', (modelname,))
    
    def insert_model(self, modelname, modelpath):
        with open(modelpath, "rb") as file:
            binary_data = file.read()
        query = 'INSERT INTO models ("Id", "Name", "ModelBin", "TrainedTime") VALUES (%s, %s, %s, now()) RETURNING *; '
        params = (str(uuid.uuid4()), modelname, psycopg2.Binary(binary_data))
        return self.db.execute_query(query, params)
    
class ForecastRepository:
    def __init__(self, db: DbConnection):
        self.db = db

    def insert_forecast(self, model_id, timestamp, value):
        """Inserts a forecast record linked to a model."""
        query = 'INSERT INTO forecasts ("Id", "ModelId", "Timestamp", "Value") VALUES (%s, %s, %s, %s) RETURNING *;'
        params = (str(uuid.uuid4()), model_id, timestamp, value)
        return self.db.execute_query(query, params, fetch_one=True)

    def get_forecasts_by_model(self, model_id):
        """Gets all forecasts for a given model."""
        query = 'SELECT * FROM forecasts WHERE "ModelId" = %s ORDER BY "Timestamp" ASC;'
        return self.db.execute_query(query, (model_id,))

    def get_latest_forecast(self, model_id):
        """Gets the latest forecast for a model."""
        query = 'SELECT * FROM forecasts WHERE "ModelId" = %s ORDER BY "Timestamp" DESC LIMIT 1;'
        return self.db.execute_query(query, (model_id,), fetch_one=True)
    
def gen_uuid():
    return str(uuid.uuid4())