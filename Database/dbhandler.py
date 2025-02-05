import psycopg2
import uuid

class DbConnection:
    def __init__(self, dbname, username, password, hostname, port):
        self.connection = psycopg2.connect(database=dbname, user=username, password=password, host=hostname, port=port)
        self.cursor = self.connection.cursor()
    
    def execute_query(self, query_string, params=None):
        if params == None:
            self.cursor.execute(query_string)
        else:
            self.cursor.execute(query_string, params)
        data = self.cursor.fetchall()
        self.connection.commit()
        return data
    
    # Following are the actual query statements
    def get_all(self):
        return self.execute_query("SELECT * from models;")
    
    def get_by_model_name(self, modelname):
        return self.execute_query(f"SELECT * from models WHERE Name = {modelname} ORDER BY TrainedTime ASC;")
    
    def insert(self, modelname, modelpath):
        with open(modelpath, "rb") as file:
            binary_data = file.read()
        query = 'INSERT INTO models ("Id", "Name", "ModelBin", "TrainedTime") VALUES (%s, %s, %s, now()) RETURNING *; '
        params = (str(uuid.uuid4()), modelname, psycopg2.Binary(binary_data))
        self.cursor.execute(query, params)
        return self.cursor.fetchone()
    
        

def gen_uuid():
    return str(uuid.uuid4())