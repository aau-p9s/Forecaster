from Database.dbhandler import DbConnection
import psycopg2
from Database.Utils import gen_uuid

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
        params = (gen_uuid(), modelname, psycopg2.Binary(binary_data), trainedTime, serviceId)
        return self.db.execute_query(query, params)