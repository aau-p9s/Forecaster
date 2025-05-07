from time import time
import pickle
from Database.dbhandler import DbConnection
import psycopg2
from Database.Utils import gen_uuid
from Database.Models.Model import Model

class ModelRepository:
    def __init__(self, db: DbConnection):
        self.db = db

    def get_all_models_by_service(self, serviceId) -> list[Model]:
        rows = self.db.execute_get('SELECT id, name, bin from models WHERE "serviceid" = %s;', [serviceId])
        return [Model(row[0], pickle.loads(row[2]), serviceId) for row in rows]

    def get_by_modelname_and_service(self, modelname, serviceId) -> Model:
        rows = self.db.execute_get('SELECT id, name, bin FROM models WHERE "Name" = %s AND "ServiceId" = %s ORDER BY "trainedat" ASC LIMIT 1;', [modelname, serviceId])
        if len(rows) > 0:
            row = rows[0]
            modelObj = pickle.loads(row[2])
            return Model(row[0], row[1], modelObj)
        raise psycopg2.DatabaseError

    def get_all_models(self) -> list[Model]:
        return [row[0] for row in self.db.execute_get('SELECT id from models')]

    def insert_model(self, model:Model) -> Model:
        result = self.db.execute_get('INSERT INTO models ("id", "name", "bin", "trainedat", "serviceid") VALUES (%s, %s, %s, %s, %s) RETURNING id, name, bin', [gen_uuid(), type(model.model).__name__, model.get_binary(), model.trainedTime, model.serviceId])
        obj = pickle.loads(result[0][2])
        return Model(result[0][0], obj, result[0][1])
