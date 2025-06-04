from datetime import date, datetime
from time import time
from uuid import UUID
import cloudpickle as pickle
from Database.dbhandler import DbConnection
import psycopg2
from Database.Utils import gen_uuid
from Database.Models.Model import Model
from darts.models.forecasting.forecasting_model import ForecastingModel

class ModelRepository:
    def __init__(self, db: DbConnection):
        self.db = db

    def get_all_models_by_service(self, serviceId:UUID) -> list[Model]:
        rows = self.db.execute_get('SELECT id, name, bin from models WHERE "serviceid" = %s;', [str(serviceId)])
        if len(rows) == 0:
            raise psycopg2.DatabaseError
        
        return [Model(UUID(row[0]), row[1], pickle.loads(row[2]), serviceId) for row in rows if loadable(row[2])]

    def get_by_modelname_and_service(self, modelName:str, serviceId:UUID) -> Model:
        rows = self.db.execute_get('SELECT id, name, bin FROM models WHERE "Name" = %s AND "ServiceId" = %s ORDER BY "trainedat" ASC LIMIT 1;', [modelName, str(serviceId)])
        if len(rows) > 0:
            row = rows[0]
            modelObj = pickle.loads(row[2])
            return Model(UUID(row[0]), row[1], modelObj, serviceId)
        raise psycopg2.DatabaseError
    
    def get_by_modelid_and_service(self, modelId:UUID, serviceId:UUID) -> Model:
        rows = self.db.execute_get('SELECT id, name, bin FROM models WHERE "Id" = %s AND "ServiceId" = %s ORDER BY "trainedat" ASC LIMIT 1;', [str(modelId), str(serviceId)])
        if len(rows) > 0:
            row = rows[0]
            modelObj = pickle.loads(row[2])
            return Model(UUID(row[0]), row[1], modelObj, serviceId)
        raise psycopg2.DatabaseError

    def get_all_models(self) -> list[UUID]:
        return [UUID(row[0]) for row in self.db.execute_get('SELECT id from models')]

    def insert_model(self, model:Model):
        self.db.execute('INSERT INTO models ("id", "name", "bin", "trainedat", "serviceid", "scaler") VALUES (%s, %s, %s, %s, %s, %s)', [str(gen_uuid()), type(model.model).__name__, model.get_binary(), model.trainedTime, str(model.serviceId), model.scaler])

    def upsert_model(self, model:Model) -> None:
        self.db.execute("UPDATE models SET bin = %s, trainedat = %s", [model.get_binary(), datetime.now()])

def loadable(model: bytes):
    try:
        pickle.loads(model)
        return True
    except Exception as e:
        print(f"Model {model} is not loadable ({e})")
        return False

