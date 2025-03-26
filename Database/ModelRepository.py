from Database.dbhandler import DbConnection
import psycopg2
from Database.Utils import gen_uuid
from Database.Models.Model import Model

class ModelRepository:
    def __init__(self, db: DbConnection):
        self.db = db

    def get_all_models_by_service(self, serviceId) -> list[Model]:
        rows = self.db.execute_query('SELECT id, name, bin from models WHERE "serviceid" = %s;', [serviceId])
        return [Model(row[0], row[1], row[2], serviceId) for row in rows]

    def get_by_modelname_and_service(self, modelname, serviceId) -> Model:
        rows = self.db.execute_query('SELECT id, name, bin FROM models WHERE "Name" = %s AND "ServiceId" = %s ORDER BY "trainedat" ASC LIMIT 1;', [modelname, serviceId])
        if len(rows) > 0:
            row = rows[0]
            return Model(row[0], row[1], row[2], serviceId)
        raise psycopg2.DatabaseError

    def get_all_models(self) -> list[Model]:
        return [row[0] for row in self.db.execute_query('SELECT id from models')]

    def insert_model(self, modelname, modelpath, trainedTime, serviceId) -> None:
        with open(modelpath, "rb") as file:
            binary_data = file.read()
        query = 'INSERT INTO models ("id", "name", "bin", "trainedat", "serviceid") VALUES (%s, %s, %s, %s, %s) RETURNING *; '
        params = [gen_uuid(), modelname, psycopg2.Binary(binary_data), trainedTime, serviceId]
        self.db.execute_query(query, params)
