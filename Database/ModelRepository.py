from datetime import datetime
from pickle import UnpicklingError
from uuid import UUID
from Database.Entities.Model import Model
from Database.Repository import Repository
from Database.dbhandler import DbConnection
import psycopg2
import traceback

from ML.Darts.Utils.load_model import load_model

class ModelRepository(Repository[Model]):
    _class = Model
    def __init__(self, db: DbConnection):
        self.db = db

    def get_all_models_by_service(self, service_id:UUID, gpu_id: int = 0) -> list[Model]:
        rows = self.db.execute_get('SELECT id, name, bin, ckpt, trainedat from models WHERE "serviceid" = %s;', [str(service_id)])
        if len(rows) == 0:
            raise psycopg2.DatabaseError
        
        models = []
        for row in rows:
            try:
                models.append(Model(row[1], service_id, load_model(row[1], row[2], row[3], gpu_id = gpu_id), row[4]).with_id(UUID(row[0])))
            except UnpicklingError as e:
                traceback.print_exception(e)
                print(f"Model {row[1]} failed to load {e}", flush=True)
        return models

    def upsert_model(self, model:Model) -> None:
        model_bin, ckpt_bin = model.get_binary()
        self.db.execute("UPDATE models SET bin = %s, trainedat = %s, ckpt = %s where id = %s", [model_bin, datetime.now(), ckpt_bin, str(model.id)])

    def table_name(self) -> str:
        return super().table_name() + "s"
