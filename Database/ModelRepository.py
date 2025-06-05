from datetime import datetime
from pickle import UnpicklingError
import tempfile
from uuid import UUID
import cloudpickle as pickle
from darts.models.forecasting.forecasting_model import ForecastingModel
from Database.dbhandler import DbConnection
import psycopg2
from Database.Utils import gen_uuid
from Database.Models.Model import Model
import torch
from darts.utils.likelihood_models import GaussianLikelihood
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
import traceback

from Utils.getEnv import getEnv

enable_gpu = getEnv("FORECASTER__ENABLE__GPU", "1") == "1"

class PositiveGaussianLikelihood(GaussianLikelihood):
    def forward(self, *args, **kwargs):
        result = super().forward(*args, **kwargs)

        # Ensure that std is always non-negative before taking the exp
        if torch.any(result.std < 0):
            result.std = torch.abs(result.std)  # Take absolute value to prevent negative std

        result.std = torch.exp(result.std)  # Ensures a positive std after exp
        return result

class ModelRepository:
    def __init__(self, db: DbConnection):
        self.db = db

    def get_all_models_by_service(self, serviceId:UUID) -> list[Model]:
        rows = self.db.execute_get('SELECT id, name, bin, ckpt from models WHERE "serviceid" = %s;', [str(serviceId)])
        if len(rows) == 0:
            raise psycopg2.DatabaseError
        
        models = []
        for row in rows:
            try:
                models.append(Model(UUID(row[0]), row[1], load_model(row[1], row[2], row[3]), serviceId))
            except UnpicklingError as e:
                print(f"Model {row[1]} failed to load {e}")
        return models

    def get_by_modelname_and_service(self, modelName:str, serviceId:UUID) -> Model:
        rows = self.db.execute_get('SELECT id, name, bin, ckpt FROM models WHERE "Name" = %s AND "ServiceId" = %s ORDER BY "trainedat" ASC LIMIT 1;', [modelName, str(serviceId)])
        if len(rows) > 0:
            row = rows[0]
            modelObj = load_model(row[1], row[2], row[3])
            return Model(UUID(row[0]), row[1], modelObj, serviceId)
        raise psycopg2.DatabaseError
    
    def get_by_modelid_and_service(self, modelId:UUID, serviceId:UUID) -> Model:
        rows = self.db.execute_get('SELECT id, name, bin, ckpt FROM models WHERE "Id" = %s AND "ServiceId" = %s ORDER BY "trainedat" ASC LIMIT 1;', [str(modelId), str(serviceId)])
        if len(rows) > 0:
            row = rows[0]
            modelObj = load_model(row[1], row[2], row[3])
            return Model(UUID(row[0]), row[1], modelObj, serviceId)
        raise psycopg2.DatabaseError

    def get_all_models(self) -> list[UUID]:
        return [UUID(row[0]) for row in self.db.execute_get('SELECT id from models')]

    def insert_model(self, model:Model):
        self.db.execute('INSERT INTO models ("id", "name", "bin", "trainedat", "serviceid", "scaler") VALUES (%s, %s, %s, %s, %s, %s)', [str(gen_uuid()), type(model.model).__name__, model.get_binary(), model.trainedTime, str(model.serviceId), model.scaler])

    def upsert_model(self, model:Model) -> None:
        self.db.execute("UPDATE models SET bin = %s, trainedat = %s where id = %s", [model.get_binary(), datetime.now(), str(model.modelId)])

def load_model(name: str, data: bytes, ckpt: bytes|None = None) -> ForecastingModel:
    with tempfile.TemporaryDirectory() as directory:
        if ckpt is not None:
            with open(f"{directory}/{name}.pth.ckpt", "wb") as file:
                file.write(ckpt)
        with open(f"{directory}/{name}.pth", "wb") as file:
            file.write(data)
        try:
            model = TorchForecastingModel.load(f"{directory}/{name}.pth", map_location= 'cuda' if enable_gpu else "cpu")
            if not enable_gpu:
                model.to_cpu()
        except Exception as e1:
            try:
                model = ForecastingModel.load(f"{directory}/{name}.pth")
            except Exception as e2:
                traceback.print_exception(e1)
                traceback.print_exception(e2)
                raise UnpicklingError
        return model 

