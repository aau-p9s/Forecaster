import os
import tempfile
from uuid import UUID
from darts.dataprocessing.transformers import Scaler
from pandas import Timedelta
from datetime import date, datetime
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from darts.models.forecasting.forecasting_model import ForecastingModel

from Database.Entities.Entity import Entity
from ML.Darts.Utils.load_model import load_model
from Utils.variables import temporary_directory

class Model(Entity[str, str, bytes, bytes|None, datetime]):
    def __init__(self, name: str, service_id:UUID, model:ForecastingModel, trained_at: datetime = datetime.now(), scaler=Scaler(MinMaxScaler(feature_range=(0, 1)))):
        self.model:ForecastingModel = model
        self.name = name if name is not None else model.__class__.__name__
        self.trained_at = trained_at
        self.service_id = service_id
        self.scaler = scaler
        super().__init__()

    def get_binary(self) -> tuple[bytes, bytes|None]:
        with tempfile.TemporaryDirectory(dir=temporary_directory) as directory:
            self.model.save(f"{directory}/model.pth")
            with open(f"{directory}/model.pth", "rb") as file:
                model_bin = file.read()
            if os.path.exists(f"{directory}/model.pth.ckpt"):
                with open(f"{directory}/model.pth.ckpt", "rb") as file:
                    ckpt_bin = file.read()
            else:
                ckpt_bin = None
            return model_bin, ckpt_bin
    
    def get_trained_frequency(self, default: Timedelta) -> Timedelta:
        if self.model.training_series is not None:
            frequency = self.model.training_series.freq
            if isinstance(frequency, int):
                raise ValueError("fuck pandas")
            fixed_frequency = f"{frequency.n}{frequency.name}"
            return pd.to_timedelta(fixed_frequency)
        else:
            return default

    @staticmethod
    def from_row(id, name, service_id, model, ckpt, trained_at):
        return Model(name, UUID(service_id), load_model(name, model, ckpt), trained_at).with_id(UUID(id))

    def to_row(self):
        model_bin, ckpt_bin = self.get_binary()
        return str(self.id), self.name, str(self.service_id), model_bin, ckpt_bin, self.trained_at
