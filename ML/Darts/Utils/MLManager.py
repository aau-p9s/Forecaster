from multiprocessing import Process
import multiprocessing as mp
from uuid import UUID

from darts import TimeSeries
from darts.dataprocessing.transformers.scaler import Scaler
from pandas import Timedelta

from Database.Entities.Historical import Historical
from Database.ForecastRepository import ForecastRepository
from Database.ModelRepository import ModelRepository
from Database.SettingsRepository import SettingsRepository
from ML.Darts.Utils.preprocessing import load_historical_data, run_transformer_pipeline

class MLManager:
    manager = mp.Manager()

    def idle(self):
        self.status.set("Idle")

    def busy(self, message: str = "Busy"):
        self.status.set(message)

    def is_idle(self):
        return self.status.get() == "Idle"

    def __init__(self, service_id:UUID, model_repository:ModelRepository, forecast_repository:ForecastRepository, settings_repository:SettingsRepository) -> None:
        self.service_id = service_id
        self.model_repository = model_repository
        self.forecast_repository = forecast_repository
        self.settings_repository = settings_repository
        self.status = self.manager.Value(str, "Busy")
        self._process: Process


    def run(self, series:Historical | None, horizon:Timedelta, gpu_id: int = 0) -> None:
        if series is not None:
            self._process = Process(target=self._run, args=[series, horizon, gpu_id])
        else:
            self._process = Process(target=self._run_plain, args=[horizon, gpu_id])
        self._process.start()

    def _run(self, historical:Historical, horizon: Timedelta, gpu_id: int = 0):
        print(f"Historical id: {historical.id}, period: {horizon}, gpu_id: {gpu_id}")
        raise NotImplementedError("Error, this method should not be called in the superclass")

    def _run_plain(self, horizon: Timedelta, gpu_id: int = 0):
        print(f"period: {horizon}, gpu_id: {gpu_id}")
        raise NotImplementedError("Error, this method should not be called in the superclass")

    def preprocess(self, historical:Historical, period: Timedelta) -> tuple[TimeSeries, TimeSeries, Scaler]:
        series:TimeSeries = load_historical_data(historical, period)
        preprocessed_series, missing_value_ratio, scaler = run_transformer_pipeline(series)
        train_series, validation_series = preprocessed_series.split_after(.80)
        print(f"preprocessed_series length: {len(preprocessed_series)}", flush=True)
        print(f"mssing_value_ratio:         {missing_value_ratio}", flush=True)
        print(f"scaler:                     {scaler}", flush=True)
        return train_series, validation_series, scaler
