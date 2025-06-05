

from multiprocessing import Process
from multiprocessing.managers import DictProxy
from time import time
import traceback
from uuid import UUID

from darts import TimeSeries
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel

from Database.ForecastRepository import ForecastRepository
from Database.ModelRepository import ModelRepository
from Database.SettingsRepository import SettingsRepository
from Database.Models.Historical import Historical
from Database.Models.Model import Model
from ML.Darts.Utils.preprocessing import load_historical_data, run_transformer_pipeline
from ML.Darts.Utils.timeout import timeout
from ML.Forecaster import Forecaster
import multiprocessing as mp


class Trainer:
    manager = mp.Manager()
    def __init__(self, service_id:UUID, model_repository:ModelRepository, forecast_repository:ForecastRepository, settings_repository:SettingsRepository, gpu_id: int = 0) -> None:
        self.id:UUID = service_id
        self.model_repository:ModelRepository = model_repository
        self.forecast_repository:ForecastRepository = forecast_repository
        self.settings_repository:SettingsRepository = settings_repository
        self.forecaster = Forecaster(service_id, model_repository, forecast_repository, settings_repository)
        self.gpu_id = gpu_id
        self.model_status = self.manager.dict()
        self.status = self.manager.Value(str, "Idle")

    def train(self, series:Historical, horizon:int) -> None:
        self.model_status.clear()
        self._process:Process = Process(target=self._train, args=[series, horizon])
        self._process.start()

    def _train(self, data:Historical, period:int) -> None:
        self.status.set("Busy")
        series:TimeSeries = load_historical_data(data, period)
        preprocessed_series, missing_value_ratio, scaler = run_transformer_pipeline(series)
        train_series, validation_series = preprocessed_series.split_after(.80)
        print(f"preprocessed_series length: {len(preprocessed_series)}", flush=True)
        print(f"mssing_value_ratio:         {missing_value_ratio}", flush=True)
        print(f"scaler:                     {scaler}", flush=True)

        models = self.model_repository.get_all_models_by_service(self.id, gpu_id=self.gpu_id)
        for model in models:
            self.model_status[model.name] = self.manager.dict({ "message": "waiting", "error": None, "start_time": None, "end_time": None })

        fitted_models = []
        for model in models:
            self.model_status[model.name]["message"] = "working"
            fitted_models.append(train_model(model, train_series.copy(), self.model_status))
            self.model_status[model.name]["message"] = "finished"
            self.model_status[model.name]["end_time"] = time()

        print("Finished training", flush=True)
        self.forecaster._predict(validation_series, period)
        self.status.set("Idle")

        for fitted_model in fitted_models:
            if fitted_model is None:
                continue
            self.model_status[fitted_model.name]["message"] = "saving"
            print("Saving model...", flush=True)
            print(f"something about model: {fitted_model.model}")
            self.model_repository.upsert_model(fitted_model)
            self.model_status[fitted_model.name]["message"] = "saved"

@timeout()
def train_model(model: Model, series: TimeSeries, model_status: DictProxy) -> Model | None:
    try:
        print(f"Training {model.name}", flush=True)
        model_status[model.name]["start_time"] = time()
        if isinstance(model.model, TorchForecastingModel):
            fitted_model = model.model.fit(series, dataloader_kwargs={ "num_workers": 12 })
        else:
            fitted_model = model.model.fit(series)
        return Model(model.modelId, model.name, fitted_model, model.serviceId, model.scaler)

    except Exception as e:
        model_status[model.name]["end_time"] = time()
        model_status[model.name]["message"] = "failed"
        model_status[model.name]["error"] = f"{e}"
        traceback.print_exc()
        return None
