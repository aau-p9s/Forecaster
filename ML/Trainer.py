

from multiprocessing import Process
from uuid import UUID

from darts import TimeSeries
from darts.models.forecasting.forecasting_model import ForecastingModel

from Database.ForecastRepository import ForecastRepository
from Database.ModelRepository import ModelRepository
from Database.SettingsRepository import SettingsRepository
from Database.Models.Historical import Historical
from Database.Models.Model import Model
from ML.Darts.Utils.preprocessing import load_historical_data, load_json_data, run_transformer_pipeline
from ML.Forecaster import Forecaster
import multiprocessing as mp


class Trainer:
    def __init__(self, service_id:UUID, model_repository:ModelRepository, forecast_repository:ForecastRepository, settings_repository:SettingsRepository) -> None:
        self.id:UUID = service_id
        self.model_repository:ModelRepository = model_repository
        self.forecast_repository:ForecastRepository = forecast_repository
        self.settings_repository:SettingsRepository = settings_repository
        self.forecaster = Forecaster(service_id, model_repository, forecast_repository, settings_repository)

    def train(self, series:Historical, horizon:int) -> None:
        self._process:Process = Process(target=self._train, args=[series, horizon])
        self._process.start()

    def _train(self, data:Historical, horizon:int) -> None:
        settings = self.settings_repository.get_settings(self.id)
        period = settings.scale_period
        series:TimeSeries = load_historical_data(data, period)
        preprocessed_series, missing_value_ratio, scaler = run_transformer_pipeline(series)
        train_series, validation_series = preprocessed_series.split_after(.75)
        print(f"preprocessed_series length: {len(preprocessed_series)}")
        print(f"mssing_value_ratio:         {missing_value_ratio}")
        print(f"scaler:                     {scaler}")

        models = self.model_repository.get_all_models_by_service(self.id)

        with mp.Pool(4) as p:
            p.map(self.train_one, [(model, train_series) for model in models])

        self.forecaster._predict(validation_series, horizon)

    def train_one(self, args:tuple[Model, TimeSeries]) -> None:
        model, series = args
        try:
            fitted_model = model.model.fit(series)
            print("Fitted model")
            self.model_repository.upsert_model(Model(model.modelId, model.name, fitted_model, model.serviceId, model.scaler))
            print("Saved model")
        except Exception as e:
            print(e)
