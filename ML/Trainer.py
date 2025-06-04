

from datetime import timedelta
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
from ML.Darts.Utils.timeout import timeout
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

    def _train(self, data:Historical, period:int) -> None:
        settings = self.settings_repository.get_settings(self.id)
        series:TimeSeries = load_historical_data(data, period)
        preprocessed_series, missing_value_ratio, scaler = run_transformer_pipeline(series)
        train_series, validation_series = preprocessed_series.split_after(.75)
        print(f"preprocessed_series length: {len(preprocessed_series)}", flush=True)
        print(f"mssing_value_ratio:         {missing_value_ratio}", flush=True)
        print(f"scaler:                     {scaler}", flush=True)

        models = self.model_repository.get_all_models_by_service(self.id)

        with mp.Pool(4) as p:
            fitted_models = p.map(train_one, [(model, train_series) for model in models])

        successfully_fitted_models = list(filter(lambda model: model is not None, fitted_models))
        for model in successfully_fitted_models:
            if model is not None:
                self.model_repository.upsert_model(model)

        print(f"Successfully fitted {len(successfully_fitted_models)}", flush=True)

        self.forecaster._predict(validation_series, period)

def train_one(args:tuple[Model, TimeSeries]) -> Model | None:
    model, series = args
    print(f"Training model: {model.name}", flush=True)
    try:
        fitted_model = fit(model.model.fit, series)
        print("Fitted model", flush=True)
        print("Saved model", flush=True)
        return Model(model.modelId, model.name, fitted_model, model.serviceId, model.scaler)
    except Exception as e:
        print(e)
        return None



@timeout
def fit(fit_method, arg):
    return fit_method(arg)
