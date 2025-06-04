from multiprocessing import Process
import traceback
from uuid import UUID
from darts import TimeSeries
from darts.metrics.metrics import METRIC_OUTPUT_TYPE
from darts.models.forecasting.forecasting_model import ForecastingModel
from Database.ForecastRepository import ForecastRepository
from Database.HistoricalRepository import HistoricalRepository
from Database.ModelRepository import ModelRepository
from darts.metrics import rmse
from Database.Models.Forecast import Forecast
from Database.Models.Historical import Historical
from Database.Models.Model import Model
from sklearn.preprocessing import MinMaxScaler
from Database.SettingsRepository import SettingsRepository
from ML.Darts.Utils.preprocessing import Scaler, load_historical_data
import multiprocessing as mp

class Forecaster:
    def __init__(self, service_id:UUID, model_repository:ModelRepository, forecast_repository:ForecastRepository, settings_repository:SettingsRepository) -> None:
        self.id = service_id
        self.model_repository = model_repository
        self.forecast_repository = forecast_repository
        self.settings_repository = settings_repository

    def predict(self, series:Historical | None, horizon:int):
        settings = self.settings_repository.get_settings(self.id)
        period = settings.scale_period
        self._process = Process(target=self._predict, args=[load_historical_data(series, period) if series else None, horizon])
        self._process.start()

    def _predict(self, series:TimeSeries | None, horizon:int) -> Forecast:
        models = self.model_repository.get_all_models_by_service(self.id)

        with mp.Pool(4) as p:
            forecasts = p.map(self.predict_once, [(model, i, series, horizon) for i, model in enumerate(models)])

        print(f"Forecasts count: {len(forecasts)}")
        forecast = min(forecasts, key=lambda x: x is not None and x.error)

        self.forecast_repository.upsert_forecast(forecast, self.id)
        return forecast

    def predict_once(self, args:tuple[Model, int, TimeSeries | None, int]) -> Forecast | None:
        model, i, series, horizon = args
        try:
            forecast = model.model.predict(horizon)
            print("Created forecast")
            if series:
                forecast_rmse = rmse(series, forecast)
                print("Calculated RMSE")
            else:
                forecast_rmse = i
            return Forecast(model.modelId, forecast, forecast_rmse)

            print("saved forecast for comparison...")
        except Exception as e:
            print(f"Model failed, continuing no next model: {e}")
            return None
