from datetime import timedelta
from multiprocessing import Process
import traceback
from uuid import UUID
from darts import TimeSeries
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

class Forecaster:
    def __init__(self, service_id:UUID, model_repository:ModelRepository, forecast_repository:ForecastRepository, settings_repository:SettingsRepository) -> None:
        self.id = service_id
        self.model_repository = model_repository
        self.forecast_repository = forecast_repository
        self.settings_repository = settings_repository

    def predict(self, series:Historical | None, period:int):
        self._process = Process(target=self._predict, args=[load_historical_data(series, period) if series else None, period])
        self._process.start()

    def _predict(self, series:TimeSeries | None, period:int) -> Forecast:
        models = self.model_repository.get_all_models_by_service(self.id)
        forecasts:list[Forecast] = []

        for i, model in enumerate(models):
            try:
                forecast = model.model.predict(period)
                print("Created forecast")
                if series:
                    forecast_rmse = rmse(series, forecast)
                    print("Calculated RMSE")
                else:
                    forecast_rmse = i
                forecasts.append(Forecast(model.modelId, forecast, forecast_rmse))
                print("saved forecast for comparison...")
            except Exception as e:
                traceback.print_exc()
                print(f"Model {model.name} failed, continuing no next model: {e}")

        print(f"Forecasts count: {len(forecasts)}")
        forecast = min(forecasts, key=lambda x: x.error)
        self.forecast_repository.upsert_forecast(forecast, self.id)

        return forecast

