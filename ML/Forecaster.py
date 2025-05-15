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
from ML.Darts.Utils.preprocessing import Scaler, load_historical_data

class Forecaster:
    def __init__(self, service_id:UUID, model_repository:ModelRepository, forecast_repository:ForecastRepository) -> None:
        self.id = service_id
        self.model_repository = model_repository
        self.forecast_repository = forecast_repository

    def predict(self, series:Historical | None, horizon:int):
        self._process = Process(target=self._predict, args=[load_historical_data(series) if series else None, horizon])
        self._process.start()

    def _predict(self, series:TimeSeries | None, horizon:int):
        models = self.model_repository.get_all_models_by_service(self.id)
        forecasts:list[Forecast] = []

        for i, model in enumerate(models):
            try:
                forecast = model.model.predict(horizon)
                print("Created forecast")
                if series:
                    forecast_rmse = rmse(series, forecast)
                    print("Calculated RMSE")
                else:
                    forecast_rmse = i
                forecasts.append(Forecast(model.modelId, forecast, forecast_rmse))
            except Exception as e:
                print(f"Model failed, continuing no next model: {e}")

        forecast = list(filter(lambda forecast: forecast.error == min(forecast.error for forecast in forecasts), forecasts))[0]
        self.forecast_repository.insert_forecast(forecast, self.id)

