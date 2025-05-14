

from multiprocessing import Process
from uuid import UUID

from darts import TimeSeries
from darts.metrics import rmse

from Database.ForecastRepository import ForecastRepository
from Database.ModelRepository import ModelRepository
from Database.Models.Forecast import Forecast
from Database.Models.Historical import Historical
from ML.Darts.Utils.preprocessing import load_historical_data, load_json_data, run_transformer_pipeline


class Trainer():
    def __init__(self, service_id:UUID, model_repository:ModelRepository, forecast_repository:ForecastRepository) -> None:
        self.id:UUID = service_id
        self.model_repository:ModelRepository = model_repository
        self.forecast_repository:ForecastRepository = forecast_repository

    def train(self, series:list[Historical], horizon:int) -> None:
        self._process:Process = Process(target=self._train, args=[series, horizon])
        self._process.start()

    def _train(self, data:list[Historical], horizon:int) -> None:
        series:TimeSeries = load_historical_data(data[0])
        preprocessed_series, missing_value_ratio, scaler = run_transformer_pipeline(series)
        train_series, validation_series = preprocessed_series.split_after(.75)
        print(f"preprocessed_series length: {len(preprocessed_series)}")
        print(f"mssing_value_ratio:         {missing_value_ratio}")
        print(f"scaler:                     {scaler}")

        models = self.model_repository.get_all_models_by_service(self.id)
        forecasts:list[Forecast] = []
        for model in models:
            try:
                fitted_model = model.model.fit(train_series)
                print("Fitted model")
                forecast = fitted_model.predict(int(horizon))
                print("Created forecast")
                validation_target = validation_series[:len(forecast)]
                forecast_rmse = rmse(validation_target, forecast)
                print(f"rmse: {forecast_rmse}")
                forecast = Forecast(model.modelId, forecast, forecast_rmse)
                forecasts.append(forecast)
            except Exception as e:
                raise e

        forecast = forecasts[max([forecast.error for forecast in forecasts])]
        self.forecast_repository.insert_forecast(forecast, self.id)
