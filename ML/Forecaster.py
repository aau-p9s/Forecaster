from pandas import Timedelta
from Database.Models.Forecast import Forecast
from Database.Models.Historical import Historical
from ML.Darts.Training.predict import predict_all
from ML.Darts.Utils.MLManager import MLManager

class Forecaster(MLManager):
    def _run(self, historical:Historical, horizon:Timedelta, gpu_id : int = 0) -> Forecast:
        self.busy()
        _, series, scaler = self.preprocess(historical, horizon)
        settings = self.settings_repository.get_settings(self.service_id)

        models = self.model_repository.get_all_models_by_service(self.service_id, gpu_id)
        forecasts:list[Forecast] = predict_all(models, series, scaler, settings.scale_period, horizon)

        print(f"Forecasts count: {len(forecasts)}", flush=True)
        forecast = min(forecasts, key=lambda x: x.error)
        self.forecast_repository.upsert_forecast(forecast, self.service_id)
        self.idle()

        return forecast
