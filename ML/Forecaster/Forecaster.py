from pandas import Timedelta
from Database.Entities.Forecast import Forecast
from Database.Entities.Historical import Historical
from ML.Darts.Utils.MLManager import MLManager
from ML.Forecaster.predict import predict_all, validate_model

class Forecaster(MLManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total = self.manager.Value(int, 0)
        self.finished = self.manager.Value(int, 0)

    def _run(self, historical:Historical, horizon:Timedelta, gpu_id : int = 0) -> Forecast:
        self.busy()
        _, series, scaler = self.preprocess(historical, horizon)
        settings = self.settings_repository.get_settings(self.service_id)

        models = self.model_repository.get_all_models_by_service(self.service_id, gpu_id)
        self.total.set(len(models))
        self.finished.set(0)
        valid_models = list(filter(validate_model, models))
        forecasts:list[Forecast] = predict_all(valid_models, series, scaler, settings.scale_period, horizon, self.finished, self.get_cores())

        print(f"Forecasts count: {len(forecasts)}", flush=True)
        forecast = min(forecasts, key=lambda x: x.error)
        self.forecast_repository.upsert_forecast(forecast)
        self.idle()

        return forecast
