from datetime import datetime
from multiprocessing import Pool, cpu_count
from multiprocessing.managers import ValueProxy
import traceback
from darts.dataprocessing.transformers.scaler import Scaler
from darts.metrics.metrics import rmse
from darts.timeseries import TimeSeries
from pandas import Timedelta
import pandas as pd

from Database.Entities.Forecast import Forecast
from Database.Entities.Model import Model
from ML.Darts.Utils.preprocessing import unscaling_pipeline
from ML.Darts.Utils.timeout import timeout
from Utils.variables import service_repository

def predict(model: Model, series: TimeSeries, scaler: Scaler, period: Timedelta, horizon: Timedelta, finished: ValueProxy):
    try:
        print(f"Predicting for period: {period} seconds", flush=True)

        # scale to seconds, and scale slightly more to adjust for the validation series during training
        trained_frequency = model.get_trained_frequency(default=pd.to_timedelta("1m"))
        print(f"Trained frequency: {trained_frequency}")
        prediction_period = int(period / trained_frequency) * 2
        forecast = timeout(model.model.predict, prediction_period)

        print("Created forecast", flush=True)
        forecast_rmse = rmse(series, forecast)
        print("Calculated RMSE", flush=True)
        unscaled_forecast = unscaling_pipeline(forecast, scaler, horizon)
        print(f"Pipeline output: length {len(unscaled_forecast)} for period {unscaled_forecast.time_index[0]} to {unscaled_forecast.time_index[-1]}", flush=True)
        finished.set(finished.get()+1)
        if not isinstance(forecast_rmse, float):
            raise ValueError(f"Error, rmse is wrong type: {type(forecast_rmse).__name__}")
        return Forecast(model.service_id, datetime.now(), model.id, unscaled_forecast, False, forecast_rmse)
    except Exception as e:
        traceback.print_exc()
        print(f"Model {model.name} failed, continuing to next model: {e}", flush=True)


    return timeout(model.model.predict, int(period.total_seconds()))

def predict_all(models: list[Model], series: TimeSeries, scaler: Scaler, period: Timedelta, horizon: Timedelta, finished: ValueProxy):
    service_count = len(list(filter(lambda service: service.autoscaling_enabled, service_repository.all())))
    with Pool(int(cpu_count()/(service_count*2))) as pool:
        predictions = pool.starmap(predict, [(model, series, scaler, period, horizon, finished) for model in models])
    successful_predictions = []
    for prediction in predictions:
        if prediction is not None:
            successful_predictions.append(prediction)
    return successful_predictions
